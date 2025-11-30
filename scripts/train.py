"""
Unified training script for both latent CoT and vanilla CoT.

Supports distributed training with PyTorch DDP (tested with 2 GPUs).

Usage:
    # Latent CoT training (loss on answer only)
    torchrun --nproc_per_node=2 scripts/train.py configs/gpt2_baseline.yaml
    
    # Vanilla CoT training (loss on steps + answer)
    torchrun --nproc_per_node=2 scripts/train.py configs/gpt2_vanilla.yaml
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    add_special_tokens,
    GSM8kDataset,
    LatentCollator,
    VanillaCoTDataset,
    VanillaCoTCollator,
    GSM8kDatasetWithSteps,
    LatentCollatorWithSteps,
    CompressionDataset,
    CompressionCollator,
    CurriculumDataset,
    CurriculumCollator,
)
from src.models import LatentGPT2Config, LatentGPT2LMHeadModel, LatentGPT2WithReconstruction, LatentGPT2WithCompression
from src.utils import (
    set_seed,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    save_checkpoint,
    AverageMeter,
    Logger,
)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    config,
    logger,
    use_reconstruction=False,
    use_compression=False,
    gradient_tracker=None,
    tb_writer=None,
):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    answer_loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    compression_loss_meter = AverageMeter()
    
    if is_main_process():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        if use_compression:
            # Compression model returns dict with answer_loss and compression_loss
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None),
                labels=batch['labels'],
                step_input_ids=batch['step_input_ids'],
                step_attention_mask=batch['step_attention_mask'],
                step_positions=batch['step_positions'],
                step_ranges=batch.get('step_ranges', None),  # For token embedding mode
            )
            loss = outputs['loss']
            answer_loss = outputs['answer_loss']
            compression_loss = outputs['compression_loss']
            recon_loss = torch.tensor(0.0)
        elif use_reconstruction:
            # SIM-CoT model returns dict with multiple losses
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None),
                labels=batch['labels'],
                step_labels=batch.get('step_labels', None),
                step_attention_mask=batch.get('step_attention_mask', None),
            )
            loss = outputs['loss']
            answer_loss = outputs['answer_loss']
            recon_loss = outputs['reconstruction_loss']
            compression_loss = torch.tensor(0.0)
        else:
            # Standard model
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None),
                labels=batch['labels'],
            )
            loss = outputs.loss
            answer_loss = loss
            recon_loss = torch.tensor(0.0)
            compression_loss = torch.tensor(0.0)
        
        # Register gradient tracking hooks before backward (if enabled)
        if gradient_tracker is not None:
            # Track latent token hidden states (only last latent)
            if (hasattr(outputs, 'latent_hiddens') and 
                outputs.latent_hiddens is not None and 
                len(outputs.latent_hiddens) > 0):
                gradient_tracker.register_latent_hooks(outputs.latent_hiddens, epoch, step)
            
            # Track context token hidden states (around last latent)
            if (hasattr(outputs, 'context_hiddens') and 
                outputs.context_hiddens is not None and 
                len(outputs.context_hiddens) > 0):
                gradient_tracker.register_context_hooks(outputs.context_hiddens, epoch, step)
            
            # Track model parameter gradients
            # Get num_latents from batch or dataset
            num_latents = 0
            if hasattr(batch, 'num_latents'):
                num_latents = batch.num_latents
            elif 'num_latents' in batch:
                num_latents = batch['num_latents']
            gradient_tracker.register_parameter_hooks(model, epoch, step, num_latents)
        
        # Backward pass
        grad_accum_steps = int(config['gradient_accumulation_steps'])
        loss = loss / grad_accum_steps
        loss.backward()
        
        # Step gradient tracker after backward
        if gradient_tracker is not None:
            gradient_tracker.step(tb_writer)
        
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['max_grad_norm']))
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        loss_meter.update(loss.item() * grad_accum_steps)
        if use_compression:
            answer_loss_meter.update(answer_loss.item())
            compression_loss_meter.update(compression_loss.item())
        elif use_reconstruction:
            answer_loss_meter.update(answer_loss.item())
            recon_loss_meter.update(recon_loss.item())
        
        # Logging
        if is_main_process():
            if step % int(config['logging_steps']) == 0:
                metrics = {
                    'loss': loss_meter.avg,
                    'lr': scheduler.get_last_lr()[0],
                }
                if use_compression:
                    metrics['answer_loss'] = answer_loss_meter.avg
                    metrics['compression_loss'] = compression_loss_meter.avg
                elif use_reconstruction:
                    metrics['answer_loss'] = answer_loss_meter.avg
                    metrics['recon_loss'] = recon_loss_meter.avg
                
                logger.log_metrics(metrics, epoch * len(train_loader) + step, prefix="train")
            
            if use_compression:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'ans': f'{answer_loss_meter.avg:.4f}',
                    'comp': f'{compression_loss_meter.avg:.4f}'
                })
            elif use_reconstruction:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'ans': f'{answer_loss_meter.avg:.4f}',
                    'rec': f'{recon_loss_meter.avg:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg


def evaluate(
    model,
    eval_loader,
    epoch,
    logger,
    use_reconstruction=False,
    use_compression=False,
):
    """Evaluate the model."""
    model.eval()
    loss_meter = AverageMeter()
    answer_loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    compression_loss_meter = AverageMeter()
    
    if is_main_process():
        pbar = tqdm(eval_loader, desc=f"Eval Epoch {epoch}")
    else:
        pbar = eval_loader
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            if use_compression:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch.get('position_ids', None),
                    labels=batch['labels'],
                    step_input_ids=batch['step_input_ids'],
                    step_attention_mask=batch['step_attention_mask'],
                    step_positions=batch['step_positions'],
                    step_ranges=batch.get('step_ranges', None),  # For token embedding mode
                )
                loss = outputs['loss']
                answer_loss = outputs['answer_loss']
                compression_loss = outputs['compression_loss']
                recon_loss = torch.tensor(0.0)
            elif use_reconstruction:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch.get('position_ids', None),
                    labels=batch['labels'],
                    step_labels=batch.get('step_labels', None),
                    step_attention_mask=batch.get('step_attention_mask', None),
                )
                loss = outputs['loss']
                answer_loss = outputs['answer_loss']
                recon_loss = outputs['reconstruction_loss']
                compression_loss = torch.tensor(0.0)
            else:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch.get('position_ids', None),
                    labels=batch['labels'],
                )
                loss = outputs.loss
                answer_loss = loss
                recon_loss = torch.tensor(0.0)
                compression_loss = torch.tensor(0.0)
            
            loss_meter.update(loss.item())
            if use_compression:
                answer_loss_meter.update(answer_loss.item())
                compression_loss_meter.update(compression_loss.item())
            elif use_reconstruction:
                answer_loss_meter.update(answer_loss.item())
                recon_loss_meter.update(recon_loss.item())
            
            if is_main_process():
                if use_compression:
                    pbar.set_postfix({
                        'loss': f'{loss_meter.avg:.4f}',
                        'ans': f'{answer_loss_meter.avg:.4f}',
                        'comp': f'{compression_loss_meter.avg:.4f}'
                    })
                elif use_reconstruction:
                    pbar.set_postfix({
                        'loss': f'{loss_meter.avg:.4f}',
                        'ans': f'{answer_loss_meter.avg:.4f}',
                        'rec': f'{recon_loss_meter.avg:.4f}'
                    })
                else:
                    pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    if is_main_process():
        metrics = {'loss': loss_meter.avg}
        if use_compression:
            metrics['answer_loss'] = answer_loss_meter.avg
            metrics['compression_loss'] = compression_loss_meter.avg
        elif use_reconstruction:
            metrics['answer_loss'] = answer_loss_meter.avg
            metrics['recon_loss'] = recon_loss_meter.avg
        logger.log_metrics(metrics, epoch, prefix="eval")
    
    return loss_meter.avg


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file>")
        print("Example: torchrun --nproc_per_node=2 scripts/train.py configs/gpt2_baseline.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Setup distributed training
    is_distributed = setup_distributed()
    rank = get_rank()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if is_main_process():
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        print(f"\nDistributed training: {is_distributed}")
        if is_distributed:
            print(f"World size: {torch.distributed.get_world_size()}")
            print(f"Rank: {rank}")
    
    # Set seed
    set_seed(config['seed'] + rank)  # Different seed per process
    
    # Initialize logger with TensorBoard
    use_tensorboard = config.get('use_tensorboard', True)
    logger = Logger(config['output_dir'], use_tensorboard=use_tensorboard)
    logger.log(f"Starting training with config: {config_path}")
    
    if use_tensorboard and is_main_process():
        logger.log("TensorBoard logging enabled")
        logger.log(f"View logs with: tensorboard --logdir {config['output_dir']}/tensorboard")
    
    # Determine training type
    training_type = config.get('training_type', 'latent')
    logger.log(f"Training type: {training_type}")
    
    # Determine dtype
    dtype_str = config.get('dtype', 'float32')
    if dtype_str == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype_str == 'float16':
        torch_dtype = torch.float16
    elif dtype_str == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32
        logger.log(f"Warning: Unknown dtype '{dtype_str}', using float32")
    
    logger.log(f"Using dtype: {torch_dtype}")
    
    # Initialize tokenizer
    logger.log("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets and dataloaders based on training type
    max_train_samples = config.get('max_train_samples', None)
    max_eval_samples = config.get('max_eval_samples', 200)
    
    # Check if using reconstruction loss or compression loss
    use_reconstruction = config.get('use_reconstruction', False)
    use_compression = config.get('use_compression', False)
    logger.log(f"Use reconstruction loss: {use_reconstruction}")
    logger.log(f"Use compression loss: {use_compression}")
    
    if training_type == 'curriculum':
        # Curriculum training: progressive latent token increase
        # Datasets will be created dynamically in each epoch based on current stage
        start_id, latent_id, end_id = add_special_tokens(tokenizer)
        logger.log(f"Special tokens: start={start_id}, latent={latent_id}, end={end_id}")
        
        # Get curriculum settings
        max_num_latent = int(config['max_num_latent'])
        epochs_per_stage = int(config['epochs_per_stage'])
        logger.log(f"Curriculum training: max_num_latent={max_num_latent}, epochs_per_stage={epochs_per_stage}")
        logger.log(f"Total stages: {max_num_latent} (stages 0-{max_num_latent-1})")
        logger.log(f"Stage schedule: epoch // {epochs_per_stage}")
        
        # Store curriculum settings for use in training loop
        curriculum_settings = {
            'start_id': start_id,
            'latent_id': latent_id,
            'end_id': end_id,
            'max_num_latent': max_num_latent,
            'epochs_per_stage': epochs_per_stage,
        }
        
        # Placeholder - datasets will be created in training loop
        train_dataset = None
        eval_dataset = None
        collator = None
        
    elif training_type == 'compression':
        # Compression training: requires special dataset with both student and teacher formats
        start_id, latent_id, end_id = add_special_tokens(tokenizer)
        logger.log(f"Special tokens: start={start_id}, latent={latent_id}, end={end_id}")
        logger.log(f"Latent mode: flexible (required for compression)")
        
        # Create compression datasets
        train_dataset = CompressionDataset(
            data_path=config['train_data'],
            tokenizer=tokenizer,
            start_latent_id=start_id,
            latent_id=latent_id,
            end_latent_id=end_id,
            dataset_format=config['dataset_format'],
            max_samples=max_train_samples,
        )
        
        eval_dataset = CompressionDataset(
            data_path=config['valid_data'],
            tokenizer=tokenizer,
            start_latent_id=start_id,
            latent_id=latent_id,
            end_latent_id=end_id,
            dataset_format=config['dataset_format'],
            max_samples=max_eval_samples,
        )
        
        # Create compression collator
        collator = CompressionCollator(tokenizer=tokenizer, latent_id=latent_id)
        
    elif training_type == 'latent':
        # Latent CoT: Add special tokens
        start_id, latent_id, end_id = add_special_tokens(tokenizer)
        logger.log(f"Special tokens: start={start_id}, latent={latent_id}, end={end_id}")
        
        # Get latent mode configuration
        latent_mode = config.get('latent_mode', 'fixed')
        logger.log(f"Latent mode: {latent_mode}")
        if latent_mode == 'fixed':
            logger.log(f"  Using fixed num_latent: {config['num_latent']}")
        else:  # flexible
            logger.log(f"  Using flexible latent tokens based on ground truth steps")
        
        # Create datasets based on whether reconstruction loss is used
        if use_reconstruction:
            # Use dataset with step labels
            train_dataset = GSM8kDatasetWithSteps(
                data_path=config['train_data'],
                tokenizer=tokenizer,
                num_latent=config['num_latent'],
                start_latent_id=start_id,
                latent_id=latent_id,
                end_latent_id=end_id,
                dataset_format=config['dataset_format'],
                max_samples=max_train_samples,
                latent_mode=latent_mode,
            )
            
            eval_dataset = GSM8kDatasetWithSteps(
                data_path=config['valid_data'],
                tokenizer=tokenizer,
                num_latent=config['num_latent'],
                start_latent_id=start_id,
                latent_id=latent_id,
                end_latent_id=end_id,
                dataset_format=config['dataset_format'],
                max_samples=max_eval_samples,
                latent_mode=latent_mode,
            )
            
            # Create collator with step labels
            collator = LatentCollatorWithSteps(tokenizer=tokenizer, latent_id=latent_id)
        else:
            # Use standard dataset without step labels
            train_dataset = GSM8kDataset(
                data_path=config['train_data'],
                tokenizer=tokenizer,
                num_latent=config['num_latent'],
                start_latent_id=start_id,
                latent_id=latent_id,
                end_latent_id=end_id,
                dataset_format=config['dataset_format'],
                max_samples=max_train_samples,
                latent_mode=latent_mode,
            )
            
            eval_dataset = GSM8kDataset(
                data_path=config['valid_data'],
                tokenizer=tokenizer,
                num_latent=config['num_latent'],
                start_latent_id=start_id,
                latent_id=latent_id,
                end_latent_id=end_id,
                dataset_format=config['dataset_format'],
                max_samples=max_eval_samples,
                latent_mode=latent_mode,
            )
            
            # Create standard collator
            collator = LatentCollator(tokenizer=tokenizer, latent_id=latent_id)
        
    else:  # vanilla
        # Vanilla CoT: No special tokens needed
        train_dataset = VanillaCoTDataset(
            data_path=config['train_data'],
            tokenizer=tokenizer,
            dataset_format=config['dataset_format'],
            max_samples=max_train_samples,
        )
        
        eval_dataset = VanillaCoTDataset(
            data_path=config['valid_data'],
            tokenizer=tokenizer,
            dataset_format=config['dataset_format'],
            max_samples=max_eval_samples,
        )
        
        collator = VanillaCoTCollator(tokenizer=tokenizer)
    
    # For curriculum training, dataloaders will be created in each epoch
    if training_type != 'curriculum':
        logger.log(f"Train dataset size: {len(train_dataset)}")
        logger.log(f"Eval dataset size: {len(eval_dataset)}")
        
        # Create samplers for distributed training
        # CRITICAL: drop_last=False to ensure all ranks process same number of batches
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if is_distributed else None
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False) if is_distributed else None
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['per_device_batch_size'],
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
            drop_last=False,  # Keep all samples for DDP synchronization
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['per_device_batch_size'],
            sampler=eval_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
            drop_last=False,  # Keep all samples for DDP synchronization
        )
    else:
        # Placeholder for curriculum
        train_loader = None
        eval_loader = None
    
    # Initialize model
    logger.log("Initializing model...")
    if training_type == 'curriculum':
        # Curriculum training uses standard latent model
        model_config = LatentGPT2Config.from_pretrained(
            config['model_name'],
            latent_token_id=curriculum_settings['latent_id'],
            start_latent_id=curriculum_settings['start_id'],
            end_latent_id=curriculum_settings['end_id'],
            use_projection=config.get('use_projection', False),
            use_layernorm=config.get('use_layernorm', False),
            torch_dtype=torch_dtype,
        )
        
        model = LatentGPT2LMHeadModel.from_pretrained(
            config['model_name'],
            config=model_config,
            torch_dtype=torch_dtype,
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.log("Curriculum model initialized (LatentGPT2LMHeadModel)")
        
    elif training_type == 'compression':
        # Compression model: student (latent) + teacher (vanilla CoT)
        logger.log("Initializing compression model with student and teacher...")
        
        # 1. Create student model (latent)
        student_config = LatentGPT2Config.from_pretrained(
            config['model_name'],
            latent_token_id=latent_id,
            start_latent_id=start_id,
            end_latent_id=end_id,
            use_projection=config.get('use_projection', False),
            use_layernorm=config.get('use_layernorm', False),
            torch_dtype=torch_dtype,
        )
        
        student_model = LatentGPT2LMHeadModel.from_pretrained(
            config['model_name'],
            config=student_config,
            torch_dtype=torch_dtype,
        )
        student_model.resize_token_embeddings(len(tokenizer))
        logger.log("Student model initialized")
        
        # 2. Load teacher model (conditional based on use_teacher_hiddens)
        use_teacher_hiddens = config.get('use_teacher_hiddens', True)
        teacher_model = None
        
        if use_teacher_hiddens:
            teacher_checkpoint_path = config['teacher_checkpoint']
            if not os.path.exists(teacher_checkpoint_path):
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
            
            logger.log(f"Loading teacher model from: {teacher_checkpoint_path}")
            teacher_model = GPT2LMHeadModel.from_pretrained(
                config['model_name'],
                torch_dtype=torch_dtype,
            )
            # Note: DO NOT resize teacher embeddings - it uses vanilla tokenizer (50257 tokens)
            # teacher_model.resize_token_embeddings(len(tokenizer))
            
            # Load teacher checkpoint
            teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')
            teacher_state_dict = teacher_checkpoint.get('model_state_dict', teacher_checkpoint)
            
            # Handle DDP wrapped checkpoint
            if any(k.startswith('module.') for k in teacher_state_dict.keys()):
                teacher_state_dict = {k.replace('module.', ''): v for k, v in teacher_state_dict.items()}
            
            teacher_model.load_state_dict(teacher_state_dict)
            logger.log("Teacher model loaded successfully")
        else:
            logger.log("use_teacher_hiddens=False: Using token embedding averages instead of teacher model")
        
        # 3. Create compression model
        compression_weight = float(config.get('compression_weight', 1.0))
        freeze_teacher = config.get('freeze_teacher', True)
        freeze_student = config.get('freeze_student', False)
        
        logger.log("Creating compression model wrapper")
        logger.log(f"  compression_weight: {compression_weight}")
        logger.log(f"  use_teacher_hiddens: {use_teacher_hiddens}")
        logger.log(f"  freeze_teacher: {freeze_teacher}")
        logger.log(f"  freeze_student: {freeze_student}")
        
        model = LatentGPT2WithCompression(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            compression_weight=compression_weight,
            latent_token_id=latent_id,
            freeze_teacher=freeze_teacher,
            freeze_student=freeze_student,
            use_teacher_hiddens=use_teacher_hiddens,
        )
        
    elif training_type == 'latent':
        # Latent model with special token processing
        # Create latent config (with original vocab size first)
        model_config = LatentGPT2Config.from_pretrained(
            config['model_name'],
            latent_token_id=latent_id,
            start_latent_id=start_id,
            end_latent_id=end_id,
            use_projection=config['use_projection'],
            use_layernorm=config['use_layernorm'],
            torch_dtype=torch_dtype,
        )
        
        # Initialize base latent model from pretrained
        base_model = LatentGPT2LMHeadModel.from_pretrained(
            config['model_name'],
            config=model_config,
            torch_dtype=torch_dtype,
        )
        
        original_vocab_size = base_model.get_input_embeddings().weight.shape[0]
        logger.log(f"Base model initialized with vocab size: {original_vocab_size}")
        
        # Load base checkpoint if specified (for transfer learning from vanilla or curriculum model)
        load_base_checkpoint = config.get('load_base_checkpoint', None)
        load_checkpoint_type = config.get('load_checkpoint_type', 'auto')  # 'base' / 'latent' / 'auto'
        
        # Determine whether to resize embeddings based on checkpoint type
        should_resize_before_load = True
        if load_base_checkpoint and os.path.exists(load_base_checkpoint):
            # Check checkpoint to determine resize strategy
            checkpoint_test = torch.load(load_base_checkpoint, map_location='cpu')
            state_dict_test = checkpoint_test.get('model_state_dict', checkpoint_test)
            if any(k.startswith('module.') for k in state_dict_test.keys()):
                state_dict_test = {k.replace('module.', ''): v for k, v in state_dict_test.items()}
            checkpoint_vocab_size = state_dict_test['transformer.wte.weight'].shape[0]
            
            if load_checkpoint_type == 'base' and checkpoint_vocab_size == 50257:
                # Loading vanilla model (vocab 50257), resize AFTER loading
                should_resize_before_load = False
                logger.log(f"Will load base checkpoint first (vocab {checkpoint_vocab_size}), then resize to {len(tokenizer)}")
            elif load_checkpoint_type == 'latent' or checkpoint_vocab_size == len(tokenizer):
                # Loading latent model, resize BEFORE loading
                should_resize_before_load = True
                logger.log(f"Will resize to {len(tokenizer)} before loading checkpoint")
        
        # Resize embeddings if needed (before loading checkpoint for latent models)
        if should_resize_before_load:
            base_model.resize_token_embeddings(len(tokenizer))
            logger.log(f"Resized token embeddings from {original_vocab_size} to {len(tokenizer)}")
        
        if load_base_checkpoint and os.path.exists(load_base_checkpoint):
            logger.log(f"Loading base model checkpoint: {load_base_checkpoint}")
            logger.log(f"Checkpoint type: {load_checkpoint_type}")
            checkpoint = torch.load(load_base_checkpoint, map_location='cpu')
            
            # Handle DDP wrapped checkpoint
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Get vocab sizes
            checkpoint_vocab_size = state_dict['transformer.wte.weight'].shape[0]
            current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
            logger.log(f"Checkpoint vocab size: {checkpoint_vocab_size}, Current model vocab size: {current_vocab_size}")
            
            # Handle different checkpoint types
            if load_checkpoint_type == 'base':
                # Loading vanilla/base model (vocab should match checkpoint at 50257)
                if checkpoint_vocab_size == current_vocab_size:
                    # Perfect match - direct load
                    logger.log(f"Loading base model checkpoint (vocab {checkpoint_vocab_size})")
                    base_model.load_state_dict(state_dict, strict=True)
                    logger.log("✓ Checkpoint loaded successfully")
                    
                    # Now resize for special tokens if needed
                    if current_vocab_size != len(tokenizer):
                        logger.log(f"Resizing token embeddings from {current_vocab_size} to {len(tokenizer)}")
                        base_model.resize_token_embeddings(len(tokenizer))
                        logger.log(f"✓ New special tokens ({len(tokenizer) - current_vocab_size}) initialized randomly")
                else:
                    logger.log(f"Error: Expected base checkpoint with vocab {current_vocab_size}, got {checkpoint_vocab_size}")
                    raise ValueError(f"Vocab size mismatch: checkpoint has {checkpoint_vocab_size}, model has {current_vocab_size}")
                    
            elif load_checkpoint_type == 'latent':
                # Explicitly loading latent model (vocab 50260)
                if checkpoint_vocab_size == 50260 and current_vocab_size == 50260:
                    logger.log("Loading latent model checkpoint (vocab 50260)")
                    base_model.load_state_dict(state_dict, strict=True)
                    logger.log("✓ Checkpoint loaded successfully (all weights matched)")
                elif checkpoint_vocab_size == 50260 and current_vocab_size == 50257:
                    logger.log("Warning: Loading latent checkpoint into base model - vocab size mismatch")
                    base_model.load_state_dict(state_dict, strict=False)
                else:
                    logger.log(f"Warning: Unexpected vocab size for latent checkpoint: {checkpoint_vocab_size}")
                    base_model.load_state_dict(state_dict, strict=False)
                    
            else:  # auto
                # Auto-detect based on vocab size
                logger.log("Auto-detecting checkpoint type...")
                if checkpoint_vocab_size == current_vocab_size:
                    # Exact match - load all weights
                    base_model.load_state_dict(state_dict, strict=True)
                    logger.log(f"✓ Checkpoint loaded successfully (vocab size: {checkpoint_vocab_size})")
                elif checkpoint_vocab_size == 50257 and current_vocab_size == 50260:
                    # Loading base checkpoint into latent model
                    logger.log(f"Detected base model checkpoint → loading into latent model")
                    base_model.load_state_dict(state_dict, strict=False)
                    logger.log("✓ Checkpoint loaded (new token embeddings initialized randomly)")
                elif checkpoint_vocab_size == 50260 and current_vocab_size == 50260:
                    # Loading latent/curriculum checkpoint into latent model
                    base_model.load_state_dict(state_dict, strict=True)
                    logger.log(f"✓ Latent checkpoint loaded successfully")
                else:
                    logger.log(f"Warning: Vocab size mismatch - checkpoint: {checkpoint_vocab_size}, model: {current_vocab_size}")
                    logger.log("Loading with strict=False - some weights may not match")
                    base_model.load_state_dict(state_dict, strict=False)
        elif load_base_checkpoint:
            logger.log(f"Warning: Base checkpoint not found: {load_base_checkpoint}")
        
        # Wrap with SIM-CoT if using reconstruction loss
        if use_reconstruction:
            use_auxiliary_gpt2 = config.get('use_auxiliary_gpt2', True)
            reconstruction_weight = float(config.get('reconstruction_weight', 1.0))
            freeze_base_model = config.get('freeze_base_model', False)
            freeze_auxiliary = config.get('freeze_auxiliary', False)
            
            logger.log("Wrapping base model with SIM-CoT reconstruction module")
            logger.log(f"  use_auxiliary_gpt2: {use_auxiliary_gpt2}")
            logger.log(f"  reconstruction_weight: {reconstruction_weight}")
            logger.log(f"  freeze_base_model: {freeze_base_model}")
            logger.log(f"  freeze_auxiliary: {freeze_auxiliary}")
            
            model = LatentGPT2WithReconstruction(
                base_model=base_model,
                tokenizer=tokenizer,
                use_auxiliary_gpt2=use_auxiliary_gpt2,
                reconstruction_weight=reconstruction_weight,
                latent_token_id=latent_id,
                eos_token_id=tokenizer.eos_token_id,
                freeze_base_model=freeze_base_model,
                freeze_auxiliary=freeze_auxiliary,
            )
        else:
            # Use base model directly without reconstruction
            model = base_model
        
    else:
        # Standard GPT-2 for vanilla CoT
        model = GPT2LMHeadModel.from_pretrained(
            config['model_name'],
            torch_dtype=torch_dtype,
        )
        model.resize_token_embeddings(len(tokenizer))
    
    model = model.cuda()
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        logger.log("Model wrapped with DDP")
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {num_params:,}")
    logger.log(f"Trainable parameters: {num_trainable:,}")
    
    # Log model info to TensorBoard
    model_info = {
        "num_parameters": num_params,
        "num_trainable_parameters": num_trainable,
        "architecture": f"{training_type}_gpt2",
    }
    logger.log_model_info(model_info)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
    )
    
    # Calculate total steps
    if training_type == 'curriculum':
        # For curriculum, create a temporary dataset to get batch count
        temp_dataset = CurriculumDataset(
            data_path=config['train_data'],
            tokenizer=tokenizer,
            current_stage=0,
            max_num_latent=curriculum_settings['max_num_latent'],
            start_latent_id=curriculum_settings['start_id'],
            latent_id=curriculum_settings['latent_id'],
            end_latent_id=curriculum_settings['end_id'],
            dataset_format=config['dataset_format'],
            max_samples=max_train_samples,
            c_thought=config.get('c_thought', 1),
            force_answer_only=config.get('force_answer_only', False),
        )
        steps_per_epoch = len(temp_dataset) // config['per_device_batch_size']
        if is_distributed:
            steps_per_epoch = steps_per_epoch // torch.distributed.get_world_size()
        total_steps = steps_per_epoch * int(config['max_epochs'])
        del temp_dataset
    else:
        total_steps = len(train_loader) * int(config['max_epochs'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_steps']),
        num_training_steps=total_steps,
    )
    
    logger.log(f"Total training steps: {total_steps}")
    logger.log(f"Warmup steps: {config['warmup_steps']}")
    
    # Load checkpoint if resuming training
    start_epoch = 0
    best_eval_loss = float('inf')
    resume_checkpoint_path = config.get('resume_from_checkpoint', None)
    
    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        logger.log(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        best_eval_loss = checkpoint.get('loss', float('inf'))
        logger.log(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best_eval_loss={best_eval_loss:.4f}")
    elif resume_checkpoint_path is not None:
        logger.log(f"Warning: Checkpoint path specified but not found: {resume_checkpoint_path}")
        logger.log("Starting training from scratch...")
    else:
        logger.log("Starting training from scratch...")
    
    # Initialize gradient tracker (if enabled)
    gradient_tracker = None
    tb_writer = None
    if config.get('track_latent_gradients', False):
        from src.utils.gradient_tracker import LatentGradientTracker
        
        gradient_tracker = LatentGradientTracker(
            output_dir=config['output_dir'],
            training_type=training_type,
            track_context_positions=config.get('track_context_positions', 3),
            log_freq=config.get('gradient_log_freq', 1),
            csv_save_freq=config.get('gradient_csv_save_freq', 100),
            logger=logger,
        )
        
        # Get TensorBoard writer from logger
        if hasattr(logger, 'writer') and logger.writer is not None:
            tb_writer = logger.writer
            logger.log("Gradient tracking enabled with TensorBoard logging")
        else:
            logger.log("Gradient tracking enabled (TensorBoard writer not available)")
    
    # Training loop
    previous_stage = -1  # Track stage changes for optimizer reset
    for epoch in range(start_epoch, int(config['max_epochs'])):
        # Handle optimizer reset for both curriculum and latent baseline
        reset_optimizer_every_epoch = config.get('reset_optimizer_every_epoch', False)
        reset_optimizer_on_stage_change = config.get('reset_optimizer_on_stage_change', False)
        
        should_reset = False
        reset_reason = ""
        
        if training_type == 'curriculum':
            # Calculate current stage based on epoch
            scheduled_stage = epoch // curriculum_settings['epochs_per_stage']
            current_num_latents = min(scheduled_stage + 1, curriculum_settings['max_num_latent'])
            
            # Calculate the epoch threshold for curriculum completion
            curriculum_complete_epoch = curriculum_settings['max_num_latent'] * curriculum_settings['epochs_per_stage']
            
            logger.log("=" * 80)
            logger.log(f"Epoch {epoch}: Curriculum Stage {scheduled_stage} (using {current_num_latents} latent tokens)")
            if epoch >= curriculum_complete_epoch:
                logger.log(f"  Note: Curriculum complete (epoch >= {curriculum_complete_epoch}), continuing training with max latent tokens")
            logger.log("=" * 80)
            
            if reset_optimizer_every_epoch:
                # CoCoNut style: reset every epoch
                should_reset = True
                reset_reason = "every epoch (CoCoNut style)"
            elif reset_optimizer_on_stage_change and scheduled_stage != previous_stage and epoch > 0:
                # Stage-based: reset only on stage change, but only during curriculum period
                if epoch < curriculum_complete_epoch:
                    should_reset = True
                    reset_reason = f"stage changed from {previous_stage} to {scheduled_stage}"
                else:
                    # After curriculum completion, no more reset
                    should_reset = False
            
            previous_stage = scheduled_stage
        elif training_type == 'latent':
            # For latent baseline: simulate stages based on epochs_per_stage (if specified)
            epochs_per_stage = config.get('epochs_per_stage', 3)  # Default to 3 like curriculum
            scheduled_stage = epoch // epochs_per_stage
            
            # Calculate simulated curriculum completion epoch (to match curriculum training)
            num_latent = config.get('num_latent', 6)
            simulated_complete_epoch = num_latent * epochs_per_stage
            
            if reset_optimizer_every_epoch:
                # Reset every epoch
                should_reset = True
                reset_reason = "every epoch"
            elif reset_optimizer_on_stage_change and scheduled_stage != previous_stage and epoch > 0:
                # Reset on simulated stage change, but only during simulated curriculum period
                if epoch < simulated_complete_epoch:
                    should_reset = True
                    reset_reason = f"simulated stage changed from {previous_stage} to {scheduled_stage} (every {epochs_per_stage} epochs)"
                else:
                    # After simulated curriculum completion, no more reset
                    should_reset = False
            
            previous_stage = scheduled_stage
        
        # Execute optimizer reset if needed
        if should_reset:
            logger.log(f"Resetting optimizer state ({reset_reason})")
            # Reset optimizer state (momentum, variance, etc.)
            from collections import defaultdict
            optimizer.state = defaultdict(dict)
            logger.log("Optimizer state reset complete")
        
        # For curriculum training, create datasets for current stage
        if training_type == 'curriculum':
            scheduled_stage = epoch // curriculum_settings['epochs_per_stage']
            current_num_latents = min(scheduled_stage + 1, curriculum_settings['max_num_latent'])
            
            # Create datasets for current stage
            train_dataset = CurriculumDataset(
                data_path=config['train_data'],
                tokenizer=tokenizer,
                current_stage=scheduled_stage,
                max_num_latent=curriculum_settings['max_num_latent'],
                start_latent_id=curriculum_settings['start_id'],
                latent_id=curriculum_settings['latent_id'],
                end_latent_id=curriculum_settings['end_id'],
                dataset_format=config['dataset_format'],
                max_samples=max_train_samples,
                c_thought=config.get('c_thought', 1),
                force_answer_only=config.get('force_answer_only', False),
            )
            
            eval_dataset = CurriculumDataset(
                data_path=config['valid_data'],
                tokenizer=tokenizer,
                current_stage=scheduled_stage,
                max_num_latent=curriculum_settings['max_num_latent'],
                start_latent_id=curriculum_settings['start_id'],
                latent_id=curriculum_settings['latent_id'],
                end_latent_id=curriculum_settings['end_id'],
                dataset_format=config['dataset_format'],
                max_samples=max_eval_samples,
                c_thought=config.get('c_thought', 1),
                force_answer_only=config.get('force_answer_only', False),
            )
            
            logger.log(f"Train dataset size: {len(train_dataset)}")
            logger.log(f"Eval dataset size: {len(eval_dataset)}")
            
            # Create collator
            collator = CurriculumCollator(
                tokenizer=tokenizer,
                latent_id=curriculum_settings['latent_id'],
                start_latent_id=curriculum_settings['start_id'],
                end_latent_id=curriculum_settings['end_id'],
            )
            
            # Create samplers and dataloaders
            # CRITICAL: drop_last=False to ensure all ranks process same number of batches
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if is_distributed else None
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False) if is_distributed else None
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['per_device_batch_size'],
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                collate_fn=collator,
                num_workers=2,
                pin_memory=True,
                drop_last=False,  # Keep all samples for DDP synchronization
            )
            
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config['per_device_batch_size'],
                sampler=eval_sampler,
                collate_fn=collator,
                num_workers=2,
                pin_memory=True,
                drop_last=False,  # Keep all samples for DDP synchronization
            )
        
        if is_distributed and training_type != 'curriculum':
            train_sampler.set_epoch(epoch)
        elif is_distributed and training_type == 'curriculum':
            # Set epoch for newly created sampler
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, config, logger,
            use_reconstruction=use_reconstruction,
            use_compression=use_compression,
            gradient_tracker=gradient_tracker,
            tb_writer=tb_writer,
        )
        
        # Evaluate
        eval_loss = evaluate(
            model, eval_loader, epoch, logger,
            use_reconstruction=use_reconstruction,
            use_compression=use_compression
        )
        
        if is_main_process():
            logger.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
            
            # Save checkpoint
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_path = os.path.join(config['output_dir'], "best_model.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, 0, eval_loss, save_path)
                logger.log(f"New best model saved (eval_loss={eval_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 1 == 0:
                save_path = os.path.join(config['output_dir'], f"checkpoint_epoch{epoch}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, 0, eval_loss, save_path)
    
    # Save final results and log hyperparameters
    if is_main_process():
        results = {
            'final_train_loss': train_loss,
            'final_eval_loss': eval_loss,
            'best_eval_loss': best_eval_loss,
            'config': config,
        }
        logger.save_results(results)
        
        # Log hyperparameters and final metrics to TensorBoard
        hparams = {
            'learning_rate': float(config['learning_rate']),
            # 'batch_size': int(config['batch_size']),
            'max_epochs': int(config['max_epochs']),
            'model_name': config['model_name'],
            'training_type': training_type,
            'num_parameters': num_params,
        }
        
        if training_type == 'latent':
            hparams.update({
                'num_latent': int(config['num_latent']),
                'use_projection': config['use_projection'],
                'use_layernorm': config['use_layernorm'],
            })
        
        final_metrics = {
            'final_train_loss': train_loss,
            'final_eval_loss': eval_loss,
            'best_eval_loss': best_eval_loss,
        }
        
        logger.log_hyperparameters(hparams, final_metrics)
        logger.log("Training completed!")
        
        # Finalize gradient tracker
        if gradient_tracker is not None:
            gradient_tracker.finalize()
        
        logger.close()  # Close TensorBoard writer
    
    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()

