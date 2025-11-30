"""
Training script for Answer-Only model (direct question → answer).

This script trains a standard GPT-2 model on GSM8k with:
- Input: Question + ### answer
- No steps in input or output
- Only answer is supervised

Usage:
    python scripts/train_answer_only.py configs/gpt2_answer_only.yaml
    
Or with torchrun for multi-GPU:
    torchrun --nproc_per_node=2 scripts/train_answer_only.py configs/gpt2_answer_only.yaml
"""

import os
import sys
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import AnswerOnlyDataset, AnswerOnlyCollator
from src.utils import setup_distributed, cleanup_distributed, save_checkpoint, AverageMeter, Logger, is_main_process


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    config,
    logger,
):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    if is_main_process():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for step, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(model.device if not isinstance(model, DDP) else model.module.device) 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        
        # Gradient accumulation
        loss = loss / int(config['gradient_accumulation_steps'])
        loss.backward()
        
        # Update weights
        if (step + 1) % int(config['gradient_accumulation_steps']) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['max_grad_norm']))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        loss_meter.update(loss.item() * int(config['gradient_accumulation_steps']))
        
        # Logging
        if is_main_process():
            if step % int(config['logging_steps']) == 0:
                metrics = {
                    'loss': loss_meter.avg,
                    'lr': scheduler.get_last_lr()[0],
                }
                logger.log_metrics(metrics, epoch * len(train_loader) + step, prefix="train")
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg


def evaluate(
    model,
    eval_loader,
    epoch,
    logger,
):
    """Evaluate the model."""
    model.eval()
    loss_meter = AverageMeter()
    
    if is_main_process():
        pbar = tqdm(eval_loader, desc=f"Eval Epoch {epoch}")
    else:
        pbar = eval_loader
    
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            batch = {k: v.to(model.device if not isinstance(model, DDP) else model.module.device) 
                    for k, v in batch.items()}
            
            # Forward
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            loss = outputs.loss
            loss_meter.update(loss.item())
            
            if is_main_process():
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    if is_main_process():
        metrics = {'loss': loss_meter.avg}
        logger.log_metrics(metrics, epoch, prefix="eval")
    
    return loss_meter.avg


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python train_answer_only.py <config_file>")
        print("Example: torchrun --nproc_per_node=2 scripts/train_answer_only.py configs/gpt2_answer_only.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    is_distributed = setup_distributed()
    
    # Get device
    if torch.cuda.is_available():
        if is_distributed:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        else:
            device = torch.device('cuda:0')
            rank = 0
            world_size = 1
    else:
        device = torch.device('cpu')
        rank = 0
        world_size = 1
    
    # Create output directory
    output_dir = config['output_dir']
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = Logger(output_dir, use_tensorboard=config.get('use_tensorboard', True))
    
    if is_main_process():
        logger.log("=" * 80)
        logger.log("Answer-Only Training (Direct Question → Answer)")
        logger.log("=" * 80)
        logger.log(f"Config: {config_path}")
        logger.log(f"Output directory: {output_dir}")
        logger.log(f"World size: {world_size}")
        logger.log(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(int(config['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config['seed']))
    
    # Load tokenizer
    if is_main_process():
        logger.log("\nLoading tokenizer...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    if is_main_process():
        logger.log("\nLoading datasets...")
    
    train_dataset = AnswerOnlyDataset(
        data_path=config['train_data'],
        tokenizer=tokenizer,
        dataset_format=config['dataset_format'],
        max_samples=config.get('max_train_samples', None),
    )
    
    eval_dataset = AnswerOnlyDataset(
        data_path=config['valid_data'],
        tokenizer=tokenizer,
        dataset_format=config['dataset_format'],
        max_samples=config.get('max_eval_samples', None),
    )
    
    if is_main_process():
        logger.log(f"Train dataset size: {len(train_dataset)}")
        logger.log(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create data loaders
    collator = AnswerOnlyCollator(tokenizer=tokenizer)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['per_device_batch_size']),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(config['per_device_batch_size']),
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Load model
    if is_main_process():
        logger.log("\nLoading model...")
    
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    model.to(device)
    
    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = DDP(model, device_ids=[local_rank])
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(f"Total parameters: {total_params:,}")
        logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
    )
    
    total_steps = len(train_loader) * int(config['max_epochs']) // int(config['gradient_accumulation_steps'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_steps']),
        num_training_steps=total_steps,
    )
    
    # Training loop
    if is_main_process():
        logger.log("\nStarting training...")
    
    best_eval_loss = float('inf')
    
    for epoch in range(int(config['max_epochs'])):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            logger=logger,
        )
        
        # Evaluate
        eval_loss = evaluate(
            model=model,
            eval_loader=eval_loader,
            epoch=epoch,
            logger=logger,
        )
        
        if is_main_process():
            logger.log(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
        
        # Save checkpoint
        if is_main_process():
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt")
            save_checkpoint(
                model=model.module if isinstance(model, DDP) else model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=0,  # Not tracking steps in this training loop
                loss=eval_loss,
                save_path=checkpoint_path,
            )
            logger.log(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_path = os.path.join(output_dir, "best_model.pt")
                save_checkpoint(
                    model=model.module if isinstance(model, DDP) else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=0,
                    loss=eval_loss,
                    save_path=best_model_path,
                )
                logger.log(f"New best model! Saved to: {best_model_path}")
    
    if is_main_process():
        logger.log("\n" + "=" * 80)
        logger.log("Training completed!")
        logger.log("=" * 80)
    
    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()

