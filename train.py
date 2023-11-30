#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from model import GPT2Model, GPT2Config
from dataset import TextDataset
import requests
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import get_scheduler

logger = get_logger(__name__, log_level="INFO")

def download_checkpoint(model_name, save_path):
    """
    Downloads the GPT-2 checkpoint from Hugging Face's repository.
    """
    base_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
    response = requests.get(base_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Error downloading the model checkpoint: Status code {response.status_code}")


# Parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--model_checkpoint", type=str, default='gpt2', help="GPT-2 model checkpoint")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--use_fsdp", action="store_true", help="Use Fully Sharded Data Parallel")
    parser.add_argument("--wandb_project", type=str, default="gpt2_training", help="W&B project name")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler (linear, cosine, etc.)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of accumulation steps before performing a backward/update pass")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for saving outputs and checkpoints")
    parser.add_argument("--use_pretrained", action="store_true", help="Whether to use a pretrained model")
    parser.add_argument("--save_download", action="store_true", help="Whether to keep the downloaded model checkpoint")

    return parser.parse_args()

def main():
    args = parse_args()

    set_seed(args.seed)

    # Weights & Biases initialization
    if args.use_wandb:
        wandb.init(project=args.wandb_project)

    accelerator_log_kwargs = {}

    # Accelerate setup
    if args.use_wandb:
        accelerator_log_kwargs["log_with"] = "wandb" if args.use_wandb else None
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Load dataset
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    dataset = TextDataset(file_path=args.dataset_path, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model setup
    config = GPT2Config(vocab_size=tokenizer.vocab_size, max_position_embeddings=1024, n_layer=12, n_head=12, n_embd=768)
    model = GPT2Model(config)

    if args.use_pretrained:
        checkpoint_path = "gpt2_checkpoint.bin"
        download_checkpoint(args.model_checkpoint, checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
        if not args.save_download:
            os.remove(checkpoint_path)

    no_decay = ["bias", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=args.num_epochs * len(dataloader),
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    max_train_steps = (args.num_epochs * len(dataloader)/args.gradient_accumulation_steps)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)


    # Train the model
    for epoch in range(args.num_epochs):

        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                inputs, labels = batch['input_ids'], batch['labels']
            logits = model(input_ids=inputs, labels=labels)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss.item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                total_loss += acc_loss
                if args.use_wandb:
                    wandb.log({'step_loss': acc_loss})
                acc_loss = 0

            logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss}')

        if args.use_wandb:
            wandb.log({'epoch': epoch, 'avg_loss': avg_loss})

    # Save the model
    if accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(model)
        torch.save(model_to_save.state_dict(), 'gpt2_finetuned.pth')

if __name__ == "__main__":
    main()
