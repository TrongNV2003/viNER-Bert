import os
import sys
my_path = os.path.abspath(os.path.dirname(__file__).replace("ner", ""))
sys.path.append(my_path)

import time
import torch
import argparse
from loguru import logger
from dotenv import load_dotenv

from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForTokenClassification,
)

from ner.services.dataloader import Dataset, DataCollator
from ner.services.callbacks.memory_callback import MemoryLoggerCallback
from ner.phobert_tokenizer_fast.tokenization_phobert_fast import PhobertTokenizerFast
from ner.services.metrics import compute_metrics
from ner.utils.get_labels import get_unique_labels
from ner.utils.model_utils import set_seed, get_vram_usage, count_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True, help="Model checkpoint or name")
parser.add_argument("--train_file", type=str, default="dataset/train_word.json", required=True, help="Path to training data")
parser.add_argument("--valid_file", type=str, default="dataset/dev_word.json", required=True, help="Path to validation data")
parser.add_argument("--test_file", type=str, default="dataset/test_word.json", required=True, help="Path to test data")
parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
parser.add_argument("--pad_mask_id", type=int, default=-100, help="Padding mask ID for loss calculation")
parser.add_argument("--text_col", type=str, default="tokens", help="Column name for text data")
parser.add_argument("--label_col", type=str, default="ner_tags", help="Column name for label data")
parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Optimizer to use for training")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model and logs")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Number of warmup steps")
parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation")
parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy")
parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to save")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
parser.add_argument("--logging_dir", type=str, default=None, help="Directory for logging")
parser.add_argument("--fp16", action="store_true", default=False, help="Enable mixed precision training (FP16)")
parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 training")
parser.add_argument("--metric_for_best_model", type=str, default="epoch", help="Metric to select best model")
parser.add_argument("--greater_is_better", action="store_true", default=True, help="Whether higher metric is better")
parser.add_argument("--load_best_model_at_end", action="store_true", default=True, help="Load best model at the end")
parser.add_argument("--use_word_splitter", action="store_true", default=False, help="Split words from ice_tea to ice tea")
parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing")
parser.add_argument("--record_output_file", type=str, default="output.json", help="Output file for evaluation results")
parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of dataloader workers")
parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for dataloader")
parser.add_argument("--report_to", type=str, help="Reporting tool for training metrics")
args = parser.parse_args()

def get_tokenizer(checkpoint: str):
    config = AutoConfig.from_pretrained(checkpoint)
    if "RobertaForMaskedLM" in config.architectures:
        tokenizer = PhobertTokenizerFast.from_pretrained(checkpoint, use_fast=True)
        logger.info(f"Using PhobertTokenizerFast for {checkpoint}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True, use_fast=True)
        logger.info(f"Using AutoTokenizer for {checkpoint}")
    return tokenizer

def get_model(
    checkpoint: str,
    device: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
) -> AutoModelForTokenClassification:
    config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # config.classifier_activation = "gelu"
    # config.global_rope_theta = 16000.0
    # config.local_rope_theta = 10000.0
    
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        config=config,
    )
    model = model.to(device)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)

    unique_labels = get_unique_labels(args.train_file, label_col=args.label_col)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.model)
    model = get_model(args.model, device, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)
    max_length = getattr(model.config, 'max_position_embeddings', args.max_length)
    
    train_set = Dataset(json_file=args.train_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col, use_word_splitter=args.use_word_splitter)
    valid_set = Dataset(json_file=args.valid_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col, use_word_splitter=args.use_word_splitter)
    test_set = Dataset(json_file=args.test_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col, use_word_splitter=args.use_word_splitter)

    collator = DataCollator(tokenizer=tokenizer, max_length=max_length, pad_mask_id=args.pad_mask_id)

    count_parameters(model)

    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}"
    logging_dir = args.logging_dir if args.logging_dir else f"{save_dir}/logs"

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.pin_memory,
    )
    start_time = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label=id2label),
        callbacks=[MemoryLoggerCallback],
    )
    trainer.train()
    end_time = time.time()
    
    test_metrics = trainer.evaluate(eval_dataset=test_set)
    print(f"Test metrics: {test_metrics}")

    if torch.cuda.is_available():
        max_vram = get_vram_usage(device)
        print(f"VRAM usage: {max_vram:.2f} GB")
    print(f"Training time: {(end_time - start_time) / 60:.2f} mins")
    print(f"\nmodel: {args.model}")
    print(f"params: lr {args.learning_rate}, epochs {args.num_train_epochs}")
