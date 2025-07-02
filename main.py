import os

import time
import argparse

import torch
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer

from ner.services.evaluate import TestingArguments
from ner.services.trainer import TrainingArguments
from ner.services.dataloader import Dataset, DataCollator
from ner.phobert_tokenizer_fast.tokenization_phobert_fast import PhobertTokenizerFast   # only for phobert model
from ner.utils.get_labels import get_unique_labels
from ner.utils.model_utils import set_seed, get_vram_usage, count_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42, required=True)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--use_warmup_steps", action="store_true", default=False)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--valid_batch_size", type=int, default=16, required=True)
parser.add_argument("--test_batch_size", type=int, default=16, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train_word.jsonl", required=True)
parser.add_argument("--valid_file", type=str, default="dataset/dev_word.jsonl", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test_word.jsonl", required=True)
parser.add_argument("--text_col", type=str, default="tokens", help="Column name for text data")
parser.add_argument("--label_col", type=str, default="ner_tags", help="Column name for label data")
parser.add_argument("--output_dir", type=str, default="./models", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json", required=True)
parser.add_argument("--early_stopping_patience", type=int, default=3, required=True)
parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
parser.add_argument("--evaluate_on_accuracy", action="store_true", default=False)
args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> PhobertTokenizerFast:
    tokenizer = PhobertTokenizerFast.from_pretrained(checkpoint, add_prefix_space=True, use_fast=True)
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
    
    train_set = Dataset(json_file=args.train_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col)
    valid_set = Dataset(json_file=args.valid_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col)
    test_set = Dataset(json_file=args.test_file, label2id=label2id, text_col=args.text_col, label_col=args.label_col)

    collator = DataCollator(tokenizer=tokenizer, max_length=args.max_length, pad_mask_id=args.pad_mask_id)

    model = get_model(
        args.model, device, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    print(f"\nLabel: {model.config.id2label}")

    count_parameters(model)

    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}"
    
    start_time = time.time()
    trainer = TrainingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_warmup_steps=args.use_warmup_steps,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        evaluate_on_accuracy=args.evaluate_on_accuracy,
    )
    trainer.train()
    end_time = time.time()


    # Evaluate model
    tuned_model = AutoModelForTokenClassification.from_pretrained(save_dir)
    tester = TestingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        model=tuned_model,
        pin_memory=args.pin_memory,
        test_set=test_set,
        test_batch_size=args.test_batch_size,
        collate_fn=collator,
        output_file=args.record_output_file,
    )
    tester.evaluate()

    print(f"\nmodel: {args.model}")
    if torch.cuda.is_available():
        max_vram = get_vram_usage(device)
        print(f"VRAM tối đa tiêu tốn khi huấn luyện: {max_vram:.2f} GB")
    print(f"Training time: {(end_time - start_time) / 60} mins")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")