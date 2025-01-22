import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from training.dataloader import Dataset, LlmDataCollator
from training.evaluate import Tester
from training.trainer import LlmTrainer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2")
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--save_dir", type=str, default="./bert-classification")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--valid_batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=8)
parser.add_argument("--train_file", type=str, default="dataset/train_word.json")
parser.add_argument("--valid_file", type=str, default="dataset/dev_word.json")
parser.add_argument("--test_file", type=str, default="dataset/test_word.json")
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True, use_fast = True)
    return tokenizer


def get_model(
    checkpoint: str, device: str, num_labels: str, id2label: list, label2id: list
    ) -> AutoModelForTokenClassification:
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    print(f"Number of labels in label_mapping: {len(label2id)}")
    print(f"Model num_labels: {model.config.num_labels}")
    
    return model

def count_parameters(model: torch.nn.Module) -> None:
    """
    Prints the total number of parameters and trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        None
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    set_seed(args.seed)

    unique_labels = ['O', 'B-AGE', 'B-DATE', 'B-GENDER', 'B-JOB', 'B-LOCATION', 'B-NAME', 'B-ORGANIZATION', 'B-PATIENT_ID', 'B-SYMPTOM_AND_DISEASE', 'B-TRANSPORTATION', 'I-AGE', 'I-DATE', 'I-JOB', 'I-LOCATION', 'I-NAME', 'I-ORGANIZATION', 'I-PATIENT_ID', 'I-SYMPTOM_AND_DISEASE', 'I-TRANSPORTATION']
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.model)
    
    train_set = Dataset(json_file=args.train_file, label_mapping=label2id)
    valid_set = Dataset(json_file=args.valid_file, label_mapping=label2id)
    test_set = Dataset(json_file=args.test_file, label_mapping=label2id)

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length, pad_mask_id=args.pad_mask_id)

    model = get_model(args.model, args.device, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)
    count_parameters(model)

    trainer = LlmTrainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator,
    )
    trainer.train()

    # Evaluate model on test set
    MODEL = "bert-classification"
    tuned_model = AutoModelForTokenClassification.from_pretrained(MODEL)

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator)
    tester = Tester(model=tuned_model, test_loader=test_loader, output_file=args.record_output_file, labels_mapping=id2label)

    tester.evaluate()
