import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from tqdm import tqdm
from loguru import logger

from services.utils import AverageMeter

class LlmTrainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        collator_fn=None,
        evaluate_on_accuracy: bool = False,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.evaluate_on_accuracy = evaluate_on_accuracy
        self.best_valid_score = 0 if evaluate_on_accuracy else float("inf")


    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    logits = outputs.logits
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, num_labels)
                        labels.view(-1)                    # (batch_size * seq_len)
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

            valid_score = self.evaluate(self.valid_loader)

            if self.evaluate_on_accuracy:
                if valid_score > self.best_valid_score:
                    print(f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self._save()
                    print(f"Saved best model.")

            else:
                if valid_score < self.best_valid_score:
                    print(f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self._save()
                    print(f"Saved best model.")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        all_preds = []
        all_labels = []

        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, num_labels)
                    labels.view(-1)                    # (batch_size * seq_len)
                )
            
                eval_loss.update(loss.item(), input_ids.size(0))

                preds = torch.argmax(logits, dim=-1).cpu().numpy()  # (batch_size, seq_len)
                labels_np = labels.cpu().numpy()
                
                for i in range(len(preds)):
                    mask = labels_np[i] != -100  # Chỉ lấy các token không phải padding
                    all_preds.extend(preds[i][mask].tolist())
                    all_labels.extend(labels_np[i][mask].tolist())

                if self.evaluate_on_accuracy and all_preds:
                    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
                    tepoch.set_postfix({"valid_loss": eval_loss.avg, "valid_acc": accuracy})
                else:
                    tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_preds == all_labels)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # In các metric
        logger.info(f"\n=== Validation Metrics ===")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")
        
        return accuracy if self.evaluate_on_accuracy else eval_loss.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
