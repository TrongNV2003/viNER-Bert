import torch
from torch.utils.data import DataLoader, Dataset

import time
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


class TestingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        model: torch.nn.Module,
        pin_memory: bool,
        test_set: Dataset,
        test_batch_size: int,
        id2label: dict,
        collate_fn: Optional[Callable] = None,
        output_file: Optional[str] = None,
    ) -> None:
        self.id2label = id2label
        self.output_file = output_file
        self.device = device
        self.model = model.to(self.device)
        self.test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self):
        self.model.eval()
        results = []
        latencies = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader), unit="batches"):
                text_input_ids = batch["input_ids"].to(self.device)
                text_attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_start_time = time.time()
                outputs = self.model(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                    labels=labels,
                )
                logits = outputs.logits
                batch_end_time = time.time()
                latency = batch_end_time - batch_start_time
                latencies.append(latency)

                preds = torch.argmax(logits, dim=-1)

                # Chỉ lấy các token mà labels không phải là -100
                active_indices = labels.view(-1) != -100
                filtered_labels = labels.view(-1)[active_indices].cpu().numpy().tolist()
                filtered_preds = preds.view(-1)[active_indices].cpu().numpy().tolist()

                all_preds.extend(filtered_preds)
                all_labels.extend(filtered_labels)

                for i in range(len(text_input_ids)):
                    # Chỉ xử lý các token có nhãn không phải là -100
                    active = labels[i].cpu().numpy() != -100
                    true_label_indices = labels[i].cpu().numpy()[active]
                    predicted_label_indices = preds[i].cpu().numpy()[active]

                    true_label_names = self._map_labels(true_label_indices)
                    predicted_label_names = self._map_labels(predicted_label_indices)
                    results.append({
                        "true_labels": true_label_names,
                        "predicted_labels": predicted_label_names,
                        "latency": float(latency),
                    })

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {self.output_file}")

        self.score(all_labels, all_preds)
        self.calculate_latency(latencies)

        num_samples = len(results)
        print(f"num samples: {num_samples}")

    def _map_labels(self, label_indices: list) -> list:
        return [self.id2label.get(idx, "O") for idx in label_indices]

    def score(self, all_labels: list, all_preds: list) -> None:
        true_labels = [self.id2label.get(idx, "O") for idx in all_labels]
        preds_labels = [self.id2label.get(idx, "O") for idx in all_preds]


        # Tạo danh sách các sequence, giả sử tất cả thuộc về một sequence
        true_labels = [true_labels]
        preds_labels = [preds_labels]

        precision = precision_score(true_labels, preds_labels, average="weighted")
        recall = recall_score(true_labels, preds_labels, average="weighted")
        f1 = f1_score(true_labels, preds_labels, average="weighted")
        report = classification_report(true_labels, preds_labels)
        print(report)
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def calculate_latency(self, latencies: list) -> None:
        stats = {
            "p95_ms": float(np.percentile(latencies, 95) * 1000),
            "p99_ms": float(np.percentile(latencies, 99) * 1000),
            "mean_ms": float(np.mean(latencies) * 1000),
        }
        print("\nLatency Statistics:")
        print(f"P95 Latency: {stats['p95_ms']:.2f} ms")
        print(f"P99 Latency: {stats['p99_ms']:.2f} ms")
        print(f"Mean Latency: {stats['mean_ms']:.2f} ms")
        return stats
        