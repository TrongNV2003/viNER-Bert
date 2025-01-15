import time
import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import json

class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        output_file: str,
        labels_mapping: dict,
    ) -> None:
        """
        Args:
            model (torch.nn.Module): Mô hình đã huấn luyện.
            test_loader (DataLoader): DataLoader cho tập test.
            output_file (str): Đường dẫn file để lưu kết quả.
            labels_mapping (dict): Từ điển ánh xạ từ chỉ số nhãn sang tên nhãn, ví dụ {0: "O", 1: "B-PER", ...}.
        """
        self.test_loader = test_loader
        self.output_file = output_file
        self.labels_mapping = labels_mapping

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def evaluate(self):
        """
        This function will eval the model on test set and return the accuracy, F1-score and latency

        Parameters:
            None

        Returns:
            None
        """

        self.model.eval()
        latencies = []
        all_labels = []
        all_preds = []
        total_loss = 0
        results = []
        
        start_time = time.time()    #throughput
        with torch.no_grad():
            for batch in self.test_loader:
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

                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                for i in range(len(text_input_ids)):
                    true_label_indices = labels.cpu().numpy()[i]
                    predicted_label_indices = preds.cpu().numpy()[i]

                    true_label_names = self._map_labels(true_label_indices)
                    predicted_label_names = self._map_labels(predicted_label_indices)
                    results.append({
                        "true_labels": true_label_names,
                        "predicted_labels": predicted_label_names,
                        "latency": float(latency),
                    })
        total_time = time.time() - start_time
        num_samples = len(results)

        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {self.output_file}")

        self.score(all_labels, all_preds)
        self.calculate_latency(latencies)

        throughput = num_samples / total_time
        print(f"num samples: {num_samples}")
        print(f"Throughput: {throughput:.2f} samples/s")

    def _map_labels(self, label_indices: list) -> list:
        """
        Ánh xạ các chỉ số nhãn sang tên nhãn.

        Args:
            label_indices (list): Danh sách chỉ số nhãn của một sequence.

        Returns:
            list: Danh sách tên nhãn tương ứng.
        """
        return [self.labels_mapping.get(idx, "O") if idx != -100 else "O" for idx in label_indices]


    def score(self, true_labels: list, preds: list) -> None:
        """
        Tính toán và in ra các chỉ số Precision, Recall, F1-score.

        Args:
            true_labels (list): Danh sách các sequence nhãn thực tế.
            preds (list): Danh sách các sequence nhãn dự đoán.
        """

        true_labels_names = [self._map_labels(seq) for seq in true_labels]
        preds_labels_names = [self._map_labels(seq) for seq in preds]

        precision = precision_score(true_labels_names, preds_labels_names)
        recall = recall_score(true_labels_names, preds_labels_names)
        f1 = f1_score(true_labels_names, preds_labels_names)
        report = classification_report(true_labels_names, preds_labels_names)

        print(report)
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def calculate_latency(self, latencies: list) -> None:
        """
        Tính toán và in ra latency P99.

        Args:
            latencies (list): Danh sách các giá trị latency.

        Returns:
            None
        """
        p99_latency = np.percentile(latencies, 99)
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")