import json
from typing import List, Mapping, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(self, json_file: str, label_mapping: dict) -> None:
        """
        Args:
            json_file (str): Path to the JSON file.
            label_mapping (dict): Từ điển { "O": 0, "B-LOC": 1, "I-LOC": 2, ... }
        """

        data = []
        # Nếu file là danh sách JSON duy nhất => dùng json.load
        # Nếu mỗi dòng một JSON => duyệt từng line
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        self.data = data
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[str], List[int]]:
        """
        Get the item at the given index

        Returns:
            text: the text of the item
            labels: Multi-label vector
        """

        item = self.data[index]
        words = item["words"]
        tags = item["tags"]

        labels = [self.label_mapping[tag] for tag in tags]
        return words, labels


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, pad_mask_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id

    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        """
        Tokenize the batch of data and align labels with tokenized input.
        
        Args:
            batch (list): List of tuples [(words, labels), ...]

        Returns:
            dict: Tokenized data with aligned labels.
        """

        all_words, all_tags = zip(*batch)

        tensor = self.tokenizer(
            list(all_words),
            is_split_into_words=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # tạo label_ids align với sub-words
        labels_batch = []
        for i in range(len(all_words)):
            word_ids = tensor.word_ids(batch_index=i)
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(self.pad_mask_id)  # gán pad_mask_id cho các token CLS, SEP, PAD để model bỏ qua
                else:
                    label_ids.append(all_tags[i][word_id])
            
            labels_batch.append(label_ids)        

        label_tensor = torch.tensor(labels_batch, dtype=torch.long)
        tensor["labels"] = label_tensor

        return tensor