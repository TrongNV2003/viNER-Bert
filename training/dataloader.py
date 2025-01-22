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

        tokenized_inputs = []
        labels_batch = []

        for words, tags in zip(all_words, all_tags):
            tokens = []
            labels = []
            for word, tag in zip(words, tags):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [self.tokenizer.unk_token]
                tokens.extend(word_tokens)
                # Gán nhãn cho tất cả các sub-tokens của từ đó
                labels.append(tag)  # Gán cho sub-token đầu tiên
                for _ in word_tokens[1:]:
                    labels.append(self.pad_mask_id)
            
            # Special tokens
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            labels = [self.pad_mask_id] + labels + [self.pad_mask_id]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            # Truncating
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]
            
            # Padding
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [self.pad_mask_id] * padding_length

            tokenized_inputs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            labels_batch.append(labels)
        
        batch_input_ids = torch.tensor([item["input_ids"] for item in tokenized_inputs], dtype=torch.long)
        batch_attention_mask = torch.tensor([item["attention_mask"] for item in tokenized_inputs], dtype=torch.long)
        batch_labels = torch.tensor(labels_batch, dtype=torch.long)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }
