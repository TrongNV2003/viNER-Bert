import json
from typing import List, Mapping, Tuple

import torch
from transformers import AutoTokenizer


class Dataset:
    def __init__(
        self,
        json_file: str,
        label2id: dict,
        text_col: str,
        label_col: str,
        ) -> None:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        self.data = data
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[str], List[int]]:
        item = self.data[index]
        words = item[self.text_col]
        tags = item[self.label_col]

        labels = [self.label2id[tag] for tag in tags]
        return words, labels

class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, pad_mask_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id

    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        all_words, all_tags = zip(*batch)

        tensor = self.tokenizer(
            list(all_words),
            is_split_into_words=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels_batch = []
        for i in range(len(all_words)):
            word_ids = tensor.word_ids(batch_index=i)
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(self.pad_mask_id)
                else:
                    label_ids.append(all_tags[i][word_id])
            
            labels_batch.append(label_ids)        
        label_tensor = torch.tensor(labels_batch, dtype=torch.long)
        tensor["labels"] = label_tensor
        return tensor
