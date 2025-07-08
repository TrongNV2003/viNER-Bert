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
        use_word_splitter: bool = False,
        ) -> None:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if use_word_splitter:
                    words, tags = [], []
                    for word, tag in zip(item[text_col], item[label_col]):
                        split_words, split_tags = self._split_word(word, tag)
                        words.extend(split_words)
                        tags.extend(split_tags)
                    item[text_col] = words
                    item[label_col] = tags
                data.append(item)

        self.data = data
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def _split_word(self, word, tag):
        if "_" not in word:
            return [word], [tag]
        parts = word.split("_")
        if tag == "O":
            return parts, ["O"] * len(parts)
        elif tag.startswith("B-"):
            label_type = tag[2:]
            new_tags = ["B-" + label_type] + ["I-" + label_type] * (len(parts)-1)
            return parts, new_tags
        elif tag.startswith("I-"):
            label_type = tag[2:]
            new_tags = ["I-" + label_type] * len(parts)
            return parts, new_tags
        else:
            return parts, [tag] * len(parts)

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
