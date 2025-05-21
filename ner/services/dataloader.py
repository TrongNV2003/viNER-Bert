import json
from typing import List, Mapping, Tuple

import torch
from transformers import AutoTokenizer


class Dataset:
    def __init__(self, json_file: str, label2id: dict) -> None:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        self.data = data
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[str], List[int]]:
        item = self.data[index]
        words = item["words"]
        tags = item["tags"]

        labels = [self.label2id[tag] for tag in tags]
        return words, labels

class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, pad_mask_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id

    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        """
        Fast tokenize and align labels with Phobert tokenizer model
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

# manually tokenize and align labels with slow tokenizer model
    # def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:

        # all_words, all_tags = zip(*batch)
        # tokenized_inputs = []
        # labels_batch = []

        # for words, tags in zip(all_words, all_tags):
        #     tokens = []
        #     labels = []
        #     for word, tag in zip(words, tags):
        #         word_tokens = self.tokenizer.tokenize(word)
        #         if not word_tokens:
        #             word_tokens = [self.tokenizer.unk_token]
        #         tokens.extend(word_tokens)
        #         # Gán nhãn cho tất cả các sub-tokens của từ đó
        #         labels.append(tag)  # Gán cho sub-token đầu tiên
        #         for _ in word_tokens[1:]:
        #             labels.append(self.pad_mask_id)
            
        #     # Special tokens
        #     tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        #     labels = [self.pad_mask_id] + labels + [self.pad_mask_id]

        #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #     attention_mask = [1] * len(input_ids)

        #     # Truncating
        #     if len(input_ids) > self.max_length:
        #         input_ids = input_ids[:self.max_length]
        #         attention_mask = attention_mask[:self.max_length]
        #         labels = labels[:self.max_length]
            
        #     # Padding
        #     padding_length = self.max_length - len(input_ids)
        #     if padding_length > 0:
        #         input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        #         attention_mask = attention_mask + [0] * padding_length
        #         labels = labels + [self.pad_mask_id] * padding_length

        #     tokenized_inputs.append({
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #     })
        #     labels_batch.append(labels)
        
        # batch_input_ids = torch.tensor([item["input_ids"] for item in tokenized_inputs], dtype=torch.long)
        # batch_attention_mask = torch.tensor([item["attention_mask"] for item in tokenized_inputs], dtype=torch.long)
        # batch_labels = torch.tensor(labels_batch, dtype=torch.long)

        # return {
        #     "input_ids": batch_input_ids,
        #     "attention_mask": batch_attention_mask,
        #     "labels": batch_labels
        # }
