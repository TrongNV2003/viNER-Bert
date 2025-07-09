import json
import torch
from typing import List
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

from ner.services.preprocessing import TextPreprocess
from ner.phobert_tokenizer_fast.tokenization_phobert_fast import PhobertTokenizerFast


class Inference:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PhobertTokenizerFast,
        id2label: dict,
        device: torch.device,
        
    ) -> None:
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.process = TextPreprocess()

        self.model = model.to(self.device)
        self.model.eval()

    def run(self, texts: List[str], max_length: int = 256, output_file: str = None) -> List[dict]:
        all_predictions = []

        for text in texts:
            processed_text = self.process.process_text(text)

            tensors = self.tokenizer(
                processed_text,
                is_split_into_words=True,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            word_ids = tensors.word_ids(batch_index=0)
            tensors = {key: value.to(self.device) for key, value in tensors.items()}
            with torch.no_grad():
                outputs = self.model(**tensors)
                logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            predicted_label_names = self._map_labels(predictions)
            word_labels = []

            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                label = predicted_label_names[idx]
                word_labels.append({'token': processed_text[word_id], 'label': label})

            all_predictions.append({
                'text': text,
                'tokens': word_labels
            })

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_predictions, f, ensure_ascii=False, indent=4)

        return all_predictions

    def _map_labels(self, label_indices: list) -> list:
        return [self.id2label.get(idx, "O") for idx in label_indices]

def get_tokenizer(checkpoint: str):
    config = AutoConfig.from_pretrained(checkpoint)
    if "RobertaForTokenClassification" in config.architectures:
        tokenizer = PhobertTokenizerFast.from_pretrained(checkpoint, use_fast=True)
        logger.info(f"Using PhobertTokenizerFast for {checkpoint}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True, use_fast=True)
        logger.info(f"Using AutoTokenizer for {checkpoint}")
    return tokenizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "nlp_trainer/ner/models/phobert-base-v2/best_model"
    output_path = "output_infer.json"

    tokenizer = get_tokenizer(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    text = "Xin chào các bạn, tôi là Trọng, tôi năm nay 21 tuổi. Tôi đến từ Hà Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9"

    infer = Inference(model, tokenizer=tokenizer, id2label=model.config.id2label, device=device)
    max_length = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    print(f"Max length: {max_length}")
    
    predictions = infer.run(
        [text],
        max_length=max_length,
        output_file=output_path,
    )

    for prediction in predictions:
        output_text = prediction['text']
        tokens = [token['token'] for token in prediction['tokens']]
        labels = [token['label'] for token in prediction['tokens']]
        print("Text:", output_text)
        print("Tokens:", tokens)
        print("Labels:", labels)