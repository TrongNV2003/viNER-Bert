import json
import torch
import argparse
from typing import List
from transformers import AutoModelForTokenClassification

from ner.services.preprocessing import TextPreprocess
from ner.phobert_tokenizer_fast.tokenization_phobert_fast import PhobertTokenizerFast


class Inference:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PhobertTokenizerFast,
        labels_mapping: dict,
        device: torch.device = None,
        
    ) -> None:
        """
        Args:
            model (torch.nn.Module): Mô hình đã huấn luyện.
            tokenizer (PhobertTokenizerFast): Tokenizer tương ứng với mô hình.
            labels_mapping (dict): Từ điển ánh xạ từ chỉ số nhãn sang tên nhãn, ví dụ {0: "O", 1: "B-PER", ...}.
            device (torch.device, optional): Thiết bị để chạy mô hình (CPU hoặc GPU). Nếu không cung cấp, tự động chọn.
        """

        self.tokenizer = tokenizer
        self.labels_mapping = labels_mapping
        self.process = TextPreprocess()

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def run(self, texts: List[str], max_length: int = 256, output_file: str = None) -> List[dict]:
        """
        Infer trên một hoặc nhiều context được cung cấp

        Args:
            texts (str): Danh sách các văn bản cần kiểm thử.
            max_length (int, optional): Độ dài tối đa sau khi tokenize. Mặc định là 128.
            output_file (str, optional): Đường dẫn file để lưu kết quả. Nếu không cung cấp, không lưu.

        Returns:
            list: Danh sách các dự đoán cho mỗi văn bản.
        """

        all_predictions = []

        for text in texts:
            processed_text = self.process.process_text(text)

            tokenized_inputs = self.tokenizer(
                processed_text,
                is_split_into_words=True,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized_inputs["input_ids"].to(self.device)
            attention_mask = tokenized_inputs["attention_mask"].to(self.device)
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            predicted_label_names = self._map_labels(predictions)
            # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
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
        return [self.labels_mapping.get(idx, "O") for idx in label_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
    parser.add_argument("--output_dir", type=str, default="./models", required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--record_output_file", type=str, default="output_infer.json")
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}"

    tokenizer = PhobertTokenizerFast.from_pretrained(save_dir)
    model = AutoModelForTokenClassification.from_pretrained(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "Xin chào các bạn, tôi là Trọng, tôi năm nay 21 tuổi. Tôi đến từ Hà Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9"

    infer = Inference(model, tokenizer=tokenizer, id2label=model.config.id2label, device=device)

    predictions = infer.run(
        [args.input],
        max_length=256,
        output_file=args.record_output_file
    )

    for prediction in predictions:
        output_text = prediction['text']
        tokens = [token['token'] for token in prediction['tokens']]
        labels = [token['label'] for token in prediction['tokens']]

        print("Text:", output_text)
        print("Tokens:", tokens)
        print("Labels:", labels)


"""OUTPUT:
{'text': 'Xin chào các bạn, tôi là Trọng, tôi năm nay 21 tuổi. Tôi đến từ Hà '
        'Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch '
        'Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9',
'tokens': [{'label': 'O', 'token': 'Xin'},
        {'label': 'O', 'token': 'chào'},
        {'label': 'O', 'token': 'các'},
        {'label': 'O', 'token': 'bạn'},
        {'label': 'O', 'token': ','},
        {'label': 'O', 'token': 'tôi'},
        {'label': 'O', 'token': 'là'},
        {'label': 'B-NAME', 'token': 'Trọng'},
        {'label': 'O', 'token': ','},
        {'label': 'O', 'token': 'tôi'},
        {'label': 'O', 'token': 'năm'},
        {'label': 'O', 'token': 'nay'},
        {'label': 'B-AGE', 'token': '21'},
        {'label': 'O', 'token': 'tuổi'},
        {'label': 'O', 'token': '.'},
        {'label': 'O', 'token': 'Tôi'},
        {'label': 'O', 'token': 'đến'},
        {'label': 'O', 'token': 'từ'},
        {'label': 'B-LOCATION', 'token': 'Hà_Nội'},
        {'label': 'O', 'token': ','},
        {'label': 'O', 'token': 'mảnh'},
        {'label': 'O', 'token': 'đất'},
        {'label': 'O', 'token': 'của'},
        {'label': 'B-LOCATION', 'token': 'Việt_Nam'},
        {'label': 'O', 'token': '.'},
        {'label': 'O', 'token': 'Hôm_nay'},
        {'label': 'O', 'token': 'tôi'},
        {'label': 'O', 'token': 'đi'},
        {'label': 'O', 'token': 'khám'},
        {'label': 'O', 'token': 'ở'},
        {'label': 'B-LOCATION', 'token': 'bệnh_viện'},
        {'label': 'I-LOCATION', 'token': 'Bạch_Mai'},
        {'label': 'O', 'token': '.'},
        {'label': 'O', 'token': 'Tôi'},
        {'label': 'O', 'token': 'vào'},
        {'label': 'O', 'token': 'thăm'},
        {'label': 'O', 'token': 'bệnh_nhân'},
        {'label': 'O', 'token': 'bị'},
        {'label': 'O', 'token': 'COVID-19'},
        {'label': 'O', 'token': 'COVID-19'},
        {'label': 'O', 'token': 'COVID-19'},
        {'label': 'O', 'token': 'COVID-19'},
        {'label': 'O', 'token': 'chiều'},
        {'label': 'O', 'token': 'ngày'},
        {'label': 'B-DATE', 'token': '15'},
        {'label': 'I-DATE', 'token': 'tháng'},
        {'label': 'I-DATE', 'token': '9'}]}
"""