import torch
from transformers import AutoModelForTokenClassification
from training.preprocessing import TextPreprocess
import json
from typing import List, Union
from tokenizer_fast.tokenization_phobert_fast import PhobertTokenizerFast

preprocess = TextPreprocess()

class Tester:
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

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate(self, text: str, max_length: int = 128, output_file: str = None) -> List[dict]:
        """
        Đánh giá mô hình trên một danh sách các văn bản.

        Args:
            texts (str): Danh sách các văn bản cần kiểm thử.
            max_length (int, optional): Độ dài tối đa sau khi tokenize. Mặc định là 128.
            output_file (str, optional): Đường dẫn file để lưu kết quả. Nếu không cung cấp, không lưu.

        Returns:
            list: Danh sách các dự đoán cho mỗi văn bản.
        """

        all_predictions = []

        processed_text = preprocess.process_text(text)

        tokenized_inputs = self.tokenizer(
            processed_text,
            is_split_into_words=True,  # Nếu bạn tokenize từng từ
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )

        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)
        word_ids = tokenized_inputs.word_ids(batch_index=0)  # Lấy word_ids cho batch đầu tiên
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Lấy chỉ số nhãn dự đoán cho mỗi token
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # Vì batch_size=1

        # Chuyển các chỉ số nhãn thành tên nhãn
        predicted_label_names = self._map_labels(predictions)

        # Lấy các từ gốc và loại bỏ các token padding
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # Nếu đã sử dụng `is_split_into_words=True`, có thể cần ánh xạ lại với từ gốc

        # Tạo danh sách các từ và nhãn tương ứng, loại bỏ padding và các token đặc biệt
        word_labels = []
        current_word = None
        current_label = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Bỏ qua các token đặc biệt
            if word_id != current_word:
                # Bắt đầu từ mới
                current_word = word_id
                current_label = predicted_label_names[idx]
                word_labels.append({'token': processed_text[word_id], 'label': current_label})
            else:
                # Token thuộc cùng một từ, thường là các token con, bỏ qua hoặc gán nhãn đặc biệt
                # Ở đây, chúng ta sẽ bỏ qua việc gán nhãn cho các token con
                pass

        all_predictions.append({
            'text': text,
            'tokens': word_labels
        })

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_predictions, f, ensure_ascii=False, indent=4)

        return all_predictions

    def _map_labels(self, label_indices: list) -> list:
        """
        Ánh xạ các chỉ số nhãn sang tên nhãn.

        Args:
            label_indices (list): Danh sách chỉ số nhãn của một sequence.

        Returns:
            list: Danh sách tên nhãn tương ứng.
        """
        return [self.labels_mapping.get(idx, "O") for idx in label_indices]

if __name__ == "__main__":
    MODEL_NAME = "bert-classification"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PhobertTokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    unique_labels = [
        'O', 'B-AGE', 'B-DATE', 'B-GENDER', 'B-JOB', 'B-LOCATION', 'B-NAME',
        'B-ORGANIZATION', 'B-PATIENT_ID', 'B-SYMPTOM_AND_DISEASE',
        'B-TRANSPORTATION', 'I-AGE', 'I-DATE', 'I-JOB', 'I-LOCATION',
        'I-NAME', 'I-ORGANIZATION', 'I-PATIENT_ID', 'I-SYMPTOM_AND_DISEASE',
        'I-TRANSPORTATION'
    ]
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    text = "Xin chào các bạn, tôi là Trọng, tôi năm nay 21 tuổi. Đồng thời tôi đến từ Hà Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9"

    tester = Tester(
        model=model,
        tokenizer=tokenizer,
        labels_mapping=id2label,
        device=device
    )

    predictions = tester.evaluate(
        text,
        max_length=128,
        output_file=None
    )
    for prediction in predictions:
        text = prediction['text']
        tokens = [token['token'] for token in prediction['tokens']]
        labels = [token['label'] for token in prediction['tokens']]

        print("Text:", text)
        print("Tokens:", tokens)
        print("Labels:", labels)

    """OUTPUT:
    Text: Xin chào các bạn, tôi là Trọng, tôi năm nay 18 tuổi. Đồng thời tôi đến từ Hà Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9
    Tokens: ['Xin', 'chào', 'các', 'bạn', ',', 'tôi', 'là', 'Trọng', ',', 'tôi', 'năm', 'nay', '18', 'tuổi', '.', 'Đồng_thời', 'tôi', 'đến', 'từ', 'Hà_Nội', ',', 'mảnh', 'đất', 'của', 'Việt_Nam', '.', 'Hôm_nay', 'tôi', 'đi', 'khám', 'ở', 'bệnh_viện', 'Bạch_Mai', '.', 'Tôi', 'vào', 'thăm', 'bệnh_nhân', 'bị', 'COVID-19', 'chiều', 'ngày', '15', 'tháng', '9']
    Labels: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NAME', 'O', 'O', 'O', 'O', 'B-AGE', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOCATION', 'O', 'O', 'O', 'O', 'B-LOCATION', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOCATION', 'I-LOCATION', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']
    """