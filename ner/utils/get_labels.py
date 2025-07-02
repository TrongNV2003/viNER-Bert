import json

def get_unique_labels(jsonl_file: str, label_col: str) -> list:
    unique_labels = set()
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                unique_labels.update(obj[label_col])
    print(f"\nDanh sách nhãn: {sorted(unique_labels)}")
    print(f"Tổng số lượng nhãn: {len(unique_labels)}")
    return sorted(unique_labels)