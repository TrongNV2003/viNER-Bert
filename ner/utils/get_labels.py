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

def get_full_labels(unique_labels):
    full_labels = set(unique_labels)
    for label in unique_labels:
        if label.startswith("B-"):
            i_label = "I-" + label[2:]
            if i_label not in full_labels:
                full_labels.add(i_label)
    print(f"\nDanh sách nhãn đầy đủ: {sorted(full_labels)}")
    print(f"Tổng số lượng nhãn đầy đủ: {len(full_labels)}")
    return sorted(full_labels)