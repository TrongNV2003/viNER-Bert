# viNER-Bert
Recognize entities from a input sentence.

## Dataset usage
In this repo, i choose dataset: [PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19) for training and evaluating models.

## Installation
```sh
pip install -r requirements.txt
```

## Usage
training and evaluating models:
```sh
python main.py
```

## Evaluate:
- Evaluate metrics:

| Models                | Precision   | Recall      | F1 Score    | Latency (ms) |
|---------------------- |:-----------:|:-----------:|:-----------:|:------------:|
| PhoBert base          | 89.52       | 93.55       | 91.49       | 7.95         |



### Future plans:
- Đóng thành API để inference
- Triển khai trên UI