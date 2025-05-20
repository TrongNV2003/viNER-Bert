# viNER-Bert
Name Entity Recognition from a input sentence.

## Dataset usage
In this repo, i use dataset: [PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19) for training and evaluating models.

## Installation
```sh
pip install -r requirements.txt
```

## Usage
Training and evaluating models:
```sh
bash train.sh
```

Inference models:
```sh
bash inference.sh
```

## Evaluate:
Evaluate metrics:

| Models                | Precision   | Recall      | F1 Score    | Latency (ms) |
|---------------------- |:-----------:|:-----------:|:-----------:|:------------:|
| PhoBert base          | 94.33       | 94.72       | 94.52       | 10.62        |
| PhoBert-base-v2       | 94.63       | 95.52       | 95.07       | 9.20         |
| PhoBert-large         | 94.06       | 94.59       | 94.32       | 15.91        |


### Future plans:
- Đóng thành API để inference
- Triển khai trên UI