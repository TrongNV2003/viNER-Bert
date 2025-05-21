python -m ner.inference \
    --model vinai/phobert-base-v2 \
    --output_dir ./models \
    --input "Xin chào các bạn, tôi là Trọng, tôi năm nay 21 tuổi. Tôi đến từ Hà Nội, mảnh đất của Việt Nam. Hôm nay tôi đi khám ở bệnh viện Bạch Mai. Tôi vào thăm bệnh nhân bị COVID-19 chiều ngày 15 tháng 9" \
    --record_output_file output_infer.json \
    