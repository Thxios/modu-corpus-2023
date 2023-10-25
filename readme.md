## 감정 분석 과제 모델 기술서

2023년 국립국어원 인공 지능 언어 능력 평가 대회 감정 분석 과제에 대한 모델 기술서 입니다.


### 재현


의존성 설치
```shell
pip3 install -r requirements.txt
```

데이터 전처리
```shell
python3 json_to_csv.py [과제_데이터_경로] --output_dir data
python3 preprocess.py \
  --train_csv data/train.csv data/val.csv \
  --test_csv data/train.csv data/test.csv \
  --output_dir processed
```

학습
```shell
python3 train.py \
  --output_dir results \
  --train_dataset processed/train_pro.csv \
  --test_dataset processed/test_pro.csv \
  --eval_size 0.05 \
  --batch_size 16 \
  --max_steps 10000 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --label_smoothing 0.1 \
  --seed 1234
```

제출 파일 생성
```shell
python make_submission.py \
  results/logit.npy \
  --test_jsonl_file [과제_테스트_데이터_경로]
```


