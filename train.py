
import os
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import \
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, \
    Trainer, TrainingArguments, default_data_collator, EarlyStoppingCallback, \
    BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level=logging.INFO)


LABELS = [
    'joy',
    'anticipation',
    'trust',
    'surprise',
    'disgust',
    'fear',
    'anger',
    'sadness'
]
NUM_LABELS = len(LABELS)  # 8

sample_pre = '[이름][계정]님의 알티이벤트 [이름][계정]리유저블컵 드립백커피 당첨선물이 왔습니다ㅜㅜ'
sample_tg = '컵'
sample_suf = '도 너무 예쁘고 커피에 붙어있는 스티커가 너무 귀여워서 이걸 뜯을수나 있을지ㅜㅜㅠㅜ '



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, nargs='+', required=True)
    parser.add_argument('--eval_dataset', type=str, required=False)
    parser.add_argument('--test_dataset', type=str, required=False)

    parser.add_argument('--model_ckpt',
                        type=str, default='beomi/polyglot-ko-12.8b-safetensors')
    parser.add_argument('--batch_size',
                        type=int, default=16)
    parser.add_argument('--eval_batch_size',
                        type=int, default=64)
    parser.add_argument('--eval_size',
                        type=float, default=0.05)

    parser.add_argument('--learning_rate',
                        type=float, default=1e-4)
    parser.add_argument('--weight_decay',
                        type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type',
                        type=str, default='linear')

    parser.add_argument('--max_seq_len',
                        type=int, default=128)
    parser.add_argument('--label_smoothing',
                        type=float, default=0.1)

    parser.add_argument('--max_steps',
                        type=int, default=10000)
    parser.add_argument('--warmup_steps',
                        type=int, default=500)
    parser.add_argument('--early_stopping',
                        type=int, default=12)
    parser.add_argument('--resume_from_ckpt',
                        type=str, required=False)

    parser.add_argument('--lora_r',
                        type=int, default=16)
    parser.add_argument('--lora_alpha',
                        type=int, default=32)
    parser.add_argument('--lora_dropout',
                        type=float, default=0.05)

    parser.add_argument('--logging_steps',
                        type=int, default=25)
    parser.add_argument('--eval_steps',
                        type=int, default=250)
    parser.add_argument('--save_total_limit',
                        type=int, default=3)

    parser.add_argument('--seed',
                        type=int, default=1234)
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_prompt(pre, tg, suf):
    prompt = f'대상: "{tg}"\n' \
             f'글: "{pre}«{tg}»{suf}"'
    return prompt

def get_tokenizer(ckpt):
    print(f'===== LOADING TOKENIZER =====')
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    print(f'TOKENIZER:', tokenizer, sep='\n')
    sample_txt = get_prompt(sample_pre, sample_tg, sample_suf)
    token_ids = tokenizer(sample_txt)['input_ids']
    id_str = tokenizer.convert_ids_to_tokens(token_ids)
    decoded = tokenizer.decode(token_ids)
    print(f'TOKENIZE SAMPLE:', sample_txt, id_str, decoded, sep='\n')
    print(f'===== TOKENIZER LOADED =====')
    print()

    return tokenizer


def get_process_fn(tokenizer, prompt_fn, max_seq_len, label_smoothing):
    def process_fn(examples):
        texts = []
        for pre, tg, suf in zip(examples['prefix'],
                                examples['target'],
                                examples['suffix']):
            texts.append(prompt_fn(pre, tg, suf))

        ret = tokenizer(texts,
                        padding='max_length',
                        max_length=max_seq_len,
                        truncation=True)
        ret.pop('token_type_ids')
        ret['prompt'] = texts

        labels_batch = {k: examples[k] for k in examples.keys() if k in LABELS}
        if labels_batch:
            labels_matrix = np.zeros((len(examples['target']), NUM_LABELS))
            for idx, label in enumerate(LABELS):
                labels_matrix[:, idx] = labels_batch[label]
            if label_smoothing > 0:
                labels_matrix = label_smoothing + (1 - 2 * label_smoothing) * labels_matrix
            ret['labels'] = labels_matrix.tolist()

        return ret
    return process_fn


def get_dataset(
        train_dataset,
        process_fn,
        eval_dataset=None,
        test_dataset=None,
        eval_size=None,
        seed=None,
):
    print(f'===== LOADING DATASET =====')

    def load_df(path):
        return pd.read_csv(path, encoding='utf-8', keep_default_na=False)

    train_df = pd.concat(
        [load_df(p_) for p_ in train_dataset]).reset_index(drop=True)
    print(train_df.columns)

    dataset_df = dict(train=train_df)

    if eval_dataset is not None:
        eval_df = load_df(eval_dataset)
        dataset_df.update(eval=eval_df)
    elif eval_size is not None:
        assert seed is not None
        print("split train dataset to (train, eval)")
        train_df, eval_df = train_test_split(
            dataset_df['train'], test_size=eval_size, random_state=seed)
        dataset_df.update(train=train_df, eval=eval_df)

    test_df = load_df(test_dataset).reset_index(drop=True) \
        if test_dataset is not None else None


    print(f'TRAIN_DF:', dataset_df['train'], sep='\n')
    if 'eval' in dataset_df:
        print(f'EVAL_DF:', dataset_df['eval'], sep='\n')
    if test_dataset is not None:
        print(f'TEST_DF:', test_df, sep='\n')

    dataset = DatasetDict({
        k: Dataset.from_pandas(v) for k, v in dataset_df.items()
    })

    dataset = dataset.map(process_fn,
                          batched=True,
                          remove_columns=dataset['train'].column_names)
    if test_df is not None:
        test_ds = Dataset.from_pandas(test_df)
        test_ds = test_ds.map(process_fn,
                              batched=True,
                              remove_columns=test_ds.column_names)
        dataset['test'] = test_ds

    print(f'TOKENIZED DATASET:', dataset, sep='\n')
    print(f'SAMPLE TRAIN:', dataset['train'][0], sep='\n')
    if test_dataset is not None:
        print(f'SAMPLE TEST:', dataset['test'][0], sep='\n')

    print(f'===== DATASET LOADED =====')
    print()
    return dataset


def get_model(ckpt, lora_r, lora_alpha, lora_dropout, tokenizer=None):
    print(f'===== LOADING MODEL =====')
    config = AutoConfig.from_pretrained(
        ckpt,
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
    )
    config.use_cache = False
    if tokenizer is not None:
        config.pad_token_id = tokenizer.pad_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['query_key_value'],
        task_type="SEQ_CLS",
    )

    print(f'MODEL CONFIG:', config, sep='\n')
    print(f'BnB CONFIG:', bnb_config, sep='\n')
    print(f'LoRA CONFIG:', lora_config, sep='\n')

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        config=config,
        quantization_config=bnb_config,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    def print_trainable_parameters(model):
        n_trainable = 0
        n_all = 0
        for _, param in model.named_parameters():
            _n_param = param.numel()
            n_all += _n_param
            if param.requires_grad:
                n_trainable += _n_param
        print(f'trainable params: {n_trainable}, all params: {n_all}')

    print(f'MODEL: ', model, sep='\n')
    print_trainable_parameters(model)

    print(f'===== MODEL LOADED =====')
    print()

    return model


def train(
        model,
        dataset,
        train_args,
        early_stopping=None,
        resume_from_ckpt=None
):
    print(f'===== TRAIN =====')
    print(f'TRAIN_ARGS:', train_args, sep='\n')

    def multi_label_metrics(predictions, labels, threshold=0.5):
        probs = torch.sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = np.zeros(labels.shape)
        y_true[np.where(labels >= 0.5)] = 1

        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    callbacks = []
    if early_stopping is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping))
    trainer = Trainer(
        model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'] if 'eval' in dataset else None,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    if resume_from_ckpt is not None:
        print(f'RESUME_FROM_CKPT:', resume_from_ckpt, sep='\n')

    print(f'===== START TRAINING =====')
    trainer.train(resume_from_checkpoint=resume_from_ckpt)
    print(f'===== TRAIN DONE =====')
    print()


def make_logit(model, dataset, batch_size, save_dir):
    print(f'===== PREDICT =====')

    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    def predict(model):
        logits_ = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                logit = model(**batch).logits
                logits_.append(logit.detach().cpu().numpy())
        logits_ = np.concatenate(logits_)
        return logits_

    logits = predict(model)
    print(f'LOGITS:', logits, sep='\n')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'logit.npy'), logits)

    print(f'===== PREDICT DONE =====')
    print()




def main():
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'arguments.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    tokenizer = get_tokenizer(args.model_ckpt)
    process_fn = get_process_fn(tokenizer, get_prompt, args.max_seq_len, args.label_smoothing)

    dataset = get_dataset(
        args.train_dataset,
        process_fn,
        eval_dataset=args.eval_dataset,
        test_dataset=args.test_dataset,
        eval_size=args.eval_size,
        seed=args.seed
    )

    model = get_model(
        args.model_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        tokenizer=tokenizer,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=2,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        do_train=True,
        do_eval=True,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
    )

    train(model, dataset, train_args,
          early_stopping=args.early_stopping,
          resume_from_ckpt=args.resume_from_ckpt)

    if args.test_dataset is not None:
        make_logit(model, dataset['test'],
                   batch_size=args.eval_batch_size,
                   save_dir=output_dir)

    print('RESULT SAVED IN:', args.output_dir, sep='\n')


if __name__ == '__main__':
    main()
