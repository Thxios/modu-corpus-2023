
import os
import json
import numpy as np
import torch
import argparse



LABELS = [
    'joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']
NUM_LABELS = len(LABELS)  # 8
label2id = {label: i for i, label in enumerate(LABELS)}

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('logit_npy_file', type=str, required=True)
    parser.add_argument('submission_name', type=str, required=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test_jsonl_file', type=str, required=True)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f'load logits from {args.logit_npy_file}')
    logits = np.load(args.logit_npy_file)
    probs: np.ndarray = torch.sigmoid(torch.from_numpy(logits)).numpy()
    print(f'probs {probs.shape}:', probs, sep='\n')

    test_json = []
    with open(args.test_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_json.append(json.loads(line))
    assert len(test_json) == len(probs)

    pred = probs > args.threshold
    n_true = np.sum(pred)
    print(f'pred - True {n_true}:', probs, sep='\n')

    for i, sample in enumerate(test_json):
        output = {label: 'False' for label in LABELS}
        for label_idx in range(NUM_LABELS):
            if pred[i][label_idx]:
                output[LABELS[label_idx]] = 'True'
        sample['output'] = output

    if args.submission_name is None:
        submission_name = args.logit_npy_file
    else:
        submission_name = args.submission_name
    submission_name = submission_name + '.jsonl'

    with open(submission_name, 'w', encoding='utf-8') as f:
        for sample in test_json:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    print(f'saved in: {submission_name}')


if __name__ == '__main__':
    main()


