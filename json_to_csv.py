
import os
import json
import pandas as pd
import argparse



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    return parser


LABELS = [
    'joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']
NUM_LABELS = len(LABELS)  # 8
COLUMNS = ['prefix', 'target', 'suffix']


def load_jsonl(jsonl_path):
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(json.loads(line))
    return lines


def parse_json(js, has_label=True):
    def parse_one(sample):
        inp = sample['input']
        text = inp['form']
        if inp['target']['form'] is None:
            target_st = 0
            target_ed = len(text)
        else:
            target_st = inp['target']['begin']
            target_ed = inp['target']['end']
        prefix = text[:target_st]
        target = text[target_st:target_ed]
        suffix = text[target_ed:]
        if has_label:
            labels = [0] * NUM_LABELS
            for i, name in enumerate(LABELS):
                assert sample['output'][name] in ('True', 'False')
                if sample['output'][name] == 'True':
                    labels[i] = 1
        else:
            labels = []
        ret = [prefix, target, suffix, *labels]
        return ret

    df = [parse_one(s) for s in js]
    df = pd.DataFrame(
        df, columns=COLUMNS + (LABELS if has_label else []))
    return df


def main():
    parser = get_parser()
    args = parser.parse_args()

    TRAIN_JSON_PATH = os.path.join(args.data_dir, 'nikluge-ea-2023-train.jsonl')
    VAL_JSON_PATH = os.path.join(args.data_dir, 'nikluge-ea-2023-dev.jsonl')
    TEST_JSON_PATH = os.path.join(args.data_dir, 'nikluge-ea-2023-test.jsonl')
    print(f'TRAIN:\t{TRAIN_JSON_PATH}')
    print(f'VAL:  \t{VAL_JSON_PATH}')
    print(f'TEST: \t{TEST_JSON_PATH}')

    train_json = load_jsonl(TRAIN_JSON_PATH)
    val_json = load_jsonl(VAL_JSON_PATH)
    test_json = load_jsonl(TEST_JSON_PATH)
    print(f'TRAIN:\t{len(train_json)}')
    print(f'VAL:  \t{len(val_json)}')
    print(f'TEST: \t{len(test_json)}')
    print()

    print('LABELS:')
    for label_idx, label_name in enumerate(LABELS):
        print(f'  {label_idx}: {label_name}')
    print('COLUMNS:', COLUMNS)


    train_df = parse_json(train_json)
    print('===== TRAIN DF =====', train_df, sep='\n')
    val_df = parse_json(val_json)
    print('===== VAL DF =====', val_df, sep='\n')
    test_df = parse_json(test_json, has_label=False)
    print('===== TEST DF =====', test_df, sep='\n')
    print()

    train_csv_path = os.path.join(args.output_dir, 'train.csv')
    val_csv_path = os.path.join(args.output_dir, 'val.csv')
    test_csv_path = os.path.join(args.output_dir, 'test.csv')
    print(f'SAVING TO:')
    print(f'TRAIN:\t{train_csv_path}')
    print(f'VAL:  \t{val_csv_path}')
    print(f'TEST: \t{test_csv_path}')

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
