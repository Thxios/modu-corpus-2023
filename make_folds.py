

import os
import pandas as pd
from sklearn.model_selection import KFold
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('-k,', '--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)

    return parser


def make_folds(csv_path, output_dir, k, seed):
    kf = KFold(k, shuffle=True, random_state=seed)
    df = pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False)
    os.makedirs(output_dir, exist_ok=True)
    out_name, _ = os.path.splitext(os.path.basename(csv_path))
    for i, (_, test_idx) in enumerate(kf.split(df)):
        fold_df = df.iloc[test_idx]
        save_name = os.path.join(output_dir, f'{out_name}_fold{i}.csv')
        print(save_name)
        fold_df.to_csv(save_name, index=False, encoding='utf-8')


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    make_folds(args.csv_path, args.output_dir, args.n_folds, args.seed)


if __name__ == '__main__':
    main()

