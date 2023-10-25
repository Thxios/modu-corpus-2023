
import os
import re
import pandas as pd
import argparse



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, nargs='+', required=True)
    parser.add_argument('--test_csv', type=str, nargs='+', required=True)

    parser.add_argument('--remove_repeat',
                        type=int, default=3)
    parser.add_argument('--name_token',
                        type=str, default='[이름]')
    parser.add_argument('--account_token',
                        type=str, default='[계정]')
    return parser


def make_replace_fn(pat, to):
    def replace_fn(text):
        return re.sub(pat, to, text)
    return replace_fn


def make_repeat_replace_fn(c, n):
    c_ = c
    if c in '.^$*+?{}[]\\|()':
        c_ = '\\' + c
    pat = f'{c_}{{{n},}}'
    return make_replace_fn(re.compile(pat), c*n)


def replace_all_repeats(n):
    assert int(n) > 1
    pat = f'(.)\\1{{{int(n)},}}'
    pat = re.compile(pat)
    def replace_fn(s):
        findings = set(re.findall(pat, s))
        for rc in findings:
            s = make_repeat_replace_fn(rc, n)(s)
        return s
    return replace_fn


def process(df: pd.DataFrame, replace_pats, remove_repeat_n):
    for col in ('prefix', 'target', 'suffix'):
        for pat, to in replace_pats:
            df[col] = df[col].apply(make_replace_fn(pat, to))
        df[col] = df[col].apply(replace_all_repeats(remove_repeat_n))
    return df


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f'TRAIN_CSV {len(args.train_csv)}:', *args.train_csv, sep='\n')
    print(f'TEST_CSV {len(args.test_csv)}:', *args.test_csv, sep='\n')
    train_df = pd.concat([
        pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False)
        for csv_path in args.train_csv
    ])
    test_df = pd.concat([
        pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False)
        for csv_path in args.test_csv
    ])
    print()

    replace_pats = [
        (re.compile(r'&others&'), ''),
        (re.compile(r'&name[0-9]*&'), args.name_token),
        (re.compile(r'\(?&account[0-9]*&\)?'), args.account_token),
    ]
    print(f'REPLACE_PATTERNS', *replace_pats, sep='\n')
    print(f'REPLACE_REPEAT_N: {args.replace_repeat_n}')
    print()

    train_processed = process(train_df, replace_pats, args.replace_repeat_n)
    print(f'TRAIN_PROCESSED', train_processed, sep='\n')
    test_processed = process(test_df, replace_pats, args.replace_repeat_n)
    print(f'TEST_PROCESSED', train_processed, sep='\n')
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    train_processed.to_csv(os.path.join(
        args.output_dir, 'train_pro.csv'), index=False, encoding='utf-8')
    test_processed.to_csv(os.path.join(
        args.output_dir, 'test_pro.csv'), index=False, encoding='utf-8')
    print(f'SAVED IN {args.output_dir}')



if __name__ == '__main__':
    main()

