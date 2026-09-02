

from argparse import ArgumentParser


def parse_train_args(args=None):
    parser = ArgumentParser(description="Use `python train.py pretrain|finetune ...`")
    parser.add_argument("stage", choices=("pretrain", "finetune"))
    return parser.parse_args(args)


def parse_predict_args(args=None):
    parser = ArgumentParser(description="Use `python predict.py ...`")
    return parser.parse_args(args)
