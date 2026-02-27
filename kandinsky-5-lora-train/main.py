import torch
import logging
import warnings
from argparse import ArgumentParser
from transformers.utils.logging import disable_progress_bar

from train.utils import load_conf
from train.lora_train import run


def disable_warnings():
    disable_progress_bar()
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
    )


if __name__ == '__main__':
    disable_warnings()
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument('--reuse-config', type=int, default=1,
                        help='option to disable reusing config stored previously')
    args = parser.parse_args()
    conf = load_conf(args.config)
    run(conf)
