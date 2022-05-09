import argparse
import logging

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--train_path', type=str, default='../data/train')
    parser.add_argument('--test_path', type=str, default='../data/test')
    parser.add_argument('--checkpoint_path', type=str, default='./model/checkpoints/vgg')
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def logger_config(file_name=None):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    basic_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(basic_format, date_format)
    console_handler = logging.StreamHandler()  # output to console
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    if file_name:
        file_handler = logging.FileHandler(file_name, mode="w")  # output to file
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger