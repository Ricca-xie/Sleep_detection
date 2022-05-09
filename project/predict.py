import numpy as np
import torch
import argparse
from my_dataloader import data_loader
from model.VGG16 import VGG
from tqdm import tqdm
from timm.utils import accuracy
from data_process import extract_hpss_features_sg
device = 'cuda:1'#'cpu'
device = torch.device(device)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='../data/single_test')
    parser.add_argument('--checkpoint_path', type=str, default='./model/checkpoints/vgg_pretrain.pt')
    parser.add_argument("--window_length", default=320, type=int, help="window_length") # sequence max length is 10 s, 240 frames.
    parser.add_argument("--window_shift", default=160, type=int, help="window_shift")
    parser.add_argument("--sample_rate", default=44100, type=int, help="target rate")
    parser.add_argument("--max_len", default=300, type=int, help="max_len")
    return parser.parse_args()


def predict(args):

    test_data, test_label = extract_hpss_features_sg(args.test_path, args.sample_rate, args.max_len, window_length=args.window_length, window_shift=args.window_shift)

    dev_data = data_loader(args.test_path, test_data, test_label, type='spec-n', is_data = True, batch_size=args.batchsize, is_shuffle=False, is_single = True)
    # model = model.load_state_dict(torch.load(args.checkpoint_path))
    # import pdb;pdb.set_trace()
    lable_map = {0:"norm", 1:"snore"}
    model = VGG()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)
    model.eval()
    loss_total = 0.
    acc1_total = 0.
    acc5_total = 0.

    with torch.no_grad():

        for step, (x, label) in enumerate(dev_data):
            x = x.to(device)
            pred = model(x)
            type = lable_map[pred.item()]
            print("Audio : ", type)


if __name__ == "__main__":
    args = parse_config()
    predict(args)
