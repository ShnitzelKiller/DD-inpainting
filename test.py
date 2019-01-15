import argparse
import torch
from torchvision import transforms

import opt
#from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from dataset import DDDataset

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_width', type=int, default=256)
parser.add_argument('--image_height', type=int, default=256)
parser.add_argument('--suffix', type=str, default='_N')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--out_file', type=str, default='result.jpg')

args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_height, args.image_width)

dataset_val = DDDataset(args.root, (args.image_height,args.image_width),insuffixes = [args.suffix], masks=[(args.mask_root, '_objectmask.png')], train=False)

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, args.out_file)
