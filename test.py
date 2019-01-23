import argparse
import torch
from torchvision import transforms

import opt
#from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt, str2bool
from dataset import DDDataset

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_width', type=int, default=256)
parser.add_argument('--image_height', type=int, default=256)
parser.add_argument('--suffix', type=str, default='_N')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--depth_root', type=str, default=None)
parser.add_argument('--out_file', type=str, default='result.jpg')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--random_masks', type=str, default='false')

parser.add_argument('--gamma', type=float, default=1.5)
parser.add_argument('--exposure', type=float, default=1)
parser.add_argument('--white_level', type=float, default=1.0)
parser.add_argument('--black_level', type=float, default=0.0)
parser.add_argument('--random_images', type=str, default='false')
args = parser.parse_args()
random_masks = str2bool(args.random_masks)
random_images = str2bool(args.random_images)
use_depth = args.depth_root is not None

device = torch.device(args.device)

size = (args.image_height, args.image_width)

masks = args.mask_root if random_masks else [(args.mask_root, '_objectmask.png')]

dataset_val = DDDataset(args.root, (args.image_height,args.image_width),insuffixes = [args.suffix], masks=masks, train=False, auto_resize=not random_masks, random_masks=random_masks, depth_map=(args.depth_root, '_WO.exr') if use_depth else None)

model = PConvUNet(input_guides=1 if use_depth else 0).to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, args.out_file, gamma=args.gamma, exposure=args.exposure, black=args.black_level, white=args.white_level, random=random_images)
