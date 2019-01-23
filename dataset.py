# TODO: Augmentations

#from OpenEXR import *
#import Imath
#from exrload import loadEXR
import cv2
from functools import reduce
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageMorph
import os,sys
from util.nn_interpolate import nn_interpolate
import random
import itertools

def exr_loader(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    return img

def pil_loader(path, filters=[]):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        for fun in filters:
            img = fun(img)
        return img.convert('RGB')

def loader(path, filters=[]):
    if path[-3:] == 'png' or path[-3:] == 'jpg':
        return pil_loader(path, filters)
    elif path[-3:] == 'exr':
        return exr_loader(path)
    else:
        return None

class DDDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=None, crop=True, xf=None, condition=None, 
                 insuffixes = ['_Y.exr', '_ALL.exr', '_WO.exr'], 
                 outsuffixes = None, masks = ['_objectmask.png', '_planemask.png'], outsuffix='_N.exr',
                 postprocess=None, train=True, random_masks=False, auto_resize=False, depth_map=None, depth_maps=None, debug=False):
        self.suffixes = [(path, sfx)  if isinstance(sfx, str) else sfx for sfx in insuffixes]
        self.debug = debug
        self.needs_update = True
        self.auto_resize = auto_resize
        if outsuffixes is None:
            self.outsuffixes = [outsuffix]
        else:
            self.outsuffixes = outsuffixes
        self.outsuffixes = [(path, sfx) if isinstance(sfx, str) else sfx for sfx in self.outsuffixes]
        if depth_map is not None:
            self.depthsuffixes = [depth_map]
        elif depth_maps is not None:
            self.depthsuffixes = depth_maps
        else:
            self.depthsuffixes = None
        if self.depthsuffixes is not None:
            self.depthsuffixes = [(path, sfx) if isinstance(sfx, str) else sfx for sfx in self.depthsuffixes]
        self.random_masks = random_masks
        if random_masks:
            if isinstance(masks, str):
                self.mask_path = masks
                self.masks = [f.path for f in os.scandir(masks)]
                self.N_mask = len(self.masks)
                print('number of masks:', self.N_mask)
            else:
                self.masks = [m for m in itertools.chain.from_iterable([[f.path for f in os.scandir(path)] for path in masks])]
        else:
            self.masks = [(path, sfx) if isinstance(sfx, str) else sfx for sfx in masks]
        self.img_size = img_size
        self.crop = crop
        self.transform = xf
        self.postprocess = postprocess
        extensions = self.suffixes
        if not self.random_masks:
            extensions += self.masks
        if depth_map:
            extensions += self.depthsuffixes
            
        for basedir,ext in extensions:
            n = 0
            for x in os.scandir(basedir):
                n += 1
            print('directory count: (', basedir, ')', n)
            #currfiles = [d for d in os.scandir(basedir) if d.name.endswith(ext)]
            #print('count matching ext',ext,':', len(currfiles))
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            files = [{d.name.split("_")[0] for d in os.scandir(basedir) if d.name.endswith(ext)} for basedir,ext in extensions]
        else:
            files = [set([d.split("_")[0] for d in os.listdir(basedir) if d.endswith(ext)]) for basedir,ext in extensions]
        print('files with requested extensions:')
        for i, pair in enumerate(extensions):
            print('{} ({}): {}'.format(pair[0], pair[1], len(files[i])))
        self.files = list(reduce(lambda x, y: y.intersection(x), files))
        if condition is not None:
            self.files = list(filter(condition, self.files))
        
        self.filenames = {ext : [d.name for d in os.scandir(basedir) if d.name.endswith(ext) and d.name.split('_')[0] in self.files] for basedir,ext in extensions}
        for ext in self.filenames:
            self.filenames[ext].sort()
        if self.debug:
            for sets in zip(*[self.filenames[ext] for basedir,ext in extensions]):
                prefix = sets[0].split('_')[0]
                for name in sets:
                    if name.split('_')[0] != prefix:
                        print('prefixes: {} and {}'.format(prefix, name.split('_')[0]))
                        raise ValueError('dataset corrupt')

        n = len(self.files)
        print("total files:", n)
        split = n // 10
        if train:
            self.files = self.files[split:]
        else:
            self.files = self.files[:split]
        print("files in subset:", len(self.files))
    
    def preprocess(self, im):
        im = transforms.ToTensor()(im)
        im[torch.isinf(im)] = 0
        if self.needs_update and self.auto_resize:
            self.orig_size = im.shape[1:3]
            self.needs_update = False
            print('orig size:', self.orig_size)

        if self.transform is not None:
            im = self.transform(im)
        if self.img_size is not None:
            if not self.needs_update and im.shape[1:3] != self.orig_size and self.auto_resize:
                #resize all images to the same dimensions before cropping to ensure they're lined up
                np_im = im.numpy()
                #print('shape before resize', np_im.shape)
                #np_im = resize(np_im.transpose((1, 2, 0)), self.orig_size, anti_aliasing=False, mode='reflect').astype(np.float32).transpose((2, 0, 1))
                np_im = nn_interpolate(np_im, self.orig_size)
                im = torch.Tensor(np_im)
                #print('shape after resize:', np_im.shape)
            if self.crop and (im.shape[1] != self.img_size[0] or im.shape[2] != self.img_size[1]):
                center = im.shape[1] // 2, im.shape[2] // 2
                margin = (im.shape[1] - self.img_size[0]) // 2, (im.shape[2] - self.img_size[1]) // 2
                im = im[:,margin[0]:margin[0]+self.img_size[0],margin[1]:margin[1]+self.img_size[1]]
                #print('cropped size: ', im.shape)
            else:
                np_im = im.numpy()
                #print('original size: ', np_im.shape)
                np_im = resize(np_im.transpose((1, 2, 0)), self.img_size, anti_aliasing=True, mode='reflect').astype(np.float32).transpose((2, 0, 1))
                #print('resized: ', np_im.shape)
                return torch.Tensor(np_im)
        return im

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,i):
        dilate = lambda im: im.filter(ImageFilter.MinFilter(5))
        p = [path + os.sep + self.filenames[sfx][i] for path,sfx in self.suffixes]
        if self.debug:
            prefix = p[0].split('/')[-1].split('_')[0]
            for ps, suff in zip(p, self.suffixes):
                if not ps.endswith(suff):
                    raise ValueError('invalid p entry: {}'.format(ps))
        if self.random_masks:
            m = [self.masks[random.randint(0, self.N_mask-1)]]
            #print('using mask',m)
        else:
            m = [path + os.sep + self.filenames[sfx][i] for path,sfx in self.masks]
        if self.debug:
            for ms, suff in zip(m, self.masks):
                if not ms.split('/')[-1].startswith(prefix) or not ms.endswith(suff):
                    raise ValueError('invalid m entry: {} (prefix {}, index {})'.format(ms, prefix, i))
        if self.depthsuffixes:
            #print(self.depthsuffixes)
            d = [path + os.sep + self.filenames[sfx][i] for path,sfx in self.depthsuffixes]
            if self.debug:
                for ds, suff in zip(d, self.depthsuffixes):
                    if not ds.split('/')[-1].startswith(prefix) or not ds.endswith(suff):
                        raise ValueError('invalid d entry: {} (prefix {})'.format(ds, prefix))
            #print('d',d)
        #o = [path + os.sep + self.files[i] + sfx for path,sfx in self.outsuffixes]
        #print('p',p)
        #print('m',m)
        #print('o',o)
        ims = [self.preprocess(loader(i)) for i in p]
        masks = [self.preprocess(loader(i))[0:1] for i in m] #can also dilate with loader(i, [dilate])
        if self.depthsuffixes:
            depthmaps = [self.preprocess(loader(i))[0:1] for i in d]
        
        #imo = [self.preprocess(loader(i)) for i in o]
        if self.postprocess is not None:
            #return makeDict([tensor.unsqueeze(0) for tensor in self.postprocess(torch.cat(ims+masks, 0), torch.cat(imo, 0))])
            print('postprocess not implemented')
            exit(1)
        else:
            image = ims[0]
            allmasks = torch.round(torch.cat(masks + masks + masks, 0))
            #print('mask size:',allmasks.shape)
            #print('images size:',allimages.shape)
            
            #TODO: Decide what suffixes to consider ground truth/input image for inpainting (for now use _N (comes after "lighting correction" stage)
            gt = image
            image = image * allmasks
            if self.depthsuffixes:
                depthmap = depthmaps[0]
                image = torch.cat([image, depthmap], 0)
            return image, allmasks, gt #, torch.cat(imo, 0).unsqueeze(0)

def parseDDFilename(path):
    s = path.split("_")
    return {
        "uid": s[0][:-len("Unique")],
        "texid": s[1],
        "mainshape": s[2][:5],
        "envmap": "_".join([s[2][5:]] + s[3:-5]),
        "camtheta": int(s[-5][len("Theta"):]),
        "camphi": int(s[-4][len("Phi"):-len("Light")]),
        "lighttheta": int(s[-3][len("Theta"):]),
        "lightphi": int(s[-2][len("Phi"):]),
        "alpha": float(s[-1])/10000
    }
