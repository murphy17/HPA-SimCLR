import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torchvision.models import efficientnet_b0, densenet121
import kornia.augmentation as ka
import pytorch_lightning as pl
import itertools as it
import json
import os
bash = lambda s: os.popen(s).read().rstrip().split('\n')
import numpy as np
import numpy.random as npr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from collections import OrderedDict

def cv2_imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

class ResamplingDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        json_fns,
        group_by=None,
        filter_by={},
        num_samples=1,
        transform=None,
        reader=None,
        random_seed=0
    ):
        self.json_fns = json_fns
        self.group_by = group_by
        self.filter_by = {k:set(v) for k,v in filter_by.items()}
        self.num_samples = num_samples
        self.reader = reader
        self.transform = transform
        self.rng = npr.RandomState(random_seed)
        
        self.json_data = OrderedDict()
        
        index = 0
        for json_fn in self.json_fns:
            data = read_json(json_fn)
            key = data[group_by]
            include = True
            for k, v in self.filter_by.items():
                if data[k] not in v:
                    include = False
            if include:
                if key not in self.json_data:
                    self.json_data[key] = []
                data['index'] = index
                index += 1
                self.json_data[key].append(data)
            
        self.keys = list(self.json_data)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[int(idx)]
        samples = []
        for i in range(self.num_samples):
            # reader should output a dict
            img, data = self.reader(npr.choice(self.json_data[key]))
            if self.transform:
                img = self.transform(img)['image']
            samples.append((img,data))
        return tuple(samples)
    
class JSONImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        image_size,
        batch_size,
        patch_size,
        name_field,
        filter_by=None,
        group_by=None,
        cache_dir=None,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.filter_by = filter_by
        self.group_by = group_by
        self.name_field = name_field
        
        if cache_dir:
            [self.cache_dir] = bash(f'echo {cache_dir}')
        else:
            self.cache_dir = None
        
        self._train_transform = A.Compose([
            A.PadIfNeeded(
                self.image_size,
                self.image_size,
                border_mode=1
            ),
            A.RandomCrop(
                2*self.patch_size, 
                2*self.patch_size,
                p=1.0
            ),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ])
        
        self._test_transform = A.Compose([
            A.PadIfNeeded(
                self.image_size,
                self.image_size,
                border_mode=1
            ),
            A.CenterCrop(
                self.image_size,
                self.image_size
            ),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ])
        
    def train_transform(self, x):
        return self._train_transform(image=x)
    
    def test_transform(self, x):
        return self._test_transform(image=x)
        
    # dict would be so much better, w/e
    def reader(self, data, keys=None):
        fn = data['_x']
#         subdir = int(fn.split('_')[0])//1000
        path = f'{self.data_dir}/{fn}'
        img = cv2_imread(path)
#         img = np.array(Image.open(path))
        if keys:
            data = {k:data[k] for k in keys}
        return img, data

    def setup(self, stage=None):
        if self.cache_dir:
            bash(f'mkdir {self.cache_dir} && ls {self.data_dir} | parallel -I% rsync -ruq {self.data_dir}/% {self.cache_dir}/')
            self.data_dir = self.cache_dir
            
        json_fns = bash(f'cd {self.data_dir}; find -name "*.json"')
        json_fns = [self.data_dir+'/'+fn for fn in json_fns]

        # HARD CODING - BAD!!!
        reader = lambda x: self.reader(x, ['Patient',self.name_field,'index'])
        
        self.train_dataset = ResamplingDataset(
            json_fns,
            group_by=self.group_by,
            filter_by=self.filter_by,
            reader=reader,
            num_samples=2,
            transform=self.train_transform
        )
        
        self.test_dataset = ResamplingDataset(
            json_fns,
            group_by=self.name_field,
            filter_by=self.filter_by,
            reader=reader,
            num_samples=1,
            transform=self.test_transform
        )

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        return dataloader

from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb
def sliced_da(img, ref, dims=[1,2]):
    shape = img.shape
    
#     img = img.clone()
    img = rgb_to_ycbcr(img[None]).squeeze(0)
    ref = rgb_to_ycbcr(ref[None]).squeeze(0)
    
    img = img.flatten(1)
    ref = ref.flatten(1)
    
    for i in dims:
        _, idxs = torch.sort(img[i],dim=-1)
        img[i,idxs], _ = torch.sort(ref[i],dim=-1)

    img = img.reshape(shape)
    
    img = ycbcr_to_rgb(img[None]).squeeze(0)
    
    return img

class ColorSwap(nn.Module):
    def __init__(self, p):
        super(ColorSwap,self).__init__()
        self.p = p
        
    def forward(self, x):
        n = len(x)
        i1s = torch.randperm(n)
        i1s, i2s = i1s[:int(self.p/2*n)], i1s[-int(self.p/2*n):]
        for i1, i2 in zip(i1s, i2s):
            x12 = sliced_da(x[i1],x[i2])
            x21 = sliced_da(x[i2],x[i1])
            x[i1], x[i2] = x12, x21
        return x
    