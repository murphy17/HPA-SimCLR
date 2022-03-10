import numpy as np
import numpy.random as npr
import json
from collections import defaultdict
from os.path import abspath, expandvars
from .util import bash
import cv2
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from pytorch_lightning import LightningDataModule
from torchvision import transforms

IMAGE = 'image'

def cv2_imread(path):
    path = str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class apply_transform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = self.transform(item)
        return item
    
    def __len__(self):
        return len(self.dataset)

class GroupSampler(Dataset):
    def __init__(
        self, 
        dataset_iter,
        group_fn,
        num_samples=1,
        transform=None, 
        random_state=0
    ):
        super().__init__()
        
        self.dataset_iter = dataset_iter
        self.group_fn = group_fn
        self.num_samples = num_samples
        self.transform = transform
        self.rng = npr.RandomState(random_state)
        
        self.groups = defaultdict(list)
        for item in dataset_iter:
            g = self.group_fn(item)
            self.groups[g].append(item)
        self.keys = list(self.groups.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        items = []
        for i in range(self.num_samples):
            item = self.rng.choice(self.groups[self.keys[idx]], replace=True)
            if self.transform:
                item = self.transform(item)
            items.append(item)
        return items
    
class ContrastiveDataModule(LightningDataModule):
    def __init__(
        self, 
        image_dir,
        image_ext,
        image_size,
        patch_size,
        batch_size,
        indicator=lambda _: True,
        grouper=lambda item: item[IMAGE],
        cache_dir=None,
        num_workers=0,
        random_state=0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.indicator = indicator
        self.grouper = grouper
        self.cache_dir = expandvars(cache_dir) if cache_dir else None
        self.num_workers = num_workers
        self.random_state = random_state

    def setup(self, stage=None):
        if self.cache_dir:
            bash(f'mkdir -p {self.cache_dir}')
            img_paths = bash(f'cd {self.image_dir} && find -type f')
            cache_paths = bash(f'cd {self.cache_dir} && find -type f')
            if sorted(img_paths) != sorted(cache_paths):
                bash(f'ls {self.image_dir} | parallel -I% rsync -ruq {self.image_dir}/% {self.cache_dir}/')
            img_dir = self.cache_dir
        else:
            img_dir = self.img_dir

        subjects = []
        
        transform = tio.CropOrPad((self.image_size,self.image_size,1),padding_mode=1)
        
        json_paths = bash(f'find {img_dir} -type f -name "*.json" -not -path "*/.*"')
        for json_path in json_paths:
            with open(json_path,'r') as f:
                data = json.load(f)
            if self.indicator(data):
                img_path = json_path.replace('.json','.'+self.image_ext)
                data[IMAGE] = tio.ScalarImage(
                    img_path, 
                    reader=lambda path: (cv2_imread(path), None),
                    transform=transform,
                )
                subjects.append(tio.Subject(**data))
                
        self.dataset = tio.SubjectsDataset(subjects)
        self.train_dataset = GroupSampler(
            self.dataset.dry_iter(),
            num_samples=2,
            group_fn=self.grouper,
            random_state=self.random_state,
            transform=self._transform
        )
        self.test_dataset = apply_transform(self.dataset, self._transform)
        
    def _transform(self, item):
        item = {**item}
        item[IMAGE] = item[IMAGE][tio.DATA].squeeze(-1)
        item[IMAGE] = item[IMAGE].float() / 255
        return item
        
    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            #prefetch_factor=2,
            #pin_memory=True,
            #persistent_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size // ((self.image_size // self.patch_size) ** 2),
            shuffle=False,
            drop_last=False
        )
        return dataloader
    