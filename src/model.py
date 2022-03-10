import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from pytorch_lightning import LightningModule
from kornia import augmentation as ka

EPS = 1e-8

class ContrastiveEmbedding(LightningModule):
    def __init__(
        self,
        embedding_dim,
        patch_size,
        encoder_type='densenet121',
        temperature=1.0,
        learning_rate=5e-4,
        negative_masking=True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.encoder_type = encoder_type
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.negative_masking = negative_masking
        
        encoder, encoder_dim = self._get_encoder(encoder_type)
        
        self.encoder = nn.Sequential(
            ka.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            encoder,
            nn.Conv2d(encoder_dim, embedding_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self._transform = nn.Sequential(
            ka.RandomAffine(
                degrees=180,
                scale=(0.8,1.2),
                padding_mode=1,
                p=0.5
            ),
            ka.RandomCrop(
                (self.patch_size, self.patch_size),
                p=1.0
            ),
            ka.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.1, 
                p=0.5
            )
        )
        
    def transform(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = self._transform(x)
            return x
        
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def _get_encoder(self, encoder_type):
        encoder = getattr(models, encoder_type)(pretrained=True)
        encoder = list(encoder.children())
        output_dim = None
        if isinstance(encoder[-1],nn.Linear):
            output_dim = encoder[-1].weight.shape[1]
            encoder = encoder[:-1]
        if isinstance(encoder[-1],nn.AdaptiveAvgPool2d):
            encoder = encoder[:-1]
        encoder = nn.Sequential(*encoder)
        return encoder, output_dim

    def infonce_loss(self, z1, z2, temperature, mask=None, symmetrize=True):
        # Avoid precision issues with FP16
        with torch.cuda.amp.autocast(enabled=False):
            z1 = z1.to(torch.float32)
            z2 = z2.to(torch.float32)
            
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            
            sim12 = z1 @ z2.T / temperature

            pos_loss = -torch.diag(sim12).mean()
            
            if mask is not None:
                sim12 += mask.log()
            
            if symmetrize:
                neg_loss = (
                    torch.logsumexp(sim12, dim=0).mean() +
                    torch.logsumexp(sim12, dim=1).mean()
                ) / 2
            else:
                neg_loss = torch.logsumexp(sim12, dim=0).mean()

            return pos_loss + neg_loss

    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch
        x1, x2 = batch1['image'], batch2['image']
        N = x1.shape[0]
        d1, d2 = batch1['Patient'], batch2['Patient']
        
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        z1 = self.projection_head(z1)
        z2 = self.projection_head(z2)
        
        if self.negative_masking:
            neg_mask = (d1.view(-1,1) == d2.view(1,-1)).float()
            neg_mask[range(N),range(N)] = 0
            renorm = neg_mask.sum(1,keepdim=True).clip(EPS,float('inf')) * (N-1)
            neg_mask = neg_mask / renorm
            neg_mask[range(N),range(N)] = 1
        else:
            neg_mask = None
            
        infonce_loss = self.infonce_loss(
            z1, z2,
            mask=neg_mask,
            temperature=self.temperature,
            symmetrize=True
        )
        
        self.log('infonce_loss', infonce_loss)
        
        return infonce_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer
    