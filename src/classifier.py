import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp, softmax, log_softmax, logit
from scipy.sparse import diags
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

class SoftmaxRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha=0,
        lr=0.01,
        max_iters=100,
        verbose=False,
        random_state=0
    ):
        self.alpha = alpha
        self.max_iters = max_iters
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
        self.lr = lr
        self.random_state = random_state
        
        self.model = None
        
    def fit(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        w = sample_weight
        
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        X = self.scaler_.transform(X)
        
        N, D = X.shape
        K = y.shape[1]
        self.classes_ = np.arange(K)
            
        torch.manual_seed(self.random_state)
            
        device = 'cpu'
        
        X = torch.tensor(X,dtype=torch.float32,device=device)
        Y = torch.tensor(y,dtype=torch.float32,device=device)
        
        self.model = nn.Linear(D,K).to(device=device)
        self.model.train()
        
        opt = Adam(self.model.parameters(),lr=self.lr)

        iterator = range(self.max_iters)
        if self.verbose:
            iterator = tqdm(iterator,position=0)
        for t in iterator:
            Y_hat = self.model(X)
            loss = F.cross_entropy(Y_hat, Y, reduction='sum') + self.alpha * self.model.weight.square().sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        self.model = self.model.cpu()
        self.model.requires_grad_(False)
        self.model.eval()
    
    def predict_proba(self, X):
        X = self.scaler_.transform(np.array(X))
        X = torch.tensor(X,dtype=torch.float32)
        Y_hat = self.model(X)
        Y_hat = Y_hat.cpu().numpy()
        Y_hat = softmax(Y_hat, axis=1)
        return Y_hat
                       
    def predict(self, X):
        Y_hat = self.predict_proba(X)
        Y_hat = np.argmax(Y_hat, axis=1)
        return Y_hat
    
    def score(self, X, y, sample_weight=None):
        X = self.scaler_.transform(np.array(X))
        
        X = torch.tensor(X,dtype=torch.float32)
        Y = torch.tensor(y,dtype=torch.float32)
        
        Y_hat = self.model(X)
        loss = F.cross_entropy(Y_hat, Y, reduction='sum').item()
        
        return loss
    
# one-vs-rest platt scaling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold

def platt_scaling(scores, labels):
    lm = LogisticRegression(penalty='none')
    idx = list(set(scores.index)&set(labels.index))
    scores = {region:cross_val_predict(
        lm,
        scores.loc[idx],
        labels[region][idx],
        cv=KFold(5,shuffle=True,random_state=0),
        method='predict_proba'
    )[:,1] for region in labels.columns}
    scores = pd.DataFrame(
        scores,
        index=idx,
        columns=labels.columns
    )
    scores /= scores.sum(1).values[:,None]
    return scores