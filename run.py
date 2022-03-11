# Self-supervised learning of cell type specificity from immunohistochemical images
# Michael Murphy, Stefanie Jegelka, Ernest Fraenkel

# to run pretrained model on small example dataset of 10 genes:

# conda env create -f environment.yml
# conda activate HumanProteinAtlas
# cd ./data && tar -xvzf example.tar.gz && cd ..
# python run.py --image_dir ./data/example/images --gex_table ./data/example/kidney_rna.csv --output_dir ./data/example --checkpoint ./data/kidney.ckpt

import argparse
import numpy as np
import numpy.random as npr
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd
import os

from src.model import ContrastiveEmbedding
from src.datamodule import ContrastiveDataModule, IMAGE, NAME
from src.classifier import SoftmaxRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to folder of PNGs with JSON metadata')
    parser.add_argument('--gex_table', type=str, required=True,
                        help='Path to (genes x cell-types) CSV matrix of mean scRNA expression per type')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output folder')
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                        help='Skip training and use this model checkpoint')
    parser.add_argument('--image_ext', type=str, required=False, default='png',
                        help='Image extension')
    parser.add_argument('--group_by', type=str, required=False, default='Gene',
                        help='JSON metadata field to group images by')
    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--patch_size', type=int, required=False, default=256)
    parser.add_argument('--batch_size', type=int, required=False, default=150)
    parser.add_argument('--num_workers', type=int, required=False, default=16)
    parser.add_argument('--num_epochs', type=int, required=False, default=1000)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--temperature', type=float, required=False, default=1.0)
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-4)
    parser.add_argument('--encoder', type=str, required=False, default='densenet121')
    parser.add_argument('--random_state', type=int, required=False, default=0)
    parser.add_argument('--precision', type=int, required=False, default=32)
    parser.add_argument('--gpus', type=int, required=False, default=0)
    args = parser.parse_args()

    assert os.path.exists(args.image_dir)
    assert os.path.exists(args.gex_table)
    if args.checkpoint:
        assert os.path.exists(args.checkpoint)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    ###

    print('Initializing model')

    torch.manual_seed(args.random_state)
    npr.seed(args.random_state)
    
    rna = pd.read_csv(args.gex_table, sep=',', index_col=0).astype(float)
    rna.values[:] = rna.values / rna.values.sum(1,keepdims=True)

    dm = ContrastiveDataModule(
        args.image_dir,
        image_ext=args.image_ext,
        image_size=args.image_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        indicator=lambda item: True,
        grouper=lambda item: item[args.group_by],
        num_workers=args.num_workers,
        random_state=0
    )

    dm.setup()
    print(len(dm.dataset), 'images')
    print(len(dm.train_dataset), 'genes')
    print(rna.shape[1], 'cell types')
    
    ###
    
    if args.checkpoint is not None:
        print('Loading checkpoint',args.checkpoint)

        model = ContrastiveEmbedding.load_from_checkpoint(
            args.checkpoint,
            embedding_dim=args.embedding_dim,
            patch_size=args.patch_size,
            encoder_type=args.encoder,
            temperature=args.temperature,
            learning_rate=args.learning_rate
        )

    else:
        print('Training model')

        model = ContrastiveEmbedding(
            embedding_dim=args.embedding_dim,
            patch_size=args.patch_size,
            encoder_type=args.encoder,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
        )

        trainer = Trainer(
            gpus=args.gpus,
            precision=args.precision,
            strategy='dp',
            min_epochs=args.num_epochs,
            max_epochs=args.num_epochs
        )
        
        trainer.fit(model, dm)

    ###

    print('Embedding images')

    model.eval()
    if args.gpus > 0:
        model = model.cuda()

    embeddings = []
    images = []
    genes = []
    for item in tqdm(dm.test_dataset, position=0):
        with torch.no_grad():
            z = model(item[IMAGE].to(model.device).unsqueeze(0))
        images.append(item[NAME])
        genes.append(item[args.group_by])
        embeddings.append(z.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    ###

    print('Saving embeddings to', f'{args.output_dir}/embeddings.csv')

    embeddings = pd.DataFrame(embeddings, index=images)
    embeddings.to_csv(f'{args.output_dir}/embeddings.csv', sep=',')

    ###

    print('Fitting classifier')

    genes = pd.Series(genes,index=images,name='Gene')

    df = embeddings.join(genes).join(rna, on='Gene', how='inner')

    clf = SoftmaxRegression(max_iters=1000, lr=0.01, verbose=True)
    
    clf.fit(df[embeddings.columns].values, df[rna.columns].values)

    scores = clf.predict_proba(embeddings.values)
    scores = pd.DataFrame(scores, index=embeddings.index, columns=rna.columns)

    ###

    print('Saving per-image cell type scores to', f'{args.output_dir}/scores.csv')

    scores.to_csv(f'{args.output_dir}/scores.csv', sep=',')
    
if __name__ == "__main__":
    main()