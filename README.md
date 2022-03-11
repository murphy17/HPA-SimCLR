# Self-supervised learning of cell type specificity from immunohistochemical images
Michael Murphy, Stefanie Jegelka, Ernest Fraenkel
ISMB 2022

To run a model pretrained on HPA kidney images on a small example dataset of 10 genes:

`
git clone git@github.com:murphy17/HPA-SimCLR.git && cd HPA-SimCLR
conda env create -f environment.yml
conda activate HumanProteinAtlas
cd ./data && tar -xvzf example.tar.gz && cd ..
python run.py --image_dir ./data/example/images --gex_table ./data/example/kidney_rna.csv --output_dir ./data/example --checkpoint ./data/kidney.ckpt
`

- `--image_dir` must specify a folder of uniquely-named PNG images. Each PNG must have an accompanying JSON file of the same name with at minimum a string-valued field 'Gene'.
- `--gex_table` must specify a (genes x cell-types) CSV derived from an scRNA dataset, describing the mean counts of each gene in cells of each type. These genes must match the 
- `--output_dir` will be created if it does not exist, and will receive two CSV files: `embeddings.csv`, containing a vector-valued embedding for each image, and `scores.csv`, containing the classifier's predictions.
- `--checkpoint` is optional; if not specified, a model will be trained on the entire set of input images. Progress and checkpoints will be logged to `./lightning_logs` (the PyTorch Lightning default).
- See `run.py` for additional options.