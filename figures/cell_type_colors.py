import matplotlib.pyplot as plt

cell_type_colors = {
    'glomerular visceral epithelial cell':plt.get_cmap('Reds')(0.9),
    'mesangial cell':plt.get_cmap('Reds')(0.8),
    'parietal epithelial cell':plt.get_cmap('Reds')(0.7),
    'kidney capillary endothelial cell':plt.get_cmap('Reds')(0.6),
    'leukocyte':plt.get_cmap('Reds')(0.5),
    #
    'fibroblast':'orange',
    #
    'epithelial cell of proximal tubule':plt.get_cmap('Greens')(0.8),
    #
    'kidney loop of Henle thick ascending limb epithelial cell':plt.get_cmap('Blues')(0.9),
    'kidney distal convoluted tubule epithelial cell':plt.get_cmap('Blues')(0.7),
    'kidney connecting tubule epithelial cell':plt.get_cmap('Blues')(0.5),
    #
    'renal alpha-intercalated cell':plt.get_cmap('Purples')(0.9),
    'renal beta-intercalated cell':plt.get_cmap('Purples')(0.7),
    'renal principal cell':plt.get_cmap('Purples')(0.5),
}
