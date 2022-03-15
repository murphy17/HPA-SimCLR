import matplotlib.pyplot as plt
import seaborn as sns

cell_type_colors = {}

cell_type_colors['kidney'] = {
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

cell_type_colors['testis'] = {
    'myoid cells': sns.color_palette("hls", 11, as_cmap=True)(0.00),
    'leydig cells': sns.color_palette("hls", 11, as_cmap=True)(0.09),
    'sertoli cells': sns.color_palette("hls", 11, as_cmap=True)(0.18),
    'endothelial cells': sns.color_palette("hls", 11, as_cmap=True)(0.27),
    'macrophages': sns.color_palette("hls", 11, as_cmap=True)(0.36),
    #
    'spermatogonial stem cells': sns.color_palette("hls", 11, as_cmap=True)(0.45),
    'round spermatids': sns.color_palette("hls", 11, as_cmap=True)(0.54),
    'elongated spermatids': sns.color_palette("hls", 11, as_cmap=True)(0.63),
    'early primary spermatocytes': sns.color_palette("hls", 11, as_cmap=True)(0.72),
    'late primary spermatocytes': sns.color_palette("hls", 11, as_cmap=True)(0.81),
    'sperm cells': sns.color_palette("hls", 11, as_cmap=True)(0.90),
}
