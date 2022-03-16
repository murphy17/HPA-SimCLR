from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection

def hgtest(mtx):
    ks = mtx.copy()
    Ks = mtx.sum(0)
    ns = mtx.sum(1)
    N = mtx.sum().sum()
    
    pvals = mtx.copy() * float('nan')

    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[1]):
            k = ks.iloc[i,j]
            n = ns.iloc[i]
            K = Ks.iloc[j]
            pval = hypergeom.sf(k-1,N,n,K)
            pvals.iloc[i,j] = pval
            
    return pvals
