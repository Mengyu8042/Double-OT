# -*- coding: utf-8 -*-
import numpy as np
import ot
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def double_ot(exp1: np.array, exp2: np.array, paired: bool = True,
            reg1: float = 0.05, reg2: float = 0.05) -> np.array:
    
    """
    Implement the Double OT method for inferring gene regulatory networks.
    
    Parameters:
    exp1 : np.array
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.array
        Expression matrix for condition 2 (genes * samples).
    reg1 : float, optional
        Marginal relaxation hyperparameter (default is 0.05).
    reg2 : float, optional
        Entropy regularization hyperparameter (default is 0.05).
    paired : bool, optional
        If True, assumes the samples are paired. If False, uses partial OT to align samples (default is True).
    
    Returns:
    np.array
        Optimal transport plan matrix.
    """
    if not paired:
        # If samples are not paired, perform PCA and partial OT to align samples
        n1 = exp1.shape[1]
        n2 = exp2.shape[1]
        
        # Perform PCA on the combined data
        pca = PCA()
        data = np.vstack((exp1.T, exp2.T))
        data_pca = pca.fit_transform(data)
        
        # Calculate cosine distance between samples
        sample_dist = calculate_distance_matrix(data_pca[:n1], data_pca[n1:], 'cosine')
        
        # Uniform weights for samples
        w1 = np.ones(n1)
        w2 = np.ones(n2)
        
        # Compute the partial OT plan
        otp = ot.partial.partial_wasserstein(w1, w2, sample_dist, m=min(n1, n2))
        
        # Match indices of exp1 to exp2 based on the transport plan
        match_idx = np.zeros(n1, dtype=int)
        for i in range(n1):
            match_idx[i] = np.where(otp[i] != 0)[0][0]
        
        # Reorder exp1 columns to match exp2
        exp1 = exp1[:, match_idx]
    
    # Calculate Spearman distance between genes
    gene_dist = calculate_distance_matrix(exp1, exp2, 'spearman')
    
    # Calculate mean expression level for each gene
    exp1_mass = np.mean(exp1, axis=1)
    exp2_mass = np.mean(exp2, axis=1)
    
    # Compute the unbalanced OT plan
    ot_plan = ot.unbalanced.sinkhorn_knopp_unbalanced(
        exp1_mass, exp2_mass, gene_dist, reg=reg2, reg_m=reg1)
    
    return ot_plan
            

def pearson_correlation(exp1, exp2):
    """
    Calculate the Pearson's correlation matrix between two expression matrices.
    
    Parameters:
    exp1 : np.ndarray
        First expression matrix (genes * samples).
    exp2 : np.ndarray
        Second expression matrix (genes * samples).
    
    Returns:
    np.ndarray
        Pearson's correlation matrix (genes * genes).
    """
    return np.corrcoef(exp1, exp2)[:exp1.shape[0], exp1.shape[0]:]


def spearman_correlation(exp1, exp2):
    """
    Calculate the Spearman's correlation matrix between two expression matrices.
    
    Parameters:
    exp1 : np.ndarray
        First expression matrix (genes * samples).
    exp2 : np.ndarray
        Second expression matrix (genes * samples).
    
    Returns:
    np.ndarray
        Spearman correlation matrix (genes * genes).
    """
    rho, _ = stats.spearmanr(exp1.T, exp2.T)
    return rho[:exp1.shape[0], exp1.shape[0]:]


def scipy_distance(exp1, exp2, metric):
    """
    Calculate the distance matrix between two expression matrices using specified metric.
    
    Parameters:
    exp1 : np.ndarray
        First expression matrix (genes * samples).
    exp2 : np.ndarray
        Second expression matrix (genes * samples).
    metric : str
        Distance metric ('euclidean', 'cosine', 'l1').
    
    Returns:
    np.ndarray
        Distance matrix (genes * genes).
    """
    if metric == 'euclidean':
        return cdist(exp1, exp2)**2
    elif metric == 'cosine':
        return cdist(exp1, exp2, 'cosine')
    elif metric == 'l1':
        return cdist(exp1, exp2, 'minkowski', p=1)
    
    
def calculate_distance_matrix(exp1, exp2, method):
    """
    Calculate the gene distance matrix using specified method.
    
    Parameters:
    exp1 : np.ndarray
        First expression matrix (genes * samples).
    exp2 : np.ndarray
        Second expression matrix (genes * samples).
    method : str
        Method to define gene distance ('euclidean', 'cosine', 'l1', 'pearson', 'spearman').
    
    Returns:
    np.ndarray
        Gene distance matrix (genes * genes).
    """
    if method in ['euclidean', 'cosine', 'l1']:
        gene_dist = scipy_distance(exp1, exp2, method)
    elif method in ['pearson', 'spearman']:
        if method == 'spearman':
            cor = spearman_correlation(exp1, exp2)
        elif method == 'pearson':
            cor = pearson_correlation(exp1, exp2)
        gene_dist = 1 - np.abs(cor)
    gene_dist /= np.max(gene_dist)
    return gene_dist

