# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import ttest_rel


def different_expression_analysis(exp1, exp2, alpha=0.05, num=-1):
    """
    Perform differential expression analysis between two expression matrices.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    alpha : float, optional
        Significance level for p-value threshold (default is 0.05).
    num : int, optional
        Number of top significant genes to return. If negative, return all significant genes (default is -1).
    
    Returns:
    np.ndarray
        Indices of differentially expressed genes.
    """
    n_gene = exp1.shape[0]
    p_values = [ttest_rel(exp1[i], exp2[i]).pvalue for i in range(n_gene)]
    
    if num < 0:
        diff_idx = [i for i, p in enumerate(p_values) if p < alpha]
    else:
        filtered_p_values = [(i, p) for i, p in enumerate(p_values) if p < alpha]
        sorted_filtered_p_values = sorted(filtered_p_values, key=lambda x: x[1])
        diff_idx = [i for i, p in sorted_filtered_p_values[:num]]
    
    return np.array(diff_idx)


def generate_multinormal_matrix(n, p):
    """
    Generate a random matrix with rows sampled from a multivariate normal distribution.
    
    Parameters:
    n : int
        Number of samples (rows).
    p : int
        Number of genes (columns).
    
    Returns:
    np.ndarray
        Generated random matrix (n * p).
    """
    cov_matrix = np.diag(np.random.uniform(0.1, 2, p))
    mean_vector = np.random.uniform(2, 5, p)
    random_matrix = np.random.multivariate_normal(mean_vector, cov_matrix, n)
    return np.abs(random_matrix)


def nonlinear_f(x, func_type):
    """
    Apply a nonlinear function to the input array.
    
    Parameters:
    x : np.ndarray
        Input array.
    func_type : int
        Type of nonlinear function to apply (1: linear, 2: exponential, 3: quadratic).
    
    Returns:
    np.ndarray
        Transformed array.
    """
    if func_type == 1:
        return x  # linear
    elif func_type == 2:
        return 0.02 * np.exp(x)  # exponential
    elif func_type == 3:
        return 0.5 * x**0.5  # quadratic
    
    
def f(matrix):
    """
    Apply nonlinear functions to each row of the input matrix.
    
    Parameters:
    matrix : np.ndarray
        Input matrix.
    
    Returns:
    np.ndarray
        Transformed matrix with nonlinear functions applied to each row.
    """
    f_matrix = matrix.copy()
    n_row = f_matrix.shape[0]
    func_types = np.random.randint(1, 4, n_row)
    for ii in range(f_matrix.shape[0]):
        f_matrix[ii] = nonlinear_f(f_matrix[ii], func_types[ii])
    return f_matrix


def generate_paired_matrix(exp1, diffgene=0.2, indegree=5, snr=2):
    """
    Generate a paired expression matrix with noise and regulatory relationships between genes.
    
    Parameters:
    exp1 : np.ndarray
        Original expression matrix (genes * samples).
    diffgene : float, optional
        Proportion of differentially expressed genes (default is 0.2).
    indegree : int, optional
        Average number of parent genes (default is 5).
    snr : float, optional
        Signal-to-noise ratio (default is 2).
    
    Returns:
    tuple:
        np.ndarray : Paired expression matrix (genes * samples).
        np.ndarray : True regulatory relationships (genes * genes).
    """
    n_gene, n_sample = exp1.shape
    signal_level = np.std(exp1, axis=1)
    noise_level = signal_level / snr
    
    exp2 = np.zeros_like(exp1)
    true_plan = np.zeros((n_gene, n_gene))
    
    diff_idx = np.random.choice(range(n_gene), size=int(diffgene*n_gene), replace=False)
    
    for ii in range(n_gene):
        rand_noise = np.random.normal(0, noise_level[ii], n_sample)
        
        if ii in diff_idx:
            n_parent = np.min([np.random.poisson(indegree-1), n_gene-1])
            idx_parent = np.random.choice(diff_idx[diff_idx!=ii], n_parent, replace=False)
            idx_parent = np.append(idx_parent, ii)
            w_parent = np.random.choice([1, -1], n_parent+1) * np.random.uniform(0.5, 2, n_parent+1)
            w_vec = np.zeros(n_gene)
            w_vec[idx_parent] = w_parent
            true_plan[:, ii] = w_vec
            exp2[ii] = np.sum(w_vec.reshape(n_gene, 1) * f(exp1), axis=0) + rand_noise
        else:
            true_plan[ii, ii] = 1
            exp2[ii] = exp1[ii] + rand_noise

    return np.abs(exp2), true_plan


def diff2all_matrix(y_score_diff, diff_idx, n_gene):
    """
    Expand the score matrix from differential genes to full genes.
    
    Parameters:
    y_score_diff : np.ndarray
        Score matrix on differential genes.
    diff_idx : np.ndarray
        Indices of differential genes.
    n_gene : int
        Total number of genes.
    
    Returns:
    np.ndarray
        Score matrix on full genes.
    """
    y_score = np.zeros((n_gene, n_gene))
    rows, cols = np.ix_(diff_idx, diff_idx)
    y_score[rows, cols] = y_score_diff
    return y_score


def auroc_aupr(y_true, y_score, mask):
    """
    Calculate the AUROC and AUPR for the predicted scores.
    
    Parameters:
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores.
    mask : np.ndarray
        Mask to select valid entries for evaluation.
    
    Returns:
    tuple:
        float : AUROC.
        float : AUPR.
    """
    fpr, tpr, _ = roc_curve(y_true[mask].flatten(), y_score[mask].flatten())
    auroc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true[mask].flatten(), y_score[mask].flatten())
    aupr = auc(rec, prec)
    return auroc, aupr
        

def early_precision(y_true, y_score, mask, top_num):
    """
    Calculate early precision for the top predicted scores.
    
    Parameters:
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores.
    mask : np.ndarray
        Mask to select valid entries for evaluation.
    top_num : int
        Number of top scores to consider for early precision.
    
    Returns:
    float
        Early precision for the top predicted scores.
    """
    top_indices = np.argpartition(y_score[mask].flatten(), -top_num)[-top_num:]
    top_true = y_true[mask].flatten()[top_indices]
    top_acc = np.sum(top_true) / len(top_true)
    return top_acc


def normalize_expression_matrix(exp):
    """Normalize the expression matrix by column sums."""
    column_sums = exp.sum(axis=0)
    return np.mean(column_sums) * exp / column_sums
