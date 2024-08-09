# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_roc_pr_curves(y_true, y_scores, mask, methods, colors, indegree):
    """Plot ROC and PR curves for the methods."""
    fig, axs = plt.subplots(2, 1, figsize=(3, 6), gridspec_kw={'hspace': 0.6, 'wspace': 0.25, 'bottom': 0.2})
    
    for ii, method in enumerate(methods):
        fpr, tpr, _ = roc_curve(y_true[mask].flatten(), y_scores[ii][mask].flatten())
        auc_roc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, color=colors[ii], label=f'{method} (AUC = {auc_roc:.3f})') 
    axs[0].set_xlabel('False Positive Rate') 
    axs[0].set_ylabel('True Positive Rate') 
    axs[0].set_title(f'ROC Curve (indegree = {indegree})')       
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))                

    for ii, method in enumerate(methods):
        prec, rec, _ = precision_recall_curve(y_true[mask].flatten(), y_scores[ii][mask].flatten())
        auc_pr = auc(rec, prec)
        axs[1].plot(rec, prec, color=colors[ii], label=f'{method} (AUC = {auc_pr:.3f})')  
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title(f'PR Curve (indegree = {indegree})')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()
    

def plot_bar_charts(auroc_list, aupr_list, ep_list, indegrees, methods, colors,
                    n_gene, n_sample, diffgene):
    """Plot AUROC, AUPR, and EP bar charts for mean results with error bars."""
    auroc_mean = np.mean(auroc_list, axis=2)
    auroc_std = np.std(auroc_list, axis=2)
    aupr_mean = np.mean(aupr_list, axis=2)
    aupr_std = np.std(aupr_list, axis=2)
    ep_mean = np.mean(ep_list, axis=2)
    ep_std = np.std(ep_list, axis=2)
    
    bar_width = 0.09  # bar width
    index = np.arange(len(indegrees))  # index for indegrees
    
    fig, axs = plt.subplots(1, 3, figsize=(13, 3), gridspec_kw={'hspace': 0.2, 'wspace': 0.3, 'bottom': 0.2})
    fig.suptitle(fr'p = {n_gene}, n = {n_sample}, $\alpha$ = {int(diffgene * 100)}%', fontsize=16)

    for i, method in enumerate(methods):
        axs[0].bar(index + bar_width * i, auroc_mean[:, i], bar_width, 
                   yerr=auroc_std[:, i], label=method, color=colors[i], alpha=0.8,
                   error_kw={'capsize': 1.5, 'capthick': 0.8})
    axs[0].set_xlabel(r'In-degree parameter $\lambda$', fontsize=14)
    axs[0].set_ylabel('AUROC', fontsize=14)
    axs[0].set_xticks(index + bar_width * (len(methods) - 1) / 2)
    axs[0].set_xticklabels(indegrees, fontsize=14)
    axs[0].tick_params(axis='y', labelsize=14)

    for i, method in enumerate(methods):
        rects = axs[1].bar(index + bar_width * i, aupr_mean[:, i], bar_width, 
                           yerr=aupr_std[:, i], label=method, color=colors[i], alpha=0.8,
                           error_kw={'capsize': 1.5, 'capthick': 0.8})  
        if i == 0:
            for rect in rects:
                height = rect.get_height() 
                axs[1].annotate(f'{height:.1e}', xy=(rect.get_x(), height),
                                xytext=(2, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, rotation='vertical')
    axs[1].set_xlabel(r'In-degree parameter $\lambda$', fontsize=14)
    axs[1].set_ylabel('AUPR', fontsize=14)
    axs[1].set_xticks(index + bar_width * (len(methods) - 1) / 2)
    axs[1].set_xticklabels(indegrees, fontsize=14)
    axs[1].tick_params(axis='y', labelsize=14)

    for i, method in enumerate(methods):
        rects2 = axs[2].bar(index + bar_width * i, ep_mean[:, i], bar_width, 
                            yerr=ep_std[:, i], label=method, color=colors[i], alpha=0.8,
                            error_kw={'capsize': 1.5, 'capthick': 0.8})
        if i == 0:
            for rect in rects2:
                height = rect.get_height()
                axs[2].annotate(f'{height:.1e}', xy=(rect.get_x(), height),
                                xytext=(2, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, rotation='vertical')
    axs[2].set_xlabel(r'In-degree parameter $\lambda$', fontsize=14)
    axs[2].set_ylabel('EP', fontsize=14)
    axs[2].set_xticks(index + bar_width * (len(methods) - 1) / 2)
    axs[2].set_xticklabels(indegrees, fontsize=14)
    axs[2].tick_params(axis='y', labelsize=14)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.91, 0.9), fontsize=14)

    plt.tight_layout()
    plt.show()
    