# -*- coding: utf-8 -*-
import numpy as np
import os
import logging
import contextlib
import matlab
import matlab.engine
from rpy2.robjects import numpy2ri, r
from rpy2.robjects.vectors import StrVector
import rpy2.rinterface_lib.callbacks
from methods.GENIE3.GENIE3 import GENIE3
from methods.nonlinearODE.xgbgrn_2 import xgbgrn2

# Suppress R Warnings
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)


class SuppressRMessages:
    """Suppress R Messages."""
    def __enter__(self):
        self._original_writeconsole = rpy2.rinterface_lib.callbacks.consolewrite_print
        self._original_writeconsole_warn = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        rpy2.rinterface_lib.callbacks.consolewrite_print = self._suppress
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = self._suppress

    def __exit__(self, exc_type, exc_val, exc_tb):
        rpy2.rinterface_lib.callbacks.consolewrite_print = self._original_writeconsole
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = self._original_writeconsole_warn

    @staticmethod
    def _suppress(x):
        pass


@contextlib.contextmanager
def temporary_change_dir(path):
    """
    Context manager for changing the current working directory temporarily.
    
    Parameters:
    path : str
        Path to change the working directory to.
    
    Yields:
    None
    """
    original_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_path)


def method_pcapmi(exp1, exp2):
    """
    Perform PCA-PMI for GRN using MATLAB.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    
    Returns:
    np.ndarray
        Score matrix representing gene interactions.
    """
    data = np.vstack((exp1, exp2))
    with temporary_change_dir(os.getcwd() + "/methods/PCA-PMI"):
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        # Convert numpy arrays to MATLAB arrays
        data_m = matlab.double(data)
        # Call the MATLAB function
        res = eng.pca_pmi(data_m, 0.05, 1, nargout=3)
        # Stop MATLAB engine
        eng.quit()
    # Convert the output back to numpy array
    Gval = np.array(res[1])
    return Gval[:exp1.shape[0], exp1.shape[0]:]

    
def method_tigress(exp1, exp2):
    """
    Perform TIGRESS for GRN using R.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    
    Returns:
    np.ndarray
        Score matrix representing gene interactions.
    """
    data = np.vstack((exp1, exp2)).T
    n_gene = exp1.shape[0]
    
    with SuppressRMessages():
        numpy2ri.activate()
        r_data = numpy2ri.py2rpy(data)  # Convert NumPy array to R matrix
        genenames = ['G' + str(i) for i in range(1, 2 * n_gene + 1)]
        tflist = ['G' + str(i) for i in range(1, n_gene + 1)]
        targetlist = ['G' + str(i) for i in range(n_gene + 1, 2 * n_gene + 1)]
        r_genenames = StrVector(genenames)
        r_tflist = StrVector(tflist)
        r_targetlist = StrVector(targetlist)
        
        r.assign("r_data", r_data)
        r.assign("r_genenames", r_genenames)
        r.assign("r_tflist", r_tflist)
        r.assign("r_targetlist", r_targetlist)
        
        cmd = """
        library(tigress)
        colnames(r_data) <- r_genenames
        result <- tigress(expdata=r_data, tflist=r_tflist, targetlist=r_targetlist, 
                        allsteps=FALSE, nstepsLARS=2, nsplit=20)
        """
        r(cmd)
        tigress = r['result']
        numpy2ri.deactivate()

    return tigress
            

def method_mmhc(exp1, exp2):
    """
    Perform MMHC for GRN using R.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    
    Returns:
    np.ndarray
        Score matrix representing gene interactions.
    """
    data = np.hstack((exp1.T, exp2.T))
    n_gene = exp1.shape[0]
    
    with SuppressRMessages():
        numpy2ri.activate()
        r_data = numpy2ri.py2rpy(data)  # Convert NumPy array to R matrix
        genenames = ['G' + str(i) for i in range(1, 2 * n_gene + 1)]
        r_genenames = StrVector(genenames)
        
        r.assign("r_data", r_data)
        r.assign("r_genenames", r_genenames)
        
        cmd = """
        library(bnlearn)
        library(foreach)
        library(doParallel)
        
        data <- as.data.frame(r_data)
        names(data) <- r_genenames
        p <- ncol(data)
        n_bootstraps <- 100
        
        edge_matrix <- matrix(0, nrow = p, ncol = p, dimnames = list(colnames(data), colnames(data)))
        
        cl <- makeCluster(detectCores() - 1)
        registerDoParallel(cl)
        
        results <- foreach(i = 1:n_bootstraps, .combine='cbind', .packages='bnlearn') %dopar% {
        set.seed(i)
        bootstrap_sample <- data[sample(1:nrow(data), replace = TRUE), ]
        bn <- mmhc(bootstrap_sample)
        
        local_edge_matrix <- matrix(0, nrow = p, ncol = p, dimnames = list(colnames(data), colnames(data)))
        
        if (!is.null(bn$arcs) && nrow(bn$arcs) > 0) {
            for (arc in 1:nrow(bn$arcs)) {
            from <- bn$arcs[arc, "from"]
            to <- bn$arcs[arc, "to"]
            local_edge_matrix[from, to] <- local_edge_matrix[from, to] + 1
            }
        }
        
        return(local_edge_matrix)
        }
        stopCluster(cl)
            
        edge_matrix <- array(dim = c(p, p, n_bootstraps))
        for (i in 1:n_bootstraps) {
        edge_matrix[,,i] <- results[, (1:p) + p*(i-1)]
        }
        edge_matrix <- apply(edge_matrix, c(1, 2), sum)
        
        edge_importance <- edge_matrix / n_bootstraps
        """
        
        r(cmd)
        mmhc = r['edge_importance']
        numpy2ri.deactivate()

    return mmhc[:exp1.shape[0], exp1.shape[0]:]


def method_genie3(exp1, exp2):
    """
    Perform GENIE3 for GRN.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    
    Returns:
    np.ndarray
        Score matrix representing gene interactions.
    """
    data = np.vstack((exp1, exp2)).T
    n_gene = exp1.shape[0]
    gene_names = [str(i) for i in range(2 * n_gene)]
    regulators = gene_names[:n_gene]
    genie3 = GENIE3(data, gene_names=gene_names, regulators=regulators, ntrees=100)
    return genie3[:exp1.shape[0], exp1.shape[0]:]


def method_nonlinearODE(exp1, exp2):
    """
    Perform nonlinearODE for GRN.
    
    Parameters:
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes * samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes * samples).
    
    Returns:
    np.ndarray
        Score matrix representing gene interactions.
    """
    data = np.vstack((exp1, exp2)).T
    n_gene = exp1.shape[0]
    gene_names = [str(i) for i in range(2 * n_gene)]
    regulators = gene_names[:n_gene]
    nonlinearODE = xgbgrn2(data, gene_names, regulators)
    return nonlinearODE[:exp1.shape[0], exp1.shape[0]:]
 
