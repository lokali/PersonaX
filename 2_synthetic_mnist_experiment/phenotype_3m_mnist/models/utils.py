import os
import sys
import numpy as np
from PIL import Image
import torch
import pickle
import random
from torch import nn
import torch.nn.functional as F
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from .munkres import Munkres
import torch.distributions as dist


def initialize_ground_truth_tracking():
    """
    Initializes tracking variables when ground truth is available.
    """
    return {
        "best_r2": None,
        "best_mcc": None,
        "best_epoch": None,
        "early_stop": False,
        "r2_list": [],
        "mcc_list": [],
        "epochs_recorded": [],
    }
    
def calculate_r2(Z_true, Z_pred):
    krr = KernelRidge(kernel='rbf')
    krr.fit(Z_pred, Z_true)
    Z_pred = krr.predict(Z_pred)
    r2 = r2_score(Z_true, Z_pred)
    return r2

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

def calculate_mcc(Z_true, Z_pred, correlation_fn="Pearson"):
  """Computes score based on both training and testing codes and factors."""
  mus_train, ys_train = Z_true.T, Z_pred.T
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc