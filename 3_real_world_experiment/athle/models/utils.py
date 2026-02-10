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


import numpy as np
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from causallearn.utils.cit import CIT


def get_kci_matrix(all_data, a, b):
    kci_matrix = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            data = all_data[:, [i,a+j]]
            data = remove_nan_rows(data)
            # print("data: ", data.shape)
            kci_obj = CIT(data, "kci")
            pValue = kci_obj(0, 1)
            kci_matrix[i][j] = pValue
    # print("kci_matrix: ", kci_matrix)
    return kci_matrix

def remove_nan_rows(array):
    return array[~np.isnan(array).any(axis=1)]

def compare_kci_matrix(est_kci_matrix, true_kci_matrix):
    """
    Compare the estimated KCI matrix with the true KCI matrix.
    """
    assert est_kci_matrix.shape == true_kci_matrix.shape
    # print("est_kci_matrix: ", est_kci_matrix)
    # print("true_kci_matrix: ", true_kci_matrix)
    identical_mask = (est_kci_matrix == true_kci_matrix)
    
    # Count the number of True values
    count = np.sum(identical_mask)

    x, y, z, w = 0, 0, 0, 0
    for i in range(est_kci_matrix.shape[0]):
        for j in range(est_kci_matrix.shape[1]):
            if est_kci_matrix[i][j] == 0 and true_kci_matrix[i][j] == 0:
                x += 1
            if est_kci_matrix[i][j] == 1 and true_kci_matrix[i][j] == 0:
                y += 1
            if est_kci_matrix[i][j] == 0 and true_kci_matrix[i][j] == 1:
                z += 1
            if est_kci_matrix[i][j] == 1 and true_kci_matrix[i][j] == 1:
                w += 1
    
    # Calculate the percentage of identical values
    total_elements = est_kci_matrix.size    
    percentage = ((x+w) / total_elements) * 100
    print(f"Est-True: 0-0: {x}, 1-1: {w}, 0-1: {y}, 1-0: {z}, total_elements: {total_elements}, percentage: {percentage}%")
    return x+w, percentage


def correlation(x, y, Z_others, true_kci_matrix, method='Spearman'):
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
    dim = x.shape[0]  # 5

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
        # print("corr: ")
        # print(corr.shape)
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]
    elif method=='Kendall':
        corr, pvalue = stats.kendalltau(y.T, x.T)
        corr = corr[0:dim, dim:]
    elif method=="mutual_info":
        corr = np.zeros((dim, dim))
        # print("init x. shape: ", x.T.shape)
        # print("init y. shape: ", y.T.shape)
        for i in range(dim):
            for j in range(dim):
                if i <= j:
                    mi = mutual_info_regression(y.T[:, [i]], x.T[:, j], random_state=42)
                    corr[i][j] = mi.mean()  # Average MI across samples
                    corr[j][i] = mi.mean()
    elif method == "KCI":
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # print("x_sort: ", x_sort.shape)
    # print("y: ", y.shape)   
    # print("Z_others: ", Z_others.shape)


    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
        # print("corr_sort: ")
        # print(y.shape, x_sort.shape)
        # print(corr_sort.shape)
        # print(corr_sort)

    elif method == "KCI":
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]


        all_data = np.hstack([x_sort.T, Z_others])
        # print("all_data: ", all_data.shape)
        kci_matrix = get_kci_matrix(all_data, x_sort.shape[0], Z_others.shape[1])
        # print(f"estimated kci matrix: {kci_matrix}")
        threshold = 0.05 
        kci_matrix_bin = np.where(kci_matrix < threshold, 0, 1)
        # print(f"bin kci matrix: {kci_matrix_bin}")
        # print(f"true kci matrix: {true_kci_matrix}")
        count, percentage = compare_kci_matrix(kci_matrix_bin, true_kci_matrix)
        print(f"count: {count}, out of {true_kci_matrix.size}: overlap percentage: {percentage}%") 

        

    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]
    elif method=="mutual_info":
        corr_sort = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i <= j:
                    mi = mutual_info_regression(y.T[:, [i]], x_sort.T[:, j], random_state=42)
                    corr_sort[i][j] = mi.mean()  # Average MI across samples
                    corr_sort[j][i] = mi.mean()
        corr_sort = corr_sort[0:dim, 0:dim]
    return corr_sort, sort_idx, x_sort

def calculate_mcc(Z_true, Z_pred, Z_others, true_kci_matrix, correlation_fn="KCI"):
  """Computes score based on both training and testing codes and factors."""

#   print("Z_true: ", Z_true.shape)
#   print("Z_pred: ", Z_pred.shape)
#   print("Z_others: ", Z_others.shape)

  mus_train, ys_train = Z_true.T, Z_pred.T  # 5*200
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, Z_others, true_kci_matrix, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc