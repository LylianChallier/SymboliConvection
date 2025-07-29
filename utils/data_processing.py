"""
Data processing utilities.

This module contains functions to:
- read the data
- preprocess the data
- make dataset
"""

import os
import numpy as np
import torch
import h5py
from utils.tools.tools_hdf5 import Read_H5_Spav

def recup_slices(n_run, slicedir, sliceid, idtime, nlaps, champs):
    datafilename = f"/workdir/challier/SLICE_H5_RUN{n_run}" + f'/slice_{sliceid:02d}_' + f'{slicedir}_' + f'{idtime:07d}' + '.h5'
    xxh, yyh = Read_H5_Spav(datafilename, gridreading=True)
    slices = np.zeros([len(xxh), len(yyh), len(nlaps), len(champs)])
    tt = np.zeros(len(nlaps))
    for n in range(len(nlaps)):
        idtime = nlaps[n]
        datafilename = f"/workdir/challier/SLICE_H5_RUN{n_run}" + f'/slice_{sliceid:02d}_' + f'{slicedir}_' + f'{idtime:07d}' + '.h5'
        for i, c in enumerate(champs):
            time, fields = Read_H5_Spav(datafilename, c)
            slices[:, :, n, i] = fields
        tt[n] = time
    return slices, xxh, yyh, tt

def normalize(X):
    if isinstance(X, np.ndarray):
        X_mean = np.mean(X, axis=(0, 1))
        X_std = np.std(X, axis=(0, 1))
    elif isinstance(X, torch.Tensor):
        X_mean = torch.mean(X, axis=[0, 1])
        X_std = torch.std(X, axis=[0, 1])
    else:
        raise ValueError('Type not supported')
    X_normed = (X - X_mean) / X_std
    return X_normed, X_mean, X_std

def unormalize(X_normed, X_mean, X_std):
    X_unormed = X_normed * X_std + X_mean
    return X_unormed

def reshape(X):
    return X.reshape(X.shape[0] * X.shape[1], X.shape[2])

def prepare_data_kan(X, y, train_id, test_id, device='cpu'):
    nb_features = X.shape[2]
    data_kan = {}
    data_kan['train_input'] = torch.tensor(X[train_id, :, :].reshape(len(X[train_id]) * X.shape[1], nb_features),
                                           dtype=torch.float32, device=device)
    data_kan['test_input'] = torch.tensor(X[test_id, :, :].reshape(len(X[test_id]) * X.shape[1], nb_features),
                                          dtype=torch.float32, device=device)
    data_kan['train_label'] = torch.tensor(y[train_id, :, :].reshape(len(y[train_id]) * y.shape[1], y.shape[2]),
                                           dtype=torch.float32, device=device)
    data_kan['test_label'] = torch.tensor(y[test_id, :, :].reshape(len(y[test_id]) * y.shape[1], y.shape[2]),
                                          dtype=torch.float32, device=device)
    return data_kan

