# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 21:18:37 2025

@author: 23598
"""

import h5py
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from keras import optimizers

from sklearn.metrics import mean_squared_error
import pandas as pd
import math

from real_unfolding_transformer_DAMP_parral import Unfolded_TransformerEncoder2D_DAMP
#from real_unfolding_transformer_DAMP_parral_nopos import Unfolded_TransformerEncoder2D_DAMP


state_dict = torch.load("data/net/RUST_DAMP_MH_TRIAL20_1ddata_lr_scheduler_bs64_after_220epochs_K2_T20_fixed.pth", map_location='cpu')
#state_dict = torch.load("data/net/RUST_DAMP_MH_TRIAL10_1ddata_lr_scheduler_bs64_after_220epochs_K2_T20_fixed_midSNR.pth", map_location='cpu') #RUST_DAMP_MH_TRIAL10_1ddata_lr_scheduler_bs64_ultimate_K2_small



#K = 2
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)
maxit=20
#model_CNN = FISTA_NET(maxit=maxit) 
model_CNN = Unfolded_TransformerEncoder2D_DAMP(embed_dim=121, signal_dim=512, num_heads=11, layers=maxit, time_dim=20, denoiser_ch=64)
#model_CNN = MLP()
model_CNN.load_state_dict(state_dict)
#summary(model_CNN, (3,16,16))
#model_CNN = U2NET(3,1)#UNET()#
model_CNN.load_state_dict(state_dict)

filename1 = 'data/test/DOA_Dataset_GridAligned_Smooth_test_uniform_vsSNR_100_5deg_P2_K3new.mat' # DOA_Dataset_GridAligned_Smooth_test_uniform_5deg_P2
filename2 = 'D:/files/reafiles/DOA/TSP_JOURNAL_DoA_Estimation/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_min10dBSNR_T200_3D_slideang_offgrid_sep4coma7_deg.h5'
f1 = sio.loadmat(filename1)
GT_angles = np.squeeze(np.array(f1['all_angles']))
Ry_sam_test = np.array(f1['all_Rx']) 
Ry_sam_test = Ry_sam_test[:,:,:,0:2,:,:]
num_snapshots = Ry_sam_test.shape[1]
spatial_spectra = np.zeros((num_snapshots,121))
#Ry_sam_test = np.squeeze(Ry_sam_test)

SNRs, num_packages, K, num_snapshots = GT_angles.shape

A_pred = np.zeros((num_snapshots*num_packages,K))
A = np.zeros((num_snapshots*num_packages,K))

GT_angles_tmp = GT_angles[0,:]
Ry = torch.tensor(Ry_sam_test,dtype=torch.float32)
angles = torch.tensor(GT_angles,dtype=torch.float32)

RMSE_sam = np.zeros((SNRs))

model_CNN.eval()
with torch.no_grad():
    for s in range(SNRs):
        output = model_CNN(Ry[s,:])
        spectrum = output[maxit-1]
        spectrum = np.squeeze(np.array(spectrum.cpu().detach()))
        for ii in range(num_packages):
            x_ind_K_sam = np.argpartition(spectrum[ii,:], -K, axis=1)[:, -K:]
            A_sam = np.sort(v[x_ind_K_sam])
            A_pred[ii*num_snapshots:(ii+1)*num_snapshots,:] = A_sam
            A[ii*num_snapshots:(ii+1)*num_snapshots,:] = GT_angles[s,ii,:].T
        RMSE_sam[s] = round(np.sqrt(mean_squared_error(np.sort(A_pred), np.sort(A))),4)
    


filename = 'data/test_results/RMSE_CNN_K3_fixed_offgrid_ang_all_SNR_T1_DTransformer_5degnew1.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('CNN_RMSE', data=RMSE_sam)
#hf.create_dataset('CNN_RES', data=resolution_values)
hf.close()
