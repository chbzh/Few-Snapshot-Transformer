# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:04:54 2025

@author: 23598
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:41:10 2025

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

filename1 = 'data/test/DOA_Dataset_GridAligned_Smooth_test_uniform_running10deg_20_5_m5dB_1A_K4.mat' #DOA_Dataset_GridAligned_Smooth_test_parallel_1A
#filename1 = 'data/test/DOA_Dataset_GridAligned_Smooth_test_uniform_running5deg_20_5_m5dB_1A_K4.mat' #DOA_Dataset_GridAligned_Smooth_test_parallel_1A
f1 = sio.loadmat(filename1)
GT_angles = np.squeeze(np.array(f1['all_angles']))
Ry_sam_test = np.array(f1['all_Rx']) 
Ry_sam_test = Ry_sam_test[:,:,0:2,:,:]
num_snapshots = Ry_sam_test.shape[1]
spatial_spectra = np.zeros((num_snapshots,121))
#Ry_sam_test = np.squeeze(Ry_sam_test)

num_packages, K, num_snapshots = GT_angles.shape
A_pred = np.zeros((num_snapshots*num_packages,K))
A = np.zeros((num_snapshots*num_packages,K))

GT_angles_tmp = GT_angles[3,:]
Ry = torch.tensor(Ry_sam_test,dtype=torch.float32)
angles = torch.tensor(GT_angles,dtype=torch.float32)

for t in range(num_snapshots):  # 遍历快拍
    # 获取当前快拍的两个角度值
    theta1, theta2, theta3, theta4 = GT_angles_tmp[:, t]
    
    # 将角度值映射到网格索引
    index1 = int(round(theta1)) + 60
    index2 = int(round(theta2)) + 60
    index3 = int(round(theta3)) + 60
    index4 = int(round(theta4)) + 60
    
    # 确保索引在有效范围内
    index1 = max(0, min(120, index1))
    index2 = max(0, min(120, index2))
    index3 = max(0, min(120, index3))
    index4 = max(0, min(120, index4))
    
    # 设置独热编码
    spatial_spectra[t, index1] = 1
    spatial_spectra[t, index2] = 1
    spatial_spectra[t, index3] = 1
    spatial_spectra[t, index4] = 1

model_CNN.eval()
with torch.no_grad():
    output = model_CNN(Ry)
    spectrum = output[maxit-1]
    spectrum = np.squeeze(np.array(spectrum.cpu().detach()))
    

for ii in range(num_packages):
    x_ind_K_sam = np.argpartition(spectrum[ii,:], -K, axis=1)[:, -K:]
    A_sam = np.sort(v[x_ind_K_sam])
    A_pred[ii*num_snapshots:(ii+1)*num_snapshots,:] = A_sam
    A[ii*num_snapshots:(ii+1)*num_snapshots,:] = GT_angles[ii,:].T
    
    

RMSE_sam = round(np.sqrt(mean_squared_error(np.sort(A_pred), np.sort(A))),4)

plt.figure()
plt.imshow(spatial_spectra)
plt.figure()
plt.imshow(spectrum[4,:])


'''
filename4 = 'data/test_results/Slide_angsep5_K4_min10dB_T1_Transformer.h5'
hf = h5py.File(filename4, 'w')
hf.create_dataset('GT_angles', data=A)
hf.create_dataset('CNN_pred_angles', data=A_pred)
hf.close()
'''
