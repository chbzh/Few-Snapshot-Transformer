# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:47:45 2025

@author: 23598
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from position_embedding import SinusoidalPositionalEmbedding
from multihead_attention import MultiheadAttention
from thop import profile
from models import *
import math
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))  # 可学习的位置编码

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
    
    

class Denoiser1(nn.Module):
    def __init__(self, in_ch=1,mid_ch=64):
        super(Denoiser1, self).__init__()
        self.threshold = nn.Parameter(torch.ones(1)*0.1)
        
        # Save the passed values
        self.sparse_encoding = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(mid_ch)
            )
        self.sparse_decoding = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(mid_ch, in_ch, kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(in_ch),
            )
        
    def forward(self, r):
        batchsize = r.size(0)
        time_dim = r.size(1)
        r = r.reshape(batchsize,1,time_dim,-1)
        #print(r.shape)
        r = self.sparse_encoding(r)
        r = torch.sign(r)*F.relu(torch.abs(r)-self.threshold)
        r = self.sparse_decoding(r)
        r = r.reshape(batchsize,time_dim,-1)
        
        return r

class TransformerEncoderLayer_DAMP(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, signal_dim=16, num_heads=4, denoiser_ch=64, time_dim=10, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.signal_dim = signal_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True
        )
        self.denoiser1 = Denoiser1(mid_ch=denoiser_ch)
        self.positional_encoding = LearnablePositionalEncoding(d_model=signal_dim, max_len=time_dim)
        self.attn_mask = attn_mask
        self.crossmodal = True
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        #self.fc1 = ComplexLinear(self.embed_dim, self.embed_dim)   # The "Add & Norm" part in the paper
        self.A = nn.Linear(self.embed_dim, self.signal_dim)
        self.AT = nn.Linear(self.signal_dim, self.embed_dim)   # The "Add & Norm" part in the paper

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        #self.threshold = nn.Parameter(torch.ones(1)*0.1,requires_grad=True)
        self.sparsity = nn.Parameter(torch.ones(1)*2/121,requires_grad=True)
        

    def forward(self, h, x, v):
        """
        Args:
            h: Spatial Spectrum from the last iteration.
            x: Input cov vector (real).
            v: ossanger term: same shape as x.
            x = Ah+n
        """
        ## Attention Part
        # Residual and Layer Norm
        x = self.positional_encoding(x)
        residual = h
        #residual_B = h_B

        # Multihead Attention
        h_aaa = self.attention_block(h, h, h)
        
        h = h_aaa 
        
         
        h = self.layer_norms_A[0](h)
        #h_B = self.layer_norms_B[0](h_B)
        # Dropout and Residual
        h = F.dropout(h, p=self.res_dropout, training=self.training)
        #h_B = F.dropout(h_B, p=self.res_dropout, training=self.training)

        h = residual + h
        #h_B = residual_B + h_B
    
        
        # ##FC Part
        residual = h
        #residual_B = h_B
        
        # FC1
        '''
        h_A, h_B = self.fc1(h_A, h_B)
        h_A = F.relu(h_A)
        h_B = F.relu(h_B)
        h_A = F.dropout(h_A, p=self.relu_dropout, training=self.training)
        h_B = F.dropout(h_B, p=self.relu_dropout, training=self.training)
        '''
        
        # LISTA
        
        v = x - self.A(h) + self.sparsity*v
        
        z = h + self.AT(v)  #mediating term: same shape as h.
        
        h = h - self.denoiser1(z)
        
        
        h = self.layer_norms_A[1](h)

        h = F.dropout(h, p=self.res_dropout, training=self.training)
        
        #h = residual + h

        return h, v

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):

        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x



class Unfolded_TransformerEncoder2D_DAMP(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim=121, signal_dim=512, num_heads=11, layers=20, denoiser_ch=64, time_dim=10, attn_dropout=0, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.dropout = 0.3      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.signal_dim = signal_dim
        #self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.positional_encoding = LearnablePositionalEncoding(d_model=signal_dim, max_len=time_dim)        
        self.attn_mask = attn_mask
        self.num_layers = layers
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer_DAMP(embed_dim=embed_dim,
                                    signal_dim=signal_dim,
                                    num_heads=num_heads,
                                    denoiser_ch=denoiser_ch,
                                    time_dim=time_dim,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_x):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        batchsize = input_x.size(0)
        time_dim = input_x.size(1)
        input_x = input_x.reshape(batchsize,time_dim,-1)
        input_x = self.positional_encoding(input_x)
        h_initialize = torch.ones((batchsize,time_dim,self.embed_dim),dtype=torch.float32).to(device)
        v_initialize = torch.ones((batchsize,time_dim,self.signal_dim),dtype=torch.float32).to(device)
        output_h = []
        #input_x = self.scale_embed_position_dropout(input_x)
        #input_B = self.scale_embed_position_dropout(input_B)
        
        h, v = self.layers[0](h_initialize, input_x, v_initialize)
        output_h.append(F.sigmoid(h))
        # For each transformer encoder layer:
        
        for ii in range(1,self.num_layers):
            h, v = self.layers[ii](h, input_x, v)
            output_h.append(F.sigmoid(h))
        return output_h
'''
    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
'''

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim, eps=1e-20)
    return m

def fusion_loss(input_tensor,label,loss_type,weight=None): #exp weight
    iters = len(input_tensor)
    loss = 0
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
        for ii in range(iters):
            loss = loss+criterion(input_tensor[ii],label)*math.exp(ii+1-iters)
    elif loss_type == 'L1':
        criterion = nn.L1Loss()
        for ii in range(iters):
            loss = loss+criterion(input_tensor[ii],label)*math.exp(ii+1-iters)
    elif loss_type == 'BCE':
        if weight is None:
            criterion = nn.BCELoss()
        else:
            criterion = nn.BCELoss(weight=weight)
        for ii in range(iters):
            loss = loss+criterion(input_tensor[ii],label)*math.exp(ii+1-iters)
    else:
        print('invalid!')
        raise RuntimeError
    return loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unfolded_TransformerEncoder2D_DAMP(embed_dim=121, \
                signal_dim=512, num_heads=11, denoiser_ch=64, layers=20, time_dim=20).to(device)
    '''
    total = 0
    for i in range(1000):
        random_x_real = torch.randn((32,10,2,16,16)).to(device)
        start = datetime().now()
        random_output_real = model(random_x_real)
        end = datetime().now()
        total = total + (end-start).seconds()
    print(total/1000)
    '''
    #print(random_output_real[1].shape)
    random_x_real = torch.randn((32,10,2,16,16)).to(device)
    flops,params = profile(model, (random_x_real,))
