import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

import torch
from torch import nn

class ApplSA(nn.Module):

    def __init__(self, config, LSTM=False, splitLoss=False):

        super(ApplSA, self).__init__()
        self.self_attn = nn.MultiheadAttention(2*config.hidden, 2, batch_first=True)
        d_model = 2*config.hidden
        dim_feedforward = 1024
        dropout = config.dropout
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
    
    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        return x

class ApplFF(nn.Module):

    def __init__(self, config, LSTM=False, splitLoss=False):

        super(ApplFF, self).__init__()
        d_model = 2*config.hidden
        dim_feedforward = 1024
        dropout = config.dropout
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = self.norm2(x + self._ff_block(x))
        return x


class ApplBlock(nn.Module):

    def __init__(self, config, Last=False, LSTM=False, splitLoss=False):

        super(ApplBlock, self).__init__()
        self.multihead_attn_d = ApplSA(config)
        self.multihead_attn_f = ApplSA(config)
        self.multihead_attn_m = ApplSA(config)
        self.multihead_attn_w = ApplSA(config)

        self.multihead_attn_r_g = nn.MultiheadAttention(2*config.hidden, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(2*config.hidden)

        self.dish = ApplFF(config)
        self.frid = ApplFF(config)
        self.micro = ApplFF(config)
        self.wash = ApplFF(config)
        self.Last = Last
        if Last:
            self.dish_c = ApplFF(config)
            self.frid_c = ApplFF(config)
            self.micro_c = ApplFF(config)
            self.wash_c = ApplFF(config)

    def forward(self, d_r_a, f_r_a, m_r_a, w_r_a):
        attn_output_d = self.multihead_attn_d(d_r_a)
        attn_output_f = self.multihead_attn_f(f_r_a)
        attn_output_m = self.multihead_attn_m(m_r_a)
        attn_output_w = self.multihead_attn_w(w_r_a)

        GlobleAtten_r = torch.cat((attn_output_d.unsqueeze(3), attn_output_f.unsqueeze(3), attn_output_m.unsqueeze(3), attn_output_w.unsqueeze(3)),3)
        GlobleAtten_r = GlobleAtten_r.permute(0,1,3,2)
        # TODO: change 4
        GlobleAtten_r = GlobleAtten_r.reshape(-1,4,GlobleAtten_r.shape[-1])

        # # Globle attention
        attn_output_r_g, attn_output_weights_r_g = self.multihead_attn_r_g(GlobleAtten_r,GlobleAtten_r,GlobleAtten_r)
        # TODO: change 4
        attn_output_r_g = attn_output_r_g.reshape(d_r_a.shape[0],d_r_a.shape[1],4,GlobleAtten_r.shape[-1])

        d_r_a = attn_output_r_g[:,:,0,:]
        f_r_a = attn_output_r_g[:,:,1,:]
        m_r_a = attn_output_r_g[:,:,2,:]
        w_r_a = attn_output_r_g[:,:,3,:]

        d_r_a = self.norm1(d_r_a + attn_output_d)
        f_r_a = self.norm1(f_r_a + attn_output_f)
        m_r_a = self.norm1(m_r_a + attn_output_m)
        w_r_a = self.norm1(w_r_a + attn_output_w)

        d_r = self.dish(d_r_a)
        f_r = self.frid(f_r_a)
        m_r = self.micro(m_r_a)
        w_r = self.wash(w_r_a)

        if self.Last:
            d_c = self.dish_c(d_r_a)
            f_c = self.frid_c(f_r_a)
            m_c = self.micro_c(m_r_a)
            w_c = self.wash_c(w_r_a)
            return d_r, f_r, m_r, w_r, d_c, f_c, m_c, w_c

        return d_r, f_r, m_r, w_r


class MATconv(nn.Module):

    def __init__(self, config, LSTM=False, splitLoss=False):

        super(MATconv, self).__init__()
        self.input_size = config.input_size
        self.sharedLayer = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, padding='same'),
            nn.ReLU(True),
            nn.Conv1d(30, 30, kernel_size=8, padding='same'),
            nn.ReLU(True),
            nn.Conv1d(30, 40, kernel_size=6, padding='same'),
            nn.ReLU(True),
            nn.Conv1d(40, 50, kernel_size=5, padding='same'),
            nn.ReLU(True),
            nn.Conv1d(50, 50, kernel_size=5, padding='same'),
            nn.ReLU(True),
            nn.Conv1d(50, int(config.hidden*2), kernel_size=5, padding='same'),
            nn.ReLU(True)
        )

        self.block1 = ApplBlock(config)
        self.block2 = ApplBlock(config)
        self.block3 = ApplBlock(config,Last=True)


        self.fc_dr = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_dc = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_fr = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_fc = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_mr = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_mc = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))

        self.fc_wr = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))
        self.fc_wc = nn.Sequential(nn.Linear(2*config.hidden, config.hidden),
                                   nn.ReLU(),
                                   nn.Linear(config.hidden, 1))

    def forward(self, input_data):

        # feature weighting
        # attn_weights = torch.softmax(self.attn_layer(input_data), 2)
        # weighted_input = torch.mul(attn_weights, input_data)
        # skip_x =
        # self.sharedLayer.flatten_parameters()
        input_data = input_data.permute(0,2,1)
        input_encoded = self.sharedLayer(input_data)  # input(1, batch_size, input_size)
        input_encoded = input_encoded.permute(0,2,1)
        # print(input_encoded.shape)
        # print(input_encoded.shape)
        # print(input_data.shape)
        # input_encoded = self.norm(input_encoded + self.skipExpand(input_data))

        # Attention
        d_r, f_r, m_r, w_r = self.block1(input_encoded, input_encoded, input_encoded, input_encoded)
        d_r, f_r, m_r, w_r = self.block2(d_r, f_r, m_r, w_r)
        d_rr, f_rr, m_rr, w_rr, d_cc, f_cc, m_cc, w_cc = self.block3(d_r, f_r, m_r, w_r)

        dc = torch.sigmoid(self.fc_dc(d_cc))
        fc = torch.sigmoid(self.fc_fc(f_cc))
        mc = torch.sigmoid(self.fc_mc(m_cc))
        wc = torch.sigmoid(self.fc_wc(w_cc))

        # dr = torch.relu(self.fc_dr(d_rr)) * dc
        # fr = torch.relu(self.fc_fr(f_rr)) * fc
        # mr = torch.relu(self.fc_mr(m_rr)) * mc
        # wr = torch.relu(self.fc_wr(w_rr)) * wc

        dr = self.fc_dr(d_rr) * dc
        fr = self.fc_fr(f_rr) * fc
        mr = self.fc_mr(m_rr) * mc
        wr = self.fc_wr(w_rr) * wc

        y_pred_r = torch.cat((dr,fr,mr, wr),2)
        y_pred_c = torch.cat((dc,fc,mc, wc),2)
        # print(y_pred_r.shape)
        return y_pred_r, y_pred_c
