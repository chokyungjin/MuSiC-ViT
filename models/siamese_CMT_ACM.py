"""
Covers useful modules referred in the paper
All dimensions in comments are induced from 224 x 224 x 3 inputs
and CMT-S
Created by Kunhong Yu
Date: 2021/07/14
"""

import torch as t
from torch.nn import functional as F
import math
import torch.nn as nn

class ACMBlock(nn.Module):
    def __init__(self, in_channels):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=23),
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=23),
        )

        self.global_pooling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels//2, (1,1)),
            nn.ReLU(),
            nn.Conv2d(self.out_channels//2, self.out_channels, (1,1)),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.normalize = nn.Softmax(dim=3)

    def _get_normalized_features(self, x):
        ''' Get mean vector by channel axis '''
        
        c_mean = self.avgpool(x)
        return c_mean

    def _get_orth_loss(self, K, Q):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = cos(K, Q)
        # orth_loss = orth_loss ** 2  # cos 0
        # orth_loss = abs(orth_loss)  # cos 0
        orth_loss = t.mean(orth_loss, dim=0)
        return orth_loss
    
    def _get_orth_loss_ACM(self, K, Q, c):
        orth_loss = t.mean(K*Q/c, dim=1, keepdim=True)
        return orth_loss
    
    def forward(self, x1, x2):
        mean_x1 = self._get_normalized_features(x1)
        mean_x2 = self._get_normalized_features(x2)
        x1_mu = x1-mean_x1
        x2_mu = x2-mean_x2
        
        K = self.k_conv(x1_mu)
        Q = self.q_conv(x2_mu)

        b, c, h, w = K.shape

        K = K.view(b, c, 1, h*w)
        K = self.normalize(K)
        K = K.view(b, c, h, w)

        Q = Q.view(b, c, 1, h*w)
        Q = self.normalize(Q)
        Q = Q.view(b, c, h, w)

        K = t.einsum('nchw,nchw->nc',[K, x1_mu])
        Q = t.einsum('nchw,nchw->nc',[Q, x2_mu])
        K = K.view(K.shape[0], K.shape[1], 1, 1)
        Q = Q.view(Q.shape[0], Q.shape[1], 1, 1)

        channel_weights1 = self.global_pooling(mean_x1)
        channel_weights2 = self.global_pooling(mean_x2)
        
        
        # original ACM
        out1 = x1 + K - Q
        out2 = x2 + K - Q
        
        # MuSiC-ViT
        # out1 = x1 + K + Q
        # out2 = x2 + K + Q
        
        out1 = channel_weights1 * out1
        out2 = channel_weights2 * out2
        
        # original ACM
        # orth_loss = self._get_orth_loss_ACM(K,Q,c)
        
        # MuSiC-ViT
        orth_loss = self._get_orth_loss(K,Q)

        return out1, out2, orth_loss

#########################
#  0. Patch Aggregation #
#########################
class PatchAggregation(t.nn.Module):
    """Define Bridge/PatchAggregation module connecting each other module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 16, out_channels = 46 , 
    patch_ker=2, patch_str=2):
        """
        Args :
            --in_channels: default is 16
            --out_channels: default is 46
        """
        super(PatchAggregation, self).__init__()

        self.pa = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                        kernel_size = patch_ker, stride = patch_str),
        )

    def forward(self, x):

        x = self.pa(x)
        b, c, h, w = x.size()
        x = F.layer_norm(x, (c, h, w))

        return x


#########################
#       1. Stem         #
#########################
class Stem(t.nn.Module):
    """Define Stem module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 3, out_channels = 16):
        """
        Args :
            --in_channels: default is 3
            --out_channels: default is 16
        """
        super(Stem, self).__init__()

        self.stem = t.nn.Sequential(
            # 1.1 One Conv layer
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 2, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU(), # 112 x 112 x 16

            # 1.2 Two subsequent Conv layers
            t.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 1, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU(), # 112 x 112 x 16
            t.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 1, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU() # 112 x 112 x 16
        )

    def forward(self, x):

        x = self.stem(x)

        return x


#########################
#     3. CMT block      #
#########################
#*************
#  3.1 LPU   #
#*************
class LPU(t.nn.Module):
    """Define Local Perception Unit
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 46):
        """
        Args :
            --in_channels: default is 46
        """
        super(LPU, self).__init__()

        out_channels = in_channels
        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, groups = in_channels,
                        kernel_size = 3, stride = 1, padding = 1) # 112 x 112 x 46
        )

    def forward(self, x):

        x = x + self.dwconv(x)

        return x


#*************
#  3.2 LMHSA #
#*************
class LMHSA(t.nn.Module):
    """Define Lightweight MHSA module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, input_size, kernel_size, d_k, d_v, num_heads, in_channels = 46):
        """
        Args :
            --input_size
            --kernel_size: for DWConv
            --d_k: dimension for key and query
            --d_v: dimension for value
            --num_heads: attention heads
            --in_channels: default is 46
        """
        super(LMHSA, self).__init__()

        stride = kernel_size
        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = in_channels, groups = in_channels,
                        kernel_size = kernel_size, stride = stride)
        ) # (112 / kernel_size) x (112 x kernel_size) x 46

        self.query = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.key = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.value = t.nn.Sequential(
            t.nn.Linear(in_channels, d_v * num_heads)
        )

        self.B = t.nn.Parameter(t.rand(1, num_heads, input_size ** 2, (input_size // kernel_size) ** 2), requires_grad = True)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(self.d_k)
        self.softmax = t.nn.Softmax(dim = -1)
        self.LN = t.nn.LayerNorm(in_channels)

    def forward(self, x):
        """x has shape [m, c, h, w]"""
        b, c, h, w = x.size()
        x_ = x

        # i. reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1) # [m, h * w, c]
        x_reshape = self.LN(x_reshape)

        # ii. Get key, query and value
        q = self.query(x_reshape) # [m, h * w, d_k * num_heads]
        q = q.view(b, h * w, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [m, num_heads, h * w, d_k]

        k = self.dwconv(x) # [m, c, h', w']
        c_, h_, w_ = k.size(1), k.size(-2), k.size(-1)
        k = k.view(b, c_, h_ * w_).permute(0, 2, 1) # [m, h' * w', c]
        k = self.key(k) # [m, h' * w', d_k * num_heads]
        k = k.view(b, h_ * w_, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [m, num_heads, h' * w', d_k]

        v = self.dwconv(x)  # [m, c, h', w']
        v = v.view(b, c_, h_ * w_).permute(0, 2, 1)  # [m, h' * w', c]
        v = self.value(v)  # [m, h' * w', d_v * num_heads]
        v = v.view(b, h_ * w_, self.num_heads, self.d_v).permute(0, 2, 1, 3)  # [m, num_heads, h' * w', d_v]

        # iii. LMHSA
        logit = t.matmul(q, k.transpose(-2, -1)) / self.scale # [m, num_heads, h * w, h' * w']
        logit = logit + self.B
        attention = self.softmax(logit)
        attn_out = t.matmul(attention, v) # [m, num_heads, h * w, d_v]
        attn_out = attn_out.permute(0, 2, 1, 3) # [m, h * w, num_heads, d_v]
        attn_out = attn_out.reshape(b, h, w, self.num_heads * self.d_v).permute(0, -1, 1, 2) # [m, num_heads * d_v, h, w]

        return attn_out + x_


#*************
# 3.3 IRFFN  #
#*************
class IRFFN(t.nn.Module):
    """Define IRFNN module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 46, R = 3.6):
        """
        Args :
            --in_channels: default is 46
            --R: expansion ratio, default is 3.6
        """
        super(IRFFN, self).__init__()

        exp_channels = int(in_channels * R)
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = exp_channels, kernel_size = 1),
            t.nn.BatchNorm2d(exp_channels),
            t.nn.GELU()
        ) # 112 x 112 x exp_channels

        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = exp_channels, out_channels = exp_channels, groups = exp_channels,
                        kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(exp_channels),
            t.nn.GELU()
        ) # 112 x 112 x exp_channels

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = exp_channels, out_channels = in_channels, kernel_size = 1),
            t.nn.BatchNorm2d(in_channels)
        ) # 112 x 112 x 46

    def forward(self, x):

        _, c, h, w = x.size()
        x_ = F.layer_norm(x, (c, h, w))

        x_ = self.conv2(self.dwconv(self.conv1(x_)))

        return x + x_


#*************
#3.4 CMT block#
#*************
class CMTBlock(t.nn.Module):
    """Define CMT block"""

    def __init__(self, input_size, kernel_size, d_k, d_v, num_heads, R = 3.6, in_channels = 46):
        """
        Args :
            --input_size
            --kernel_size: for DWConv
            --d_k: dimension for key and query
            --d_v: dimension for value
            --num_heads: attention heads
            --R: expansion ratio, default is 3.6
            --in_channels: default is 46
        """
        super(CMTBlock, self).__init__()

        # 1. LPU
        self.lpu = LPU(in_channels = in_channels)

        # 2. LMHSA
        self.lmhsa = LMHSA(input_size = input_size,
                           kernel_size = kernel_size, d_k = d_k, d_v = d_v, 
                           num_heads = num_heads,
                           in_channels = in_channels)

        # 3. IRFFN
        self.irffn = IRFFN(in_channels = in_channels, R = R)

    def forward(self, x):

        x = self.lpu(x)
        x = self.lmhsa(x)
        x = self.irffn(x)

        return x


import torch as t
from torch.nn import functional as F

#########################
#   CMT Configuration   #
#########################
class siamese_CMT_ACM(t.nn.Module):
    """Define CMT model"""

    def __init__(self,
                 in_channels = 3,
                 stem_channels = 16,
                 cmt_channelses = [46, 92, 184, 368],
                 pa_channelses = [46, 92, 184, 368],
                 R = 3.6,
                 repeats = [2, 2, 10, 2],
                 input_size = 224,
                 sizes = [64, 32, 16, 8],
                 patch_ker=2,
                 patch_str=2,
                 num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --stem_channels: stem channels, default is 16
            --cmt_channelses: list, default is [46, 92, 184, 368]
            --pa_channels: patch aggregation channels, list, default is [46, 92, 184, 368]
            --R: expand ratio, default is 3.6
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 10, 2]
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(siamese_CMT_ACM, self).__init__()

        if input_size == 224:
            sizes = [56, 28, 14, 7]
        elif input_size == 160:
            sizes = [40, 20, 10, 5]
        elif input_size == 192:
            sizes = [48, 24, 12, 6]
        elif input_size == 256:
            sizes = [64, 32, 16, 8]
        elif input_size == 288:
            sizes = [72, 36, 18, 9]
        elif input_size == 512:
            sizes = sizes
            # sizes = [127, 62, 30, 14]
        else:
            raise Exception('No other input sizes!')

        # 1. Stem
        self.stem = Stem(in_channels = in_channels, out_channels = stem_channels)

        # 2. Patch Aggregation 1
        self.pa1 = PatchAggregation(in_channels = stem_channels, out_channels = pa_channelses[0], patch_ker=patch_ker, patch_str=patch_str)
        self.pa2 = PatchAggregation(in_channels = cmt_channelses[0], out_channels = pa_channelses[1], patch_ker=patch_ker, patch_str=patch_str)
        self.pa3 = PatchAggregation(in_channels = cmt_channelses[1], out_channels = pa_channelses[2], patch_ker=patch_ker, patch_str=patch_str)
        self.pa4 = PatchAggregation(in_channels = cmt_channelses[2], out_channels = pa_channelses[3], patch_ker=patch_ker, patch_str=patch_str)

        # 3. CMT block
        cmt1 = []
        for _ in range(repeats[0]):
            cmt_layer = CMTBlock(input_size = sizes[0],
                                 kernel_size = 8,
                                 d_k = cmt_channelses[0],
                                 d_v = cmt_channelses[0],
                                 num_heads = 1,
                                 R = R, in_channels = pa_channelses[0])
            cmt1.append(cmt_layer)
        self.cmt1 = t.nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_size = sizes[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = t.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_size = sizes[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = t.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_size = sizes[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = t.nn.Sequential(*cmt4)

        # 4. Global Avg Pool
        self.avg = t.nn.AdaptiveAvgPool2d(1)

        # 5. FC
        self.fc = t.nn.Sequential(
            t.nn.Linear(cmt_channelses[-1], 1280),
            t.nn.ReLU(inplace = True) # we use ReLU here as default
        )

        # 6. Classifier
        self.classifier = t.nn.Sequential(
            t.nn.Linear(1280*2, num_classes)
        )
        
        self.disease_linear = nn.Linear(1280, 2)
        
        self.acm1 = ACMBlock(in_channels=23*2)
        self.acm2 = ACMBlock(in_channels=23*4)
        self.acm3 = ACMBlock(in_channels=23*8)
        self.acm4 = ACMBlock(in_channels=23*16)

    def forward(self, x , x2):

        # 1. Stem
        x_stem = self.stem(x)

        # 1. Stem
        x2_stem = self.stem(x2)
        
        # 2. PA1 + CMTb1
        x_pa1 = self.pa1(x_stem)
        x_cmtb1 = self.cmt1(x_pa1)

        # 2. PA1 + CMTb1
        x2_pa1 = self.pa1(x2_stem)
        x2_cmtb1 = self.cmt1(x2_pa1)
        x_cmtb1, x2_cmtb1, orth_loss1 = self.acm1(x_cmtb1, x2_cmtb1)
        
        
        # 3. PA2 + CMTb2
        x_pa2 = self.pa2(x_cmtb1)
        x_cmtb2 = self.cmt2(x_pa2)

        # 3. PA2 + CMTb2
        x2_pa2 = self.pa2(x2_cmtb1)
        x2_cmtb2 = self.cmt2(x2_pa2)
        x_cmtb2, x2_cmtb2, orth_loss2 = self.acm2(x_cmtb2, x2_cmtb2)
        
        
        # 4. PA3 + CMTb3
        x_pa3 = self.pa3(x_cmtb2)
        x_cmtb3 = self.cmt3(x_pa3)

        # 4. PA3 + CMTb3
        x2_pa3 = self.pa3(x2_cmtb2)
        x2_cmtb3 = self.cmt3(x2_pa3)
        
        x_cmtb3, x2_cmtb3, orth_loss3 = self.acm3(x_cmtb3, x2_cmtb3)
        

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)

        # 5. PA4 + CMTb4
        x2_pa4 = self.pa4(x2_cmtb3)
        x2_cmtb4 = self.cmt4(x2_pa4)
        
        x_cmtb4, x2_cmtb4, orth_loss4 = self.acm4(x_cmtb4, x2_cmtb4)

        # 6. Avg
        x_avg = self.avg(x_cmtb4)
        x_avg = x_avg.squeeze()

        # 6. Avg
        x2_avg = self.avg(x2_cmtb4)
        x2_avg = x2_avg.squeeze()

        # 7. Linear + Classifier
        x_fc = self.fc(x_avg)

        # 7. Linear + Classifier
        x2_fc = self.fc(x2_avg)
        
        orth_score = (orth_loss1 + orth_loss2 + orth_loss3 + orth_loss4) / 4
        
        x1 = self.disease_linear(x_fc)
        x2 = self.disease_linear(x2_fc)
        
        if x_fc.shape[0] == 1280:
            x_fc = x_fc.unsqueeze(0)
            x2_fc = x2_fc.unsqueeze(0)
            
        cat = t.cat([x_fc, x2_fc],1)
        out = self.classifier(cat)

        return x1, x2, out, orth_score