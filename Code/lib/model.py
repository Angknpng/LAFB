import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Code.lib.res2net_v1b_base import Res2Net_model
import os


def convblock(in_,out_,ks,st,pad,dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad,dilation),
        nn.BatchNorm2d(out_))
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#*********************Challenge branch***********************************
class CB(nn.Module):
    def __init__(self):
        super(CB,self).__init__()
        self.level1_r = nn.Conv2d(256, 256, 3, 1, 1)
        self.level1_t = nn.Conv2d(256, 256, 3, 1, 1)
        self.level2_r = nn.Conv2d(512, 256, 3, 1, 1)
        self.level2_t = nn.Conv2d(512, 256, 3, 1, 1)
        self.level3_r = nn.Conv2d(1024, 256, 3, 1, 1)
        self.level3_t = nn.Conv2d(1024, 256, 3, 1, 1)
        self.level4_r = nn.Conv2d(2048, 256, 3, 1, 1)
        self.level4_t = nn.Conv2d(2048, 256, 3, 1, 1)

    def forward(self, rgb, t, level=1):
        if level ==1:
            return self.level1_r(rgb), self.level1_t(t)
        elif level==2:
            return self.level2_r(rgb), self.level2_t(t)
        elif level==3:
            return self.level3_r(rgb), self.level3_t(t)
        else:
            return self.level4_r(rgb), self.level4_t(t)

class NV(nn.Module):
    def __init__(self):
        super(NV,self).__init__()
        self.level1_r = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.level1_t = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.level2_r = nn.Conv2d(512, 256, 3, 1, 2, 2)
        self.level2_t = nn.Conv2d(512, 256, 3, 1, 2, 2)
        self.level3_r = nn.Conv2d(1024, 256, 3, 1, 2, 2)
        self.level3_t = nn.Conv2d(1024, 256, 3, 1, 2, 2)
        self.level4_r = nn.Conv2d(2048, 256, 3, 1, 2, 2)
        self.level4_t = nn.Conv2d(2048, 256, 3, 1, 2, 2)
    def forward(self, rgb, t, level=1):
        if level == 1:
            return self.level1_r(rgb), self.level1_t(t)
        elif level == 2:
            return self.level2_r(rgb), self.level2_t(t)
        elif level == 3:
            return self.level3_r(rgb), self.level3_t(t)
        else:
            return self.level4_r(rgb), self.level4_t(t)

class LI(nn.Module):
    def __init__(self):
        super(LI,self).__init__()
        act_fn = nn.ReLU(inplace=True)

        self.level1_r = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), act_fn)
        self.level1_t = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), act_fn)
        self.level1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level2_r = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), act_fn)
        self.level2_t = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), act_fn)
        self.level2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_21 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level3_r = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), act_fn)
        self.level3_t = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), act_fn)
        self.level3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_31 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level4_r = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), act_fn)
        self.level4_t = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), act_fn)
        self.level4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_41 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, t, level=1):
        if level ==1:
            t_mask = self.sigmoid(self.level1(self.level1_t(t)))
            rgb = self.level1_r(rgb)
            return self.layer_11(rgb * t_mask + rgb)
        elif level==2:
            t_mask = self.sigmoid(self.level2(self.level2_t(t)))
            rgb = self.level2_r(rgb)
            return self.layer_21(rgb * t_mask + rgb)
        elif level==3:
            t_mask = self.sigmoid(self.level3(self.level3_t(t)))
            rgb = self.level3_r(rgb)
            return self.layer_31(rgb * t_mask + rgb)
        else:
            t_mask = self.sigmoid(self.level4(self.level4_t(t)))
            rgb = self.level4_r(rgb)
            return self.layer_41(rgb * t_mask + rgb)

class TC(nn.Module):
    def __init__(self):
        super(TC,self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.level1_r = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), act_fn)
        self.level1_t = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), act_fn)
        self.level1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level2_r = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), act_fn)
        self.level2_t = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), act_fn)
        self.level2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_21 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level3_r = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), act_fn)
        self.level3_t = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), act_fn)
        self.level3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_31 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.level4_r = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), act_fn)
        self.level4_t = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), act_fn)
        self.level4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer_41 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), act_fn, )
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, t, level=1):
        if level == 1:
            rgb_mask = self.sigmoid(self.level1(self.level1_r(rgb)))
            t = self.level1_t(t)
            return self.layer_11(t * rgb_mask + t)
        elif level == 2:
            rgb_mask = self.sigmoid(self.level2(self.level2_r(rgb)))
            t = self.level2_t(t)
            return self.layer_21(t * rgb_mask + t)
        elif level == 3:
            rgb_mask = self.sigmoid(self.level3(self.level3_r(rgb)))
            t = self.level3_t(t)
            return self.layer_31(t * rgb_mask + t)
        else:
            rgb_mask = self.sigmoid(self.level4(self.level4_r(rgb)))
            t = self.level4_t(t)
            return self.layer_41(t * rgb_mask + t)

class IC(nn.Module):
    def __init__(self):
        super(IC,self).__init__()
        self.level11_r = nn.Conv2d(256, 256, 1, 1, 0)
        self.level12_r = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.level11_t = convblock(256, 256, 1, 1, 0)
        self.level12_t = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))

        self.level21_r = convblock(512, 256, 1, 1, 0)
        self.level22_r = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.level21_t = convblock(512, 256, 1, 1, 0)
        self.level22_t = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))

        self.level31_r = convblock(1024, 256, 1, 1, 0)
        self.level32_r = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.level31_t = convblock(1024, 256, 1, 1, 0)
        self.level32_t = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))

        self.level41_r = convblock(2048, 256, 1, 1, 0)
        self.level42_r = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.level41_t = convblock(2048, 256, 1, 1, 0)
        self.level42_t = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, rgb, t, level=1):
        if level ==1:
            rgb1 = self.level11_r(rgb)
            t1 = self.level11_t(t)
            return self.relu(rgb1 + self.level12_r(rgb1)), self.relu(t1 + self.level12_t(t1))
        elif level==2:
            rgb1 = self.level21_r(rgb)
            t1 = self.level21_t(t)
            return self.relu(rgb1 + self.level22_r(rgb1)), self.relu(t1 + self.level22_t(t1))
        elif level==3:
            rgb1 = self.level31_r(rgb)
            t1 = self.level31_t(t)
            return self.relu(rgb1 + self.level32_r(rgb1)), self.relu(t1 + self.level32_t(t1))
        else:
            rgb1 = self.level41_r(rgb)
            t1 = self.level41_t(t)
            return self.relu(rgb1 + self.level42_r(rgb1)), self.relu(t1 + self.level42_t(t1))
#FAL
class FAL(nn.Module):
    def __init__(self):
        super(FAL,self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.ca1 = CA(128)
        self.ca2 = CA(128)
        self.ca3 = CA(128)
        self.ca4 = CA(128)
        self.fal1 = nn.Sequential(nn.Conv2d(256 * 8, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), act_fn)
        self.fal2 = nn.Sequential(nn.Conv2d(256 * 8, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), act_fn)
        self.fal3 = nn.Sequential(nn.Conv2d(256 * 8, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), act_fn)
        self.fal4 = nn.Sequential(nn.Conv2d(256 * 8, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), act_fn)
    def forward(self, feats_list, level=1):
        if level ==1:
            return self.ca1(self.fal1(torch.cat(feats_list, dim=1)))
        elif level==2:
            return self.ca2(self.fal2(torch.cat(feats_list, dim=1)))
        elif level==3:
            return self.ca3(self.fal3(torch.cat(feats_list, dim=1)))
        else:
            return self.ca4(self.fal4(torch.cat(feats_list, dim=1)))
# channel attention
class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // 2, 1, bias=False)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Conv2d(in_ch // 2, in_ch, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_weight(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_weight(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

# space attention
class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x
#******************************************************************

class IIGM(nn.Module):
    def __init__(self, in1=128, in2=128, in3=128):
        super(IIGM, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in1, 128, 3, 1, 1), act_fn)
        self.conv2 = nn.Sequential(nn.Conv2d(in2, 128, 3, 1, 1), act_fn)
        self.conv3 = nn.Sequential(nn.Conv2d(in3, 128, 1, 1, 0), act_fn)
        # self.conv_add = convblock(128, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self,f1, f2, f3):
        if f1.size()[2:] != f3.size()[2:]:
            f1 = self.conv1(F.interpolate(f1, size=f3.size()[2:], mode='bilinear', align_corners=True))
        if f2.size()[2:] != f3.size()[2:]:
            f2 = self.conv2(F.interpolate(f2, size=f3.size()[2:], mode='bilinear', align_corners=True))
        f3 = self.conv3(f3)
        mask_12 = self.sigmoid(f1 + f2)
        f3_1 = f3 * mask_12
        f3_2 = self.relu(f3_1 + f3)
        return f3_2

#Progressively fusion model
class PFM(nn.Module):
    def __init__(self, channel_h=128, channel_l=128):
        super(PFM, self).__init__()
        self.conv_high_fusion = convblock(channel_h, 128, 1, 1, 0)
        self.conv_cat_fusion = convblock(128 *2, 128, 3, 1, 1)
        self.conv_low_fusion = convblock(channel_l, 128, 1, 1, 0)
        self.sigmoid_fusion = nn.Sigmoid()

        self.conv_high_rgb = convblock(channel_h, 128, 1, 1, 0)
        self.conv_cat_rgb = convblock(128*2, 128, 3, 1, 1)
        self.conv_low_rgb = convblock(channel_l, 128, 1, 1, 0)
        self.sigmoid_rgb = nn.Sigmoid()

        self.conv_high_t = convblock(channel_h, 128, 1, 1, 0)
        self.conv_cat_t = convblock(128*2, 128, 3, 1, 1)
        self.conv_low_t = convblock(channel_l, 128, 1, 1, 0)
        self.sigmoid_t = nn.Sigmoid()
    def forward(self, f_low, f_high, mode=2):
        if mode==2:
            if f_high.size()[2:] != f_low.size()[2:]:
                f_high = self.conv_high_fusion(F.interpolate(f_high, size=f_low.size()[2:], mode='bilinear', align_corners=True))
            f_low = self.conv_low_fusion(f_low)
            f_cat = self.conv_cat_fusion(torch.cat([f_low,f_high],1))
            return f_cat
        elif mode==1:
            if f_high.size()[2:] != f_low.size()[2:]:
                f_high = self.conv_high_rgb(F.interpolate(f_high, size=f_low.size()[2:], mode='bilinear', align_corners=True))
            f_low = self.conv_low_rgb(f_low)
            f_cat = self.conv_cat_rgb(torch.cat([f_low,f_high],1))
            # mask = self.sigmoid_rgb(f_cat)
            # f_low_mask = f_low * mask
            return f_cat
        else:
            if f_high.size()[2:] != f_low.size()[2:]:
                f_high = self.conv_high_t(F.interpolate(f_high, size=f_low.size()[2:], mode='bilinear', align_corners=True))
            f_low = self.conv_low_t(f_low)
            f_cat = self.conv_cat_t(torch.cat([f_low,f_high],1))
            return f_cat

#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

#the main network
class Net(nn.Module):
    def __init__(self,channel=32, ind=50):
        super(Net, self).__init__()
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        self.backbone_rgb = Res2Net_model(ind)
        self.backbone_t = Res2Net_model(ind)
        self.fal1 = FAL()

        self.IC1 = IC()
        self.NV1 = NV()
        self.CB1 = CB()
        self.LI1 = LI()
        self.TC1 = TC()

        self.fal2 = FAL()
        self.IC2 = IC()
        self.NV2 = NV()
        self.CB2 = CB()
        self.LI2 = LI()
        self.TC2 = TC()

        self.fal3 = FAL()
        self.IC3 = IC()
        self.NV3 = NV()
        self.CB3 = CB()
        self.LI3 = LI()
        self.TC3 = TC()

        self.fal4 = FAL()
        self.IC4 = IC()
        self.NV4 = NV()
        self.CB4 = CB()
        self.LI4 = LI()
        self.TC4 = TC()
        #decoder
        self.iigm4_fusion = IIGM()
        self.iigm3_fusion = IIGM()
        self.iigm2_fusion = IIGM()
        self.iigm1_fusion = IIGM()

        ## fusion
        self.ful_gcm_4 = GCM(128, channel)
        self.ful_gcm_3 = GCM(128 + 32, channel)
        self.ful_gcm_2 = GCM(128 + 32, channel)
        self.ful_gcm_1 = GCM(128 + 32, channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.score1_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score2_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score3_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score4_fusion = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, rgb, t):
        #-------------------encoder---------------------------#
        fr = self.backbone_rgb(rgb)
        ft = self.backbone_t(self.layer_dep0(t))

        LI_1 = self.LI1(fr[1], ft[1], 1)
        TC_1 = self.TC1(fr[1], ft[1], 1)
        IC_1r, IC_1t = self.IC1(fr[1], ft[1], 1)
        NV_1r, NV_1t = self.NV1(fr[1], ft[1], 1)
        CB_1r, CB_1t = self.CB1(fr[1], ft[1], 1)
        fal1 = self.fal1([LI_1, TC_1, IC_1r, IC_1t, NV_1r, NV_1t, CB_1r, CB_1t], 1)
        LI_2 = self.LI2(fr[2], ft[2], 2)
        TC_2 = self.TC2(fr[2], ft[2], 2)
        IC_2r, IC_2t = self.IC2(fr[2], ft[2], 2)
        NV_2r, NV_2t = self.NV2(fr[2], ft[2], 2)
        CB_2r, CB_2t = self.CB2(fr[2], ft[2], 2)
        fal2 = self.fal2([LI_2, TC_2, IC_2r, IC_2t, NV_2r, NV_2t, CB_2r, CB_2t], 2)
        LI_3 = self.LI3(fr[3], ft[3], 3)
        TC_3 = self.TC3(fr[3], ft[3], 3)
        IC_3r, IC_3t = self.IC3(fr[3], ft[3], 3)
        NV_3r, NV_3t = self.NV3(fr[3], ft[3], 3)
        CB_3r, CB_3t = self.CB3(fr[3], ft[3], 3)
        fal3 = self.fal3([LI_3, TC_3, IC_3r, IC_3t, NV_3r, NV_3t, CB_3r, CB_3t], 3)
        LI_4 = self.LI4(fr[4], ft[4], 4)
        TC_4 = self.TC4(fr[4], ft[4], 4)
        IC_4r, IC_4t = self.IC4(fr[4], ft[4], 4)
        NV_4r, NV_4t = self.NV4(fr[4], ft[4], 4)
        CB_4r, CB_4t = self.CB4(fr[4], ft[4], 4)
        fal4 = self.fal4([LI_4, TC_4, IC_4r, IC_4t, NV_4r, NV_4t, CB_4r, CB_4t], 4)
        #--------------------------decoder-------------------------

        iigm4_fusion = self.iigm4_fusion(fal2, fal3, fal4)
        iigm3_fusion = self.iigm3_fusion(fal1, fal2, fal3)
        iigm2_fusion = self.iigm2_fusion(fal4, fal3, fal2)
        iigm1_fusion = self.iigm1_fusion(fal3, fal2, fal1)

        #fusion
        x_ful_42 = self.ful_gcm_4(iigm4_fusion)

        x_ful_3_cat = torch.cat([iigm3_fusion, self.upsample_2(x_ful_42)], dim=1)
        x_ful_32 = self.ful_gcm_3(x_ful_3_cat)

        x_ful_2_cat = torch.cat([iigm2_fusion, self.upsample_2(x_ful_32)], dim=1)
        x_ful_22 = self.ful_gcm_2(x_ful_2_cat)

        x_ful_1_cat = torch.cat([iigm1_fusion, self.upsample_2(x_ful_22)], dim=1)
        x_ful_12 = self.ful_gcm_1(x_ful_1_cat)

        out1_fusion = F.interpolate(self.score1_fusion(x_ful_12), size=rgb.size()[2:], mode='bilinear', align_corners=True)
        out2_fusion = F.interpolate(self.score2_fusion(x_ful_22), size=rgb.size()[2:], mode='bilinear',
                                    align_corners=True)
        out3_fusion = F.interpolate(self.score3_fusion(x_ful_32), size=rgb.size()[2:], mode='bilinear',
                                    align_corners=True)
        out4_fusion = F.interpolate(self.score4_fusion(x_ful_42), size=rgb.size()[2:], mode='bilinear',
                                    align_corners=True)

        return out1_fusion, out2_fusion, out3_fusion, out4_fusion #out, out1_rgb, out1_t,
