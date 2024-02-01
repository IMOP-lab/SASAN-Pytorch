import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import torch.nn.functional as F
from .ASEM import *

import torch
import pytorch_lightning as pl
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    

class filter_trans(nn.Module):
    def __init__(self, mode='low'):
        super(filter_trans, self).__init__()
        # self.cutoff = cutoff
        self.mode = mode
        # print("####################using FFC#####################")
    
    def forward(self, x):
        f = torch.fft.fftn(x, dim=(2, 3, 4))
        fshift = torch.fft.fftshift(f)
        if self.mode == 'high':
            fshift = torch.fft.fftshift(f)
        
        return fshift
    
class FINE(nn.Module):
    def __init__(self, rate,cutoff,feat):#feat==channel
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate),requires_grad=True)
        self.cutoff = cutoff
        self.feat=feat
        self.mask = nn.Parameter(torch.ones(1,self.feat,self.cutoff,self.cutoff,self.cutoff),requires_grad=True)

    def forward(self, x,fier):
        col = fier.shape[2]
        start_cut = (col - self.cutoff) // 2
        end_cut = start_cut + self.cutoff
        # new_mask = self.S3DSA(self.mask)
        fier2 = fier[:,:,start_cut:end_cut,start_cut:end_cut,start_cut:end_cut].repeat(1,self.feat,1,1,1)*self.mask
        x_fft = torch.fft.fftn(x, dim=(2, 3, 4))
        x_fft = torch.fft.fftshift(x_fft)
        y = x_fft*self.rate+fier2*(1-self.rate)
        y = torch.fft.fftshift(y)
        y_ifft = torch.fft.ifftn(y, dim=(2, 3, 4))
        y_ifft_real = y_ifft.real

        return y_ifft_real
    
class SASAN(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        conv_block: bool = True,
        res_block: bool = True,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
       
        self.fine1= FINE(0.5,96,32)
        self.fine2= FINE(0.5,48,64)
        self.fine3= FINE(0.5,24,128)
        self.fine4= FINE(0.5,12,256)
        self.fine5= FINE(0.5,6,512)
        self.filter_trans = filter_trans('low')

        self.asem = ASEM(axial_dim=96, spatial_dims=3, in_channels=out_channels, head=16)

    def forward(self, x: torch.Tensor):
        filter_low = self.filter_trans(x)
        x0 = self.conv_0(x)
        x0=self.fine1(x0,filter_low)*x0
        x1 = self.down_1(x0)
        x1=self.fine2(x1,filter_low)*x1
        x2 = self.down_2(x1)
        x2=self.fine3(x2,filter_low)*x2
        x3 = self.down_3(x2)
        x3=self.fine4(x3,filter_low)*x3
        x4 = self.down_4(x3)
        x4=self.fine5(x4,filter_low)*x4
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        logits = self.asem(logits)
        
        return logits
