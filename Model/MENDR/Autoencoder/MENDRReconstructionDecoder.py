import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import floor
import copy
from copy import deepcopy
from einops import rearrange

class MENDRReconstructionDecoder(nn.Module):
    def __init__(self, num_channels, sub_patch_size, encoded_h, hidden_seq_length, seq_len, device):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = (sub_patch_size // 2) + 1
        self.stride = 1
        self.padding = max(1, (self.patch_size - 1) // 2)
        self.device = device
        self.seq_len = seq_len
        self.encoded_h = encoded_h

        '''
        self.decode_L_out = (self.L_out_2 - 1) * self.stride - 2 * 0 + 1 * (self.patch_size - 1) + 0 + 1
        self.up1 = ResidualConvTranspose1dBlock(in_channels=encoded_h, out_channels=encoded_h * 2, kernel_size=self.patch_size, stride=self.stride, padding=0, activation=nn.GELU()).to(self.device)
        self.decode_L_out = (self.decode_L_out - 1) * (self.stride) - 2 * 0 + 1 * (2 - 1) + 0 + 1
        self.up2 = ResidualConvTranspose1dBlock(in_channels=encoded_h * 2, out_channels=encoded_h * 2, kernel_size=2, stride=self.stride, padding=0, activation=nn.GELU()).to(self.device)
        self.decode_L_out = self.decode_L_out + 2 * 0 - 1 * (encoded_h - 1) - 1
        self.decode_L_out = floor((self.decode_L_out / (self.stride)) + 1)
        self.up3 = ResidualConv1dBlock(in_channels=encoded_h * 2, out_channels=19, kernel_size=encoded_h, stride=self.stride, padding=0, activation=nn.GELU()).to(self.device)
        self.up4 = nn.Linear(self.decode_L_out, self.seq_len).to(self.device)
        '''

        '''
        self.decode_L_out = (self.hidden_seq_length - 1) * self.stride - 2 * 0 + 1 * (self.patch_size - 1) + 0 + 1
        self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels=encoded_h, out_channels=encoded_h * 2, kernel_size=self.patch_size, stride=self.stride, groups=1).to(self.device), self.act)
        self.decode_L_out = (self.decode_L_out - 1) * (self.stride) - 2 * 0 + 1 * (2 - 1) + 0 + 1
        self.SEBlock1 = SEBasicBlock(encoded_h*2, encoded_h*2, reduction=15).to(self.device)
        self.up2 = nn.Sequential(nn.ConvTranspose1d(in_channels=encoded_h * 2, out_channels=encoded_h * 2, kernel_size=2, stride=self.stride, groups=1).to(self.device), self.act)
        self.decode_L_out = self.decode_L_out + 2 * 0 - 1 * (1 - 1) - 1
        self.decode_L_out = floor((self.decode_L_out / 1) + 1)
        self.up3 = nn.Sequential(nn.Conv1d(in_channels=encoded_h * 2, out_channels=19, kernel_size=1, stride=1, groups=1).to(self.device), self.act)
        self.up4 = nn.Linear(self.decode_L_out, self.seq_len).to(self.device)
        '''

        self.act = nn.GELU()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.encoded_h, out_channels=self.encoded_h, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GroupNorm(num_groups=4, num_channels=self.encoded_h),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.encoded_h, out_channels=self.encoded_h, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.encoded_h, out_channels=self.encoded_h, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.encoded_h, out_channels=1, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            #nn.GroupNorm(num_groups=4, num_channels=self.num_channels),
            nn.GELU(),

        )

        self.SEBlock = SEBasicBlock(self.num_channels, self.num_channels, reduction=1)
        self.up_lin = nn.Sequential(self.act, nn.Linear(self.seq_len, self.seq_len))

    def forward(self, x, B, P, C, T):
        # x: [Batch Size, Patches, Channels * patch_embedder.out_dim, Time Steps]
        #x  = self.SEBlock(x)
        x = rearrange(x, 'B P (C O) T -> B O (P C) T', B=B, P=P, C=C, T=T)
        # x: [Batch Size, out_dim, Patches * Channels, Time Steps]
        x = self.up(x)
        # x: [Batch Size, 1, Patches * Channels, Time Steps]
        x = x.squeeze(1)
        x = rearrange(x, 'B (P C) T -> (B P) C T', B=B, P=P, C=C, T=T)
        # x: [Batch Size * Patches, Channels, Time Steps]
        x = self.SEBlock(x)
        x = self.up_lin(x)
        # x: [Batch Size * Patches, Channels, Time Steps]
        return x

# The Conv -> BN -> GELU -> Conv part of the anwswer
# https://stackoverflow.com/questions/49045843/why-is-relu-applied-after-residual-connection-in-resnet
class ResidualConvTranspose1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, activation=nn.GELU()):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.activation = activation
        # Residual connection to match dimensions if needed
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv_transpose(x)
        out = self.activation(out)
        # Residual connection
        residual = self.residual(x)
        if residual.size(-1) != out.size(-1):
            residual = F.interpolate(residual, size=out.size(-1), mode='linear')
        return out + residual

class ResidualConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, activation=nn.GELU()):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.activation = activation
        # Residual connection to match dimensions if needed
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv(x)
        out = self.activation(out)
        # Residual connection
        residual = self.residual(x)
        if residual.size(-1) != out.size(-1):
            residual = F.interpolate(residual, size=out.size(-1), mode='linear')

        return out + residual

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
                downsample=None, groups=1,
                dilation=1, norm_layer=None,
                 *, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

if __name__ == "__main__":
    # Delta Parameters
    num_channels = 19
    sub_patch_size = 4
    encoded_h = 19
    L_out_2 = 18
    seq_len = 246

    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(1, encoded_h, L_out_2, device=device)
    model = MENDRReconstructionDecoder(num_channels=num_channels, sub_patch_size=sub_patch_size, encoded_h=encoded_h, hidden_seq_length=L_out_2, seq_len=seq_len, device=device)
    y = model(x)
    print("Delta Shape:", y.shape)
    print("Delta Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Alpha Parameters
    num_channels = 19
    sub_patch_size = 8
    encoded_h = 38
    L_out_2 = 18
    seq_len = 486

    x = torch.randn(1, encoded_h, L_out_2, device=device)
    model = MENDRReconstructionDecoder(num_channels=num_channels, sub_patch_size=sub_patch_size, encoded_h=encoded_h, hidden_seq_length=L_out_2, seq_len=seq_len, device=device)
    y = model(x)
    print("Alpha Shape:", y.shape)
    print("Alpha Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


