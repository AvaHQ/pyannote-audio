# Copyright (c) 2024 XiaoyiQin, Yuke Lin (linyuke0609@gmail.com)
#               2024 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
#import wespeaker.models.pooling_layers as pooling_layers

from pyannote.audio.models.blocks.pooling import StatsPool
from typing import Any
import warnings



class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x, weights):
        
        if weights.dim() < 3:
            weights = weights.unsqueeze(dim=1)

        _, _, _, num_frames = x.shape
        _, _, num_weights = weights.shape
        
        if num_frames != num_weights:
            warnings.warn(
                f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
            )
            weights = F.interpolate(weights, size=num_frames, mode="nearest")
        
        #weights = weights.unsqueeze(dim=1)

        #mean = torch.sum(sequences * weights, dim=2) / v1

        x = x.reshape(x.size()[0], -1, x.size()[-1])
        #x = (x*weights).squeeze(dim=1)
        x = (x[:,None,:,:]*weights[:,:,None,:])
        x = x.reshape(-1, x.size()[2], x.size()[3])
        
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)

        return x

class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1
    ):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                NormLayer(self.expansion * planes),
            )
            
        self.skip_add = nn.quantized.FloatFunctional()
        self.quant_2 = torch.quantization.QuantStub()
        self.dequant_2 = torch.quantization.DeQuantStub()
        

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dequant_2(out)
        out = self.SimAM(out)
        out = self.quant_2(out)
        #out += self.downsample(x)
        out = self.skip_add.add(self.downsample(x), out)
        out = self.relu(out)
        return out


    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class ResNet(nn.Module):
    def __init__(
        self, in_planes, block, num_blocks, in_ch=1, **kwargs
    ):
        super(ResNet, self).__init__()
        self.quant_1 = torch.quantization.QuantStub()
        self.dequant_1 = torch.quantization.DeQuantStub()
        
        self.in_planes = in_planes
        self.NormLayer = nn.BatchNorm2d
        self.ConvLayer = nn.Conv2d

        self.conv1 = self.ConvLayer(
            in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, in_planes, num_blocks[0], stride=1, block_id=1
        )
        self.layer2 = self._make_layer(
            block, in_planes * 2, num_blocks[1], stride=2, block_id=2
        )
        self.layer3 = self._make_layer(
            block, in_planes * 4, num_blocks[2], stride=2, block_id=3
        )
        self.layer4 = self._make_layer(
            block, in_planes * 8, num_blocks[3], stride=2, block_id=4
        )

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.ConvLayer,
                    self.NormLayer,
                    self.in_planes,
                    planes,
                    stride,
                    block_id,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dequant_1(x)
        return x


def SimAM_ResNet34(in_planes):
    return ResNet(in_planes, SimAMBasicBlock, [3, 4, 6, 3])


def SimAM_ResNet100(in_planes):
    return ResNet(in_planes, SimAMBasicBlock, [6, 16, 24, 3])


class SimAM_ResNet34_ASP(nn.Module):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet34_ASP, self).__init__()
        self.front = SimAM_ResNet34(in_planes)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        
        self.quant_3 = torch.quantization.QuantStub()
        self.dequant_3 = torch.quantization.DeQuantStub()
        

    def forward(self, x, weights: torch.Tensor = None):
        x = x.permute(0, 2, 1)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x, weights)
        if self.drop:
            x = self.drop(x)
        x = self.quant_3(x)
        x = self.bottleneck(x)
        return self.dequant_3(x)


class SimAM_ResNet100_ASP(nn.Module):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet100_ASP, self).__init__()
        self.front = SimAM_ResNet100(in_planes)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = SimAM_ResNet34_ASP(embed_dim=256)
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))