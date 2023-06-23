#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : pafnet.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sat Nov 26 2022
# @version   : 2.0
# @license   : The Apache License 2.0
# @note      : 
# 
# The Apache 2.0 License
# Copyright (C) 2013- Zhi Liu
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

import os
import time
import argparse
import torch as th
import torchbox as tb
import torchsar as ts
from pafnet import PAFnet
from dataset import readsamples, saveimage
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./afnet.yaml')
parser.add_argument('--solvercfg', type=str, default='./solver.yaml')

# params in Adam
parser.add_argument('--modelfile', type=str, default=None)
parser.add_argument('--loss_type', type=str, default='Entropy', help='Entropy, Contrast, LogFro')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=40)
parser.add_argument('--num_epochs', type=int, default=1000)
# parser.add_argument('--optimizer', type=str, default='Adadelta')
parser.add_argument('--optimizer', type=str, default='AdamW')
# parser.add_argument('--optimizer', type=str, default='AdamW')
# parser.add_argument('--scheduler', type=str, default='DoubleGaussianKernelLR')
# parser.add_argument('--scheduler', type=str, default='LambdaLR')
parser.add_argument('--scheduler', type=str, default='StepLR')
# parser.add_argument('--optimizer', type=str, default='SGD')
# parser.add_argument('--scheduler', type=str, default='OneCycleLR')
# parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--snapshot_name', type=str, default='2020')

# misc
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(SimPoly, SimSin...)')
parser.add_argument('--axismode', type=str, default='fftfreq')

cfg = parser.parse_args()

device = cfg.device
# device = th.device('cpu')

seed = 2020
isplot = True
isplot = False
issaveimg = True
issaveimg = False
ftshift = True

datacfg = tb.loadyaml(cfg.datacfg)
modelcfg = tb.loadyaml(cfg.modelcfg)
solvercfg = tb.loadyaml(cfg.solvercfg)
solvercfg['nepoch'] = cfg.num_epochs if cfg.num_epochs is not None else solvercfg['sbatch']
solvercfg['sbatch'] = cfg.size_batch if cfg.size_batch is not None else solvercfg['sbatch']
num_epochs = solvercfg['nepoch']
size_batch = solvercfg['sbatch']

if 'SAR_AF_DATA_PATH' in os.environ.keys():
    datafolder = os.environ['SAR_AF_DATA_PATH']
else:
    datafolder = datacfg['SAR_AF_DATA_PATH']

print(cfg)
# print(datacfg)
print(modelcfg)

fileTrain = [datafolder + datacfg['filenames'][i] for i in datacfg['trainid']]
fileValid = [datafolder + datacfg['filenames'][i] for i in datacfg['validid']]
fileTest = [datafolder + datacfg['filenames'][i] for i in datacfg['testid']]

modeTest = 'sequentially'
print("--->Test files sampling mode:", modeTest)
print(fileTest)

keys = [['SI', 'mea_steplr_poly_ca', 'mea_steplr_poly_cr']]
X, ca, cr = readsamples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

N, Na, Nr, _ = X.shape

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=cfg.axismode)
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=cfg.axismode)

if cfg.mkpetype in ['simpoly', 'SimPoly']:
    print("---Making polynominal phase error...")
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
    X = ts.focus(X, pa, None, isfft=True, ftshift=ftshift)

    carange = [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]]
    crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]
    print('~~~carange', carange)
    print('~~~crrange', crrange)
    ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
    _, _ = ppeg.mkpec(n=6000, seed=None)
    _, _ = ppeg.mkpec(n=8000, seed=None)
    ca, cr = ppeg.mkpec(n=N, seed=None)
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
    X = ts.defocus(X, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

# index = [89, 1994, 7884]
# print(index)
# X, ca, cr = X[index], ca[index], cr[index]

numSamples = X.shape[0]


print("--->Test:", X.shape, ca.shape, cr.shape)

if modelcfg['model'] == 'AFnet':
    outfolder, weight = 'record/RealPE/AFnet/AdamW/GaussianLR_nsamples6000/', 'weights/best_valid_664.pth.tar'
if modelcfg['model'] == 'PAFnet':
    outfolder, weight = 'record/RealPE/PAFnet/AdamW/GaussianLR_6000Samples/3Focuser4ConvLayers/', 'weights/best_valid_655.pth.tar'

if cfg.modelfile is not None:
    modelfile = cfg.modelfile
else:
    modelfile = outfolder + weight

os.makedirs(outfolder + '/tests', exist_ok=True)

print(device)


net = PAFnet(Na, 1, Mas=modelcfg['Mas'], lpaf=modelcfg['lpaf'], ftshift=ftshift, seed=seed)

modelparams = th.load(modelfile, map_location=device)
# print(modelparams)
net.load_state_dict(modelparams['network'])
print(net)

net.to(device=device)
net.eval()

loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(1, 2), reduction='mean')  # OK
loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(1, 2), reduction='mean')  # OK
loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(1, 2), reduction='mean')

tstart = time.time()


sizeBatch = min(numSamples, size_batch)
numBatch = int(numSamples / sizeBatch)
idx = list(range(numSamples))
lossENTv, lossCTSv, lossFROv, lossvtest = 0., 0., 0., 0.
with th.no_grad():
    for b in range(numBatch):
        i = idx[b * sizeBatch:(b + 1) * sizeBatch]
        xi, cai, cri = X[i], ca[i], cr[i]
        xi = xi.to(device)

        fi, pcai = net.forward(xi)

        lossENT = loss_ent_func(fi)
        lossCTS = loss_cts_func(fi)
        lossFRO = loss_fro_func(fi)

        if cfg.loss_type == 'Entropy':
            loss = lossENT
        if cfg.loss_type == 'Entropy+LogFro':
            loss = lossENT + lossFRO
        if cfg.loss_type == 'Contrast':
            loss = lossCTS
        if cfg.loss_type == 'Entropy/Contrast':
            loss = lossENT / lossCTS

        lossvtest += loss.item()
        lossCTSv += lossCTS.item()
        lossENTv += lossENT.item()
        lossFROv += lossFRO.item()


        if issaveimg:
            saveimage(xi, fi, i, prefixname='test', outfolder=outfolder + '/tests/')

        if isplot:
            pai = ts.polype(c=cai, x=xa)
            print(pcai.shape)
            ppai = ts.polype(c=pcai, x=xa)
            pai = pai.detach().cpu().numpy()
            ppai = ppai.detach().cpu().numpy()
            for ii in range(len(i)):
                plt.figure()
                plt.plot(pai[ii], '-b')
                plt.plot(ppai[ii], '-r')
                plt.legend(['Real', 'Estimated'])
                plt.grid()
                plt.xlabel('Aperture time (samples)')
                plt.ylabel('Phase (rad)')
                plt.title('Estimated phase error (polynomial)')
                # plt.show()
                plt.savefig(outfolder + '/tests/phase_error_azimuth' + str(i[ii]) + '.png')
                plt.close()


tend = time.time()

lossvtest /= numBatch
lossCTSv /= numBatch
lossENTv /= numBatch
lossFROv /= numBatch

print("--->Test: loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
      (lossvtest, lossENTv, lossFROv, lossCTSv, tend - tstart))


