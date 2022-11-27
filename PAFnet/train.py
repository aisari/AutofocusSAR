#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : test.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sat Nov 26 2022
# @version   : 2.0
# @license   : The Apache License 2.0
# @note      : 
# 
# The Apache 2.0 License (MIT)
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
import argparse
import torch as th
import torchbox as tb
import torchsar as ts
from pafnet import PAFnet
from dataset import readsamples

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./afnet.yaml')
parser.add_argument('--solvercfg', type=str, default='./solver.yaml')

# params in Adam
parser.add_argument('--loss_type', type=str, default='Entropy', help='Entropy, Contrast, LogFro')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=1000)
# parser.add_argument('--optimizer', type=str, default='Adadelta')
parser.add_argument('--optimizer', type=str, default='AdamW')
# parser.add_argument('--optimizer', type=str, default='AdamW')
# parser.add_argument('--scheduler', type=str, default='GaussianLR')
# parser.add_argument('--scheduler', type=str, default='LambdaLR')
parser.add_argument('--scheduler', type=str, default='StepLR')
# parser.add_argument('--optimizer', type=str, default='SGD')
# parser.add_argument('--scheduler', type=str, default='OneCycleLR')
# parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--snapshot_name', type=str, default='2020')

# misc
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(SimPoly, SimSin...)')

cfg = parser.parse_args()

seed = cfg.seed
ftshift = True

cudaTF32, cudnnTF32 = False, False
# benchmark: if False, training is slow, if True, training is fast but conv results may be slight different for different cards.
benchmark, deterministic = True, True

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

# fileTrain = [datafolder + datacfg['filenames'][0]]
# fileValid = [datafolder + datacfg['filenames'][0]]
# fileTest = [datafolder + datacfg['filenames'][0]]

modeTrain, modeValid, modeTest = 'sequentially', 'sequentially', 'sequentially'
print("--->Train files sampling mode:", modeTrain)
print(fileTrain)
print("--->Valid files sampling mode:", modeValid)
print(fileValid)
print("--->Test files sampling mode:", modeTest)
print(fileTest)

keys = [['SI', 'mea_steplr_poly_ca', 'mea_steplr_poly_cr']]
Xtrain, catrain, crtrain = readsamples(fileTrain, keys=keys, nsamples=[1200], groups=[25], mode=modeTrain, seed=seed)
Xvalid, cavalid, crvalid = readsamples(fileValid, keys=keys, nsamples=[4000], groups=[25], mode=modeValid, seed=seed)
Xtest, catest, crtest = readsamples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

N, Na, Nr, _ = Xtrain.size()

Ntrain, Nvalid, Ntest = Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='2fftfreq')
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='2fftfreq')

if cfg.mkpetype in ['simpoly', 'SimPoly']:
    print("---Making polynominal phase error...")
    pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
    Xtrain = ts.focus(Xtrain, pa, None, isfft=True, ftshift=ftshift)
    pa, pr = ts.polype(cavalid, xa), ts.polype(crvalid, xr)
    Xvalid = ts.focus(Xvalid, pa, None, isfft=True, ftshift=ftshift)
    pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
    Xtest = ts.focus(Xtest, pa, None, isfft=True, ftshift=ftshift)

    carange = [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]]
    crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]
    print('~~~carange', carange)
    print('~~~crrange', crrange)
    ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
    catrain, crtrain = ppeg.mkpec(n=Ntrain, seed=None)
    pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
    Xtrain = ts.defocus(Xtrain, pa, None, isfft=True, ftshift=ftshift)
    cavalid, crvalid = ppeg.mkpec(n=Nvalid, seed=None)
    pa, pr = ts.polype(cavalid, xa), ts.polype(crvalid, xr)
    Xvalid = ts.defocus(Xvalid, pa, None, isfft=True, ftshift=ftshift)
    catest, crtest = ppeg.mkpec(n=Ntest, seed=None)
    pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
    Xtest = ts.defocus(Xtest, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

print("--->Train:", Xtrain.shape, catrain.shape, crtrain.shape)
print("--->Valid:", Xvalid.shape, cavalid.shape, crvalid.shape)
print("--->Test:", Xtest.shape, catest.shape, crtest.shape)
numSamples = N

outfolder = './snapshot/' + modelcfg['model'] + '/' + cfg.mkpetype + '/' + cfg.loss_type + '/' + cfg.optimizer + '/' + cfg.scheduler + '/' + cfg.snapshot_name
losslog = tb.LossLog(outfolder)

os.makedirs(outfolder + '/images', exist_ok=True)
os.makedirs(outfolder + '/weights', exist_ok=True)

device = cfg.device
# device = th.device(cfg.device if th.cuda.is_available() else 'cpu')
devicename = 'E5 2696v3' if device == 'cpu' else th.cuda.get_device_name(tb.str2num(device, int))

net = PAFnet(Na, 1, Mas=modelcfg['Mas'], lpaf=modelcfg['lpaf'], xa=xa, ftshift=ftshift, seed=seed)

checkpoint_path = outfolder + '/weights/'

loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')  # OK
loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')  # OK
loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')

# if cfg.modelfile is not None:
#     modelfile = cfg.modelfile
# else:
#     modelfile = './snapshot/AFnet/AdamW/2020/weights/current.pth.tar'
# logdict = th.load(modelfile, map_location=device)
# sepoch = logdict['epoch']

logdict = {}
sepoch = 1

last_epoch = sepoch - 2

opt = solvercfg['Optimizer'][cfg.optimizer]
opt['lr'] = cfg.lr if cfg.lr is not None else opt['lr']

if cfg.optimizer == 'Adam':
    optimizer = th.optim.Adam([{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': opt['lr']}], lr=opt['lr'], betas=opt['betas'], eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'])
if cfg.optimizer in ['adamw', 'AdamW']:
    optimizer = th.optim.AdamW([{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': opt['lr']}], lr=opt['lr'], betas=opt['betas'], eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'])
if cfg.optimizer in ['sgd', 'SGD']:
    opt = solvercfg['Optimizer']['AdamW']
    optimizer = th.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=opt['lr'], momentum=opt['momentum'], dampening=opt['dampening'], weight_decay=opt['weight_decay'], nesterov=opt['nesterov'])

scheduler = cfg.scheduler
sch = solvercfg['LRScheduler'][cfg.scheduler]
if cfg.scheduler == 'GaussianLR':
    scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=sch['t_eta_max'], sigma1=sch['sigma1'], sigma2=sch['sigma2'], eta_start=sch['eta_start'], eta_stop=sch['eta_stop'], last_epoch=last_epoch)
if cfg.scheduler == 'StepLR':
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=sch['step_size'], gamma=sch['gamma'], last_epoch=last_epoch)
if cfg.scheduler == 'ExponentialLR':
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sch['gamma'], last_epoch=last_epoch)
if cfg.scheduler == 'CosineAnnealingLR':
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch['T_max'].num_epochs, eta_min=sch['eta_min'], last_epoch=last_epoch)
if cfg.scheduler == 'OneCycleLR':
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, opt['lr'], total_steps=None, epochs=scfg['nepoch'], steps_per_epoch=int(
        N / size_batch), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1000.0, last_epoch=last_epoch)

if sepoch > 1:
    net.load_state_dict(logdict['network'])
    optimizer.load_state_dict(logdict['optimizer'])
    scheduler.load_state_dict(logdict['scheduler'])
    losslog.assign('train', logdict['lossestrain'])
    losslog.assign('valid', logdict['lossesvalid'])
    losslog.assign('test', logdict['lossestest'])

print("---", cfg.optimizer, opt)
print("---", cfg.scheduler, sch)
print(optimizer)
print(scheduler)

print("---", device)
print("---", devicename)
print("---Torch Version: ", th.__version__)
print("---Torch CUDA Version: ", th.version.cuda)
print("---CUDNN Version: ", th.backends.cudnn.version())
print("---CUDA TF32: ", cudaTF32)
print("---CUDNN TF32: ", cudnnTF32)
print("---CUDNN Benchmark: ", benchmark)
print("---CUDNN Deterministic: ", deterministic)

th.backends.cuda.matmul.allow_tf32 = cudaTF32
th.backends.cudnn.allow_tf32 = cudnnTF32
th.backends.cudnn.benchmark = benchmark
th.backends.cudnn.deterministic = deterministic

print(net)
net.to(device=device)

tb.device_transfer(optimizer, 'optimizer', device=device)

if sepoch == 1:
    if cfg.loss_type == 'Entropy':
        lossmintrain, lossminvalid, lossmintest = loss_ent_func(Xtrain).item(), loss_ent_func(Xvalid).item(), loss_ent_func(Xtest).item()
    if cfg.loss_type == 'Entropy+LogFro':
        lossmintrain, lossminvalid, lossmintest = loss_ent_func(Xtrain).item() + loss_fro_func(Xtrain).item(), loss_ent_func(Xvalid).item() + loss_fro_func(Xvalid).item(), loss_ent_func(Xtest).item() + loss_fro_func(Xtest).item()
    if cfg.loss_type == 'Contrast':
        lossmintrain, lossminvalid, lossmintest = loss_cts_func(Xtrain).item(), loss_cts_func(Xvalid).item(), loss_cts_func(Xtest).item()

    losslog.add('train', lossmintrain)
    losslog.add('valid', lossminvalid)
    losslog.add('test', lossmintest)
else:
    lossmintrain = min(losslog.get('train'))
    lossminvalid = min(losslog.get('valid'))
    lossmintest = min(losslog.get('test'))

tb.setseed(seed)
idxtrain = list(range(0, Xtrain.shape[0], int(Xtrain.shape[0] / 16)))
idxvalid = list(range(0, Xvalid.shape[0], int(Xvalid.shape[0] / 16)))
idxtest = list(range(0, Xtest.shape[0], int(Xtest.shape[0] / 16)))
for k in range(sepoch, cfg.num_epochs):

    lossvtrain = net.train_epoch(Xtrain, size_batch,
                                 loss_ent_func, loss_cts_func, loss_fro_func, cfg.loss_type, k, optimizer, None, device)
    net.plot(Xtrain[idxtrain], catrain[idxtrain], crtrain[
             idxtrain], xa, idxtrain, 'train', outfolder, device)

    if cfg.scheduler in ['GaussianLR', 'LambdaLR', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR']:
        scheduler.step()

    lossvvalid = net.valid_epoch(Xvalid, size_batch,
                                 loss_ent_func, loss_cts_func, loss_fro_func, cfg.loss_type, k, device)
    net.plot(Xvalid[idxvalid], cavalid[idxvalid], crvalid[
             idxvalid], xa, idxvalid, 'valid', outfolder, device)

    lossvtest = net.test_epoch(Xtest, size_batch,
                               loss_ent_func, loss_cts_func, loss_fro_func, cfg.loss_type, k, device)
    net.plot(Xtest[idxtest], catest[idxtest], crtest[
             idxtest], xa, idxtest, 'test', outfolder, device)

    losslog.add('train', lossvtrain)
    losslog.add('valid', lossvvalid)
    losslog.add('test', lossvtest)
    losslog.plot()

    logdict['epoch'] = k
    logdict['lossestrain'] = losslog.get('train')
    logdict['lossesvalid'] = losslog.get('valid')
    logdict['lossestest'] = losslog.get('test')
    logdict['network'] = net.state_dict()
    logdict['optimizer'] = optimizer.state_dict()
    if cfg.scheduler is not None:
        logdict['scheduler'] = scheduler.state_dict()

    if lossvtrain <= lossmintrain:
        th.save(logdict, checkpoint_path + 'best_train' + '.pth.tar')
        lossmintrain = lossvtrain

    if lossvvalid <= lossminvalid:
        th.save(logdict, checkpoint_path + 'best_valid_' + str(k) + '.pth.tar')
        lossminvalid = lossvvalid

    if lossvtest <= lossmintest:
        th.save(logdict, checkpoint_path + 'best_test_' + str(k) + '.pth.tar')
        lossmintest = lossvtest

    th.save(logdict, checkpoint_path + 'current' + '.pth.tar')
