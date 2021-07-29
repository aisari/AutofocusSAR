#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

# from __future__ import print_function

import time
import torchsar as ts
import torch as th
from torch.nn.parameter import Parameter
from collections import OrderedDict
from dataset import saveimage
import matplotlib.pyplot as plt


class Focuser(th.nn.Module):

    def __init__(self, Na, Nr, mas, convp=[[16, 17, 1]], xa=None, ftshift=True):
        r"""Focuser

        Focuser

        Parameters
        ----------
        Na : {number}
            [description]
        Nr : {number}
            1 or the number of range cells
        mas : {list}
            [description]
        ftshift : {bool}, optional
            [description] (the default is True, which [default_description])
        """

        super(Focuser, self).__init__()

        self.Na = Na
        self.Nr = Nr
        self.mas = mas
        self.ftshift = ftshift
        self.Ma = len(mas)
        self.nconvs = len(convp)

        FD = OrderedDict()
        FD['conv1'] = th.nn.Conv2d(2, convp[0][0], convp[0][1:3], stride=convp[0][3:5], padding=convp[0][5:7], dilation=convp[0][7:9], groups=convp[0][9])
        FD['in1'] = th.nn.InstanceNorm2d(convp[0][0])
        FD['relu1'] = th.nn.LeakyReLU()
        for n in range(1, self.nconvs):
            FD['conv' + str(n + 1)] = th.nn.Conv2d(convp[n - 1][0], convp[n][0], convp[n][1:3], stride=convp[n][3:5],
                                                   padding=convp[n][5:7], dilation=convp[n][7:9], groups=convp[n][9])
            FD['in' + str(n + 1)] = th.nn.InstanceNorm2d(convp[n][0])
            FD['relu' + str(n + 1)] = th.nn.LeakyReLU()
        FD['gapool'] = th.nn.AdaptiveAvgPool2d((1, 1))  # N-1-Na-1

        self.features = th.nn.Sequential(FD)
        self.coefs = th.nn.Sequential(OrderedDict([
            ('fc1', th.nn.Linear(convp[-1][0], 128)),
            ('relu1', th.nn.LeakyReLU()),
            # ('relu1', th.nn.LeakyReLU(0.5)),
            ('drop1', th.nn.Dropout(p=0.5, inplace=True)),
            ('fc2', th.nn.Linear(128, self.Ma)),
            # ('relu2', th.nn.LeakyReLU(0.5)),
        ]))

        if xa is None:
            xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='fftfreq')
        xa = xa.reshape(1, Na)

        xas = th.tensor([])
        for ma in mas:
            xas = th.cat((xas, xa ** ma), axis=0)
        self.xa = Parameter(xa, requires_grad=False)  # 1-Na
        self.xas = Parameter(xas, requires_grad=False)  # Ma-Na

    def forward(self, X):
        d = X.dim()

        # Y = th.stack((X.pow(2).sum(-1).sqrt(), th.atan2(X[..., 1], X[..., 0])), dim=1)
        Y = th.stack((X[..., 0], X[..., 1]), dim=1)
        Y = self.features(Y)

        Y = Y.view(Y.size(0), -1)  # N-Na
        ca = self.coefs(Y)  # N-Ma
        pa = th.matmul(ca, self.xas)  # N-Ma x Ma-Na

        sizea = [1] * d
        sizea[0], sizea[-3], sizea[-2], sizea[-1] = pa.size(0), pa.size(1), 1, 2
        epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        X = ts.fft(X, n=None, axis=-3, norm=None, shift=self.ftshift)
        X = ts.ebemulcc(X, epa)
        X = ts.ifft(X, n=None, axis=-3, norm=None, shift=self.ftshift)

        return X, ca


def weights_init(m):
    if isinstance(m, th.nn.Conv2d):
        th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
        m.bias.data.zero_()
        # print(m.weight.data.shape, m.weight.data[0, 0])
    if isinstance(m, th.nn.Linear):
        th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
        m.bias.data.zero_()
        # print(m.weight.data.shape, m.weight.data[0, 0])


class PAFnet(th.nn.Module):

    def __init__(self, Na, Nr, Mas=[[2, 3]], Mrs=None, lpaf=[[[32, 11, 1]]], lprf=None, xa=None, ftshift=True, seed=None):
        super(PAFnet, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Mas = Mas
        self.ftshift = ftshift
        self.seed = seed
        self.focusers = []

        if self.seed is not None:
            th.manual_seed(seed)

        self.Ma = 0
        for mas, convp in zip(Mas, lpaf):
            focusern = Focuser(Na, Nr, mas, convp, xa, ftshift)
            focusern.apply(weights_init)  # init weight
            self.Ma += len(mas)
            self.focusers.append(focusern)

        self.focusers = th.nn.ModuleList(self.focusers)
        self.Nf = len(self.focusers)

    def forward(self, X, isfft=True):

        cas = th.tensor([], device=X.device)
        for n in range(self.Nf):
            X, ca = self.focusers[n](X)
            cas = th.cat((cas, ca), axis=1)  # N-Ma

        return X, cas

    def train_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, optimizer, scheduler, device):
        self.train()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = ts.randperm(0, numSamples, numSamples)
        lossENTv, lossCTSv, lossFROv, lossvtrain = 0., 0., 0., 0.
        # t1, t2, t3 = 0., 0., 0.
        for b in range(numBatch):
            # tstart1 = time.time()
            i = idx[b * sizeBatch:(b + 1) * sizeBatch]
            xi = X[i]
            xi = xi.to(device)

            optimizer.zero_grad()

            fi, casi = self.forward(xi)
            # tend1 = time.time()

            # tstart2 = time.time()
            lossENT = loss_ent_func(fi)
            lossCTS = loss_cts_func(fi)
            lossFRO = loss_fro_func(fi)
            # tend2 = time.time()

            if loss_type == 'Entropy':
                loss = lossENT
            if loss_type == 'Entropy+LogFro':
                loss = lossENT + lossFRO
            if loss_type == 'Contrast':
                loss = lossCTS
            if loss_type == 'Entropy/Contrast':
                loss = lossENT / lossCTS

            loss.backward()

            # tstart3 = time.time()
            lossvtrain += loss.item()
            lossCTSv += lossCTS.item()
            lossENTv += lossENT.item()
            lossFROv += lossFRO.item()
            # tend3 = time.time()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # t1 += tend1 - tstart1
            # t2 += tend2 - tstart2
            # t3 += tend3 - tstart3

        tend = time.time()

        lossvtrain /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch
        # print(t1, t2, t3)
        print("--->Train epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvtrain, lossENTv, lossFROv, lossCTSv, tend - tstart))
        return lossvtrain

    def valid_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, device):
        self.eval()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossENTv, lossCTSv, lossFROv, lossvvalid = 0., 0., 0., 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi = X[i]
                xi = xi.to(device)

                fi, casi = self.forward(xi)

                lossENT = loss_ent_func(fi)
                lossCTS = loss_cts_func(fi)
                lossFRO = loss_fro_func(fi)

                if loss_type == 'Entropy':
                    loss = lossENT
                if loss_type == 'Entropy+LogFro':
                    loss = lossENT + lossFRO
                if loss_type == 'Contrast':
                    loss = lossCTS
                if loss_type == 'Entropy/Contrast':
                    loss = lossENT / lossCTS

                lossvvalid += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

        tend = time.time()

        lossvvalid /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch

        print("--->Valid epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvvalid, lossENTv, lossFROv, lossCTSv, tend - tstart))

        return lossvvalid

    def test_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, device):
        self.eval()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossENTv, lossCTSv, lossFROv, lossvtest = 0., 0., 0., 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi = X[i]
                xi = xi.to(device)

                fi, casi = self.forward(xi)

                lossENT = loss_ent_func(fi)
                lossCTS = loss_cts_func(fi)
                lossFRO = loss_fro_func(fi)

                if loss_type == 'Entropy':
                    loss = lossENT
                if loss_type == 'Entropy+LogFro':
                    loss = lossENT + lossFRO
                if loss_type == 'Contrast':
                    loss = lossCTS
                if loss_type == 'Entropy/Contrast':
                    loss = lossENT / lossCTS

                lossvtest += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

        tend = time.time()

        lossvtest /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch

        print("--->Test epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvtest, lossENTv, lossFROv, lossCTSv, tend - tstart))

        return lossvtest

    def plot(self, xi, cai, cri, xa, idx, prefixname, outfolder, device):

        self.eval()
        with th.no_grad():
            xi, cai, cri = xi.to(device), cai.to(device), cri.to(device)

            fi, casi = self.forward(xi)

        saveimage(xi, fi, idx, prefixname=prefixname, outfolder=outfolder + '/images/')

        pai = ts.polype(c=cai, x=xa)
        ppai = ts.polype(c=casi, x=xa)
        pai = pai.detach().cpu().numpy()
        ppai = ppai.detach().cpu().numpy()

        for i, ii in zip(range(len(idx)), idx):
            plt.figure()
            plt.plot(pai[i], '-b')
            plt.plot(ppai[i], '-r')
            plt.legend(['Real', 'Estimated'])
            plt.grid()
            plt.xlabel('Aperture time (samples)')
            plt.ylabel('Phase (rad)')
            plt.title('Estimated phase error (polynomial degree ' + str(self.Ma) + ')')
            plt.savefig(outfolder + '/images/' + prefixname + '_phase_error_azimuth' + str(ii) + '.png')
            plt.close()


if __name__ == '__main__':

    seed = 2020
    device = 'cuda:0'
    N, Na, Nr = 4, 25, 26

    th.random.manual_seed(seed)
    X = th.randn(N, Na, Nr, 2)

    net = Focuser(Na, Nr, mas=[2, 3], convp=[[16, 11, 1, 1, 1, 0, 0, 1, 1, 1]])
    net = net.to(device)
    X = X.to(device)
    Y, ca = net.forward(X)

    print(Y.shape)
    # print(ca)

    net = PAFnet(Na, Nr, Mas=[[2, 3], [4, 5], [6, 7]], lpaf=[[[8, 11, 1, 1, 1, 0, 0, 1, 1, 1]], [[16, 7, 1, 1, 1, 0, 0, 1, 1, 1]], [[32, 5, 1, 1, 1, 0, 0, 1, 1, 1]]], seed=2020)
    net = net.to(device)

    # print(net)

    Y, cas = net.forward(X)
    print(Y.shape, cas.shape)
    print(Y[0, 0, 0])
    # print(net.cas)
