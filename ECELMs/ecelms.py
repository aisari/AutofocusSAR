#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

# from __future__ import print_fntion

import time
import torch as th
import torchbox as tb
import torchsar as ts
from torch.nn.parameter import Parameter
from collections import OrderedDict
from dataset import saveimage
import matplotlib.pyplot as plt


def focus(X, xas, ca=None, pa=None, isfft=True, ftshift=True):
    d = X.dim()
    if pa is None:
        if ca is not None:
            pa = th.matmul(ca, xas.to(ca.device))  # N-Qa x Qa-Na
        else:
            raise ValueError('---You should specify ca or pa!')

    sizea = [1] * d
    sizea[0], sizea[-3], sizea[-2], sizea[-1] = pa.size(0), pa.size(1), 1, 2
    epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
    epa = epa.reshape(sizea)

    if isfft:
        X = tb.fft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=ftshift)
    X = tb.ematmul(X, epa, cdim=-1)
    X = tb.ifft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=ftshift)

    return X


class CELM(th.nn.Module):

    def __init__(self, Na, Nr, Qas, Conv=[[16, 17, 1]], ftshift=True):
        r"""CELM

        CELM

        Parameters
        ----------
        Na : {number}
            [description]
        Nr : {number}
            1 or the number of range cells
        Qas : {list}
            [description]
        ftshift : {bool}, optional
            [description] (the default is True, which [default_description])
        """

        super(CELM, self).__init__()

        self.Na = Na
        self.Nr = Nr
        self.Qas = Qas
        self.ftshift = ftshift
        self.NQas = len(Qas) if Qas is not None else 0
        self.nconvs = len(Conv) if Conv is not None else 0
        self.negslope = 0.01
        self.L = Conv[0][0] * (256 - Conv[0][1] + 1)
        self.BETA = Parameter(th.zeros([self.L, self.NQas]), requires_grad=False)

        FD = OrderedDict()
        FD['conv1'] = th.nn.Conv2d(2, Conv[0][0], Conv[0][1:3], stride=Conv[0][3:5], padding=Conv[0][5:7], dilation=Conv[0][7:9], groups=Conv[0][9])
        FD['in1'] = th.nn.InstanceNorm2d(Conv[0][0])
        FD['relu1'] = th.nn.LeakyReLU(self.negslope)

        for n in range(1, self.nconvs):
            FD['conv' + str(n + 1)] = th.nn.Conv2d(Conv[n - 1][0], Conv[n][0], Conv[n][1:3], stride=Conv[n][3:5],
                                                   padding=Conv[n][5:7], dilation=Conv[n][7:9], groups=Conv[n][9])
            FD['in' + str(n + 1)] = th.nn.InstanceNorm2d(Conv[n][0])
            FD['relu' + str(n + 1)] = th.nn.LeakyReLU(self.negslope)
        FD['gapool'] = th.nn.AdaptiveAvgPool2d((None, 1))  # N-1-Na-1

        self.features = th.nn.Sequential(FD)

    def forward_feature(self, X):
        H = th.stack((X[..., 0], X[..., 1]), dim=1)

        H = self.features(H)
        H = H.view(H.size(0), -1)  # N-Na

        return H

    def forward_predict(self, H, BETA=None):
        if BETA is None:
            return th.matmul(H, self.BETA)  # ca/pa
        else:
            return th.matmul(H, BETA)  # ca/pa

    def optimize(self, H, T, C, device=None, assign=True):
        if device is not None:
            H = H.to(device)
        device = H.device
        T = T.to(device)
        N, L = H.shape
        if N <= L:
            IC = th.eye(N, device=device) / C
            BETA = (H.t()).mm(th.inverse(IC + H.mm(H.t()))).mm(T)
        else:
            IC = th.eye(L, device=device) / C
            BETA = (th.inverse(IC + (H.t()).mm(H))).mm(H.t()).mm(T)

        BETA = Parameter(BETA, requires_grad=False)
        if assign:
            self.BETA = BETA
        return BETA


def weights_init(m):
    if isinstance(m, th.nn.Conv2d):
        # th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu', 0.5))
        th.nn.init.normal_(m.weight.data)
        No, Ni, Hk, Wk = m.weight.data.shape
        if Hk * Wk < No * Ni:
            m.weight.data = tb.orth(m.weight.data.reshape(No * Ni, Hk * Wk)).reshape(No, Ni, Hk, Wk)
        else:
            m.weight.data = tb.orth(m.weight.data.reshape(No * Ni, Hk * Wk).t()).t().reshape(No, Ni, Hk, Wk)
        m.bias.data.zero_()
    if isinstance(m, th.nn.Linear):
        th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu', 0.5))
        m.bias.data.zero_()


class BaggingECELMs(th.nn.Module):

    def __init__(self, Na, Nr, Qas=[[2, 3]], Qrs=None, Convs=[[[32, 11, 1]]], xa=None, xr=None, ftshift=True, cstrategy='Entropy', seed=None):
        super(BaggingECELMs, self).__init__()
        self.Na = Na
        self.Nr = Nr
        self.Qas = Qas
        self.Qrs = Qrs
        self.Convs = Convs
        self.xa = xa
        self.xr = xr
        self.ftshift = ftshift
        self.cstrategy = cstrategy.lower()
        self.seed = seed
        self.NQas = len(Qas) if Qas is not None else 0
        self.NQrs = len(Qrs) if Qrs is not None else 0
        self.Ncelms = len(Convs) if Convs is not None else 0
        self.celms = []

        if self.seed is not None:
            th.manual_seed(seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = True

        if Qas is not None:
            if xa is None:
                xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='fftfreq')
            xa = xa.reshape(1, Na)
            xas = th.tensor([])
            for q in Qas:
                xas = th.cat((xas, xa ** q), axis=0)
            self.xa = Parameter(xa, requires_grad=False)  # 1-Na
            self.xas = Parameter(xas, requires_grad=False)  # NQas-Na

        if Qrs is not None:
            if xr is None:
                xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='fftfreq')
            xr = xr.reshape(1, Nr)
            xrs = th.tensor([])
            for q in Qrs:
                xrs = th.cat((xrs, xr ** q), axis=0)
            self.xr = Parameter(xr, requires_grad=False)  # 1-Nr
            self.xrs = Parameter(xrs, requires_grad=False)  # NQrs-Nr

        for Conv in Convs:
            celmn = CELM(Na, Nr, Qas, Conv, ftshift)
            celmn.apply(weights_init)  # init weight
            self.celms.append(celmn)

        self.celms = th.nn.ModuleList(self.celms)
        self.combine_metric = tb.Entropy('natural', cdim=-1, dim=(-3, -2), keepcdim=True, reduction=None)
        self.loss_mse_fn = th.nn.MSELoss(reduction='mean')

    def forwardn(self, n, X, isfft=True):

        H = self.celms[n].forward_feature(X)
        pca = self.celms[n].forward_predict(H)
        return focus(X, self.xas, ca=pca, pa=None, isfft=isfft, ftshift=self.celms[n].ftshift), pca

    def ensemble_forward(self, X, isfft=True, cstrategy=None):
        if cstrategy is not None:
            self.cstrategy = cstrategy

        if self.cstrategy == 'entropy':
            Smin = th.ones((X.shape[0],), device=X.device) * 1e32
            Y = th.zeros_like(X)
            ca = th.zeros((X.shape[0], self.celms[0].NQas), device=X.device)
            xa = self.xas.to(X.device)
            for n in range(self.Ncelms):
                H = self.celms[n].forward_feature(X)
                pca = self.celms[n].forward_predict(H)
                Z = focus(X, xa, pca, isfft=isfft, ftshift=self.celms[n].ftshift)
                S = self.combine_metric(Z).squeeze(-1)

                idx = S < Smin
                Y[idx] = Z[idx]
                ca[idx] = pca[idx]
                Smin[idx] = S[idx]
            return Y, ca
        if self.cstrategy == 'averagecoef':
            ca = 0.
            xa = self.xas.to(X.device)
            for n in range(self.Ncelms):
                H = self.celms[n].forward_feature(X)
                pca = self.celms[n].forward_predict(H)
                ca += pca
            ca /= self.Ncelms
            Y = focus(X, xa, ca=ca, pa=None, isfft=isfft, ftshift=self.celms[n].ftshift)
            return Y, ca

        if self.cstrategy == 'averagephase':
            pa = 0.
            xa = self.xas.to(X.device)
            for n in range(self.Ncelms):
                H = self.celms[n].forward_feature(X)
                pca = self.celms[n].forward_predict(H)
                pa += th.matmul(pca, xa)
            pa /= self.Ncelms
            Y = focus(X, xa, ca=None, pa=pa, isfft=isfft, ftshift=self.celms[n].ftshift)
            return Y, pa

    def train_valid(self, Xtrain, catrain, crtrain, Xvalid, cavalid, crvalid, sizeBatch, nsamples1, loss_ent_fn, loss_cts_fn, loss_fro_fn, Cs, device):

        tstart = time.time()
        bestC = [0.] * self.Ncelms
        with th.no_grad():
            for n in range(self.Ncelms):
                idxt = list(th.randint(Xtrain.shape[0], [nsamples1]))
                Xt = Xtrain[idxt]

                cat = catrain[idxt]

                Ns = Xt.shape[0]
                numBatch = int(Ns / sizeBatch) if Ns % sizeBatch == 0 else int(Ns / sizeBatch) + 1
                idx = tb.randperm(0, Ns, Ns)
                X, ca = Xt[idx], cat[idx]
                # print(X.shape, ca.shape, cr.shape, numBatch, sizeBatch, Ns)
                bestMetric, bestBETA = 1e32, 0
                for C in Cs:
                    H = th.tensor([], device=device)
                    T = th.tensor([], device=device)
                    self.train()
                    self.celms[n].train()
                    lossENTv, lossCTSv, lossFROv, lossvtrain = 0., 0., 0., 0.
                    for b in range(numBatch):
                        i = range(b * sizeBatch, (b + 1) * sizeBatch)
                        xi, cani = X[i], ca[i]
                        xi, cani = xi.to(device), cani.to(device)

                        hi = self.celms[n].forward_feature(xi)

                        H = th.cat((H, hi), axis=0)
                        T = th.cat((T, cani), axis=0)
                    BETAc = self.celms[n].optimize(H, T, C, device=device, assign=True)
                    pcan = self.celms[n].forward_predict(H)

                    Y = focus(X.to('cpu'), self.xas, pcan.to('cpu'))

                    lossENT = loss_ent_fn(Y)
                    lossCTS = loss_cts_fn(Y)
                    lossFRO = loss_fro_fn(Y)
                    loss = self.loss_mse_fn(pcan, T)

                    lossvtrain = loss.item()
                    lossCTSv = lossCTS.item()
                    lossENTv = lossENT.item()
                    lossFROv = lossFRO.item()

                    tend = time.time()

                    print("--->Train focuser: %d, C: %12.6f, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
                          (n, C, lossvtrain, lossENTv, lossFROv, lossCTSv, tend - tstart))

                    metricnc = self.validnc(n, C, Xvalid, cavalid, crvalid, sizeBatch, loss_ent_fn, loss_cts_fn, loss_fro_fn, device, name='Valid focuser')
                    # print('===', n, C, metricnc)
                    if metricnc < bestMetric:
                        bestC[n] = C
                        bestBETA = BETAc.clone()
                        bestMetric = metricnc

                self.celms[n].BETA.data = bestBETA.clone()

        return bestC

    def validnc(self, n, C, X, ca, cr, sizeBatch, loss_ent_fn, loss_cts_fn, loss_fro_fn, device, name='Valid focuser'):
        self.eval()
        self.celms[n].eval()
        N, Na, Nr, _ = X.shape

        tstart = time.time()
        numSamples = X.shape[0]
        numBatch = int(numSamples / sizeBatch) if numSamples % sizeBatch == 0 else int(numSamples / sizeBatch) + 1
        # idx = tb.randperm(0, numSamples, numSamples)
        idx = list(range(0, numSamples))
        X, ca = X[idx], ca[idx]
        with th.no_grad():
            lossENTv, lossCTSv, lossFROv, lossvvalid = 0., 0., 0., 0.
            for b in range(numBatch):
                i = range(b * sizeBatch, (b + 1) * sizeBatch)
                xi, cai = X[i], ca[i]
                xi, cai = xi.to(device), cai.to(device)

                yi, pcai = self.forwardn(n, xi, isfft=True)

                lossENT = loss_ent_fn(yi)
                lossCTS = loss_cts_fn(yi)
                lossFRO = loss_fro_fn(yi)
                loss = self.loss_mse_fn(pcai, cai)

                lossvvalid += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

            lossvvalid /= numBatch
            lossCTSv /= numBatch
            lossENTv /= numBatch
            lossFROv /= numBatch

            tend = time.time()

            print("--->" + name + ": %d, C: %12.6f, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
                  (n, C, lossvvalid, lossENTv, lossFROv, lossCTSv, tend - tstart))
        return lossENTv

    def ensemble_test(self, X, ca, cr, sizeBatch, loss_ent_fn, loss_cts_fn, loss_fro_fn, device, name='Test'):
        self.eval()
        if self.cstrategy == 'averagephase':
            ca = th.matmul(ca, self.xas.to(ca.device)) if ca is not None else None
            # cr = th.matmul(cr, self.xrs.to(cr.device)) if cr is not None else None

        N, Na, Nr, _ = X.shape
        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch) if numSamples % sizeBatch == 0 else int(numSamples / sizeBatch) + 1
        # idx = tb.randperm(0, numSamples, numSamples)
        idx = list(range(0, numSamples))
        X, ca = X[idx], ca[idx]
        with th.no_grad():
            lossENTv, lossCTSv, lossFROv, lossvtest = 0., 0., 0., 0.
            for b in range(numBatch):
                i = range(b * sizeBatch, (b + 1) * sizeBatch)
                xi, cai = X[i], ca[i]
                xi, cai = xi.to(device), cai.to(device)

                yi, pcai = self.ensemble_forward(xi, isfft=True)

                lossENT = loss_ent_fn(yi)
                lossCTS = loss_cts_fn(yi)
                lossFRO = loss_fro_fn(yi)

                loss = self.loss_mse_fn(pcai, cai)

                lossvtest += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

            lossvtest /= numBatch
            lossCTSv /= numBatch
            lossENTv /= numBatch
            lossFROv /= numBatch

            tend = time.time()

            print("--->" + name + " ensemble: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
                  (self.Ncelms, lossvtest, lossENTv, lossFROv, lossCTSv, tend - tstart))
        return lossvtest

    def plot(self, xi, cai, cri, xa, idx, prefixname, outfolder, device):

        self.eval()
        with th.no_grad():
            xi, cai = xi.to(device), cai.to(device)
            fi, casi = self.ensemble_forward(xi)

        saveimage(xi, fi, idx, prefixname=prefixname, outfolder=outfolder + '/images/')

        pai = ts.polype(c=cai, x=xa)
        if self.cstrategy == 'averagephase':
            ppai = casi
        else:
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
            plt.title('Estimated phase error (polynomial)')
            plt.savefig(outfolder + '/images/' + prefixname + '_phase_error_azimuth' + str(ii) + '.png')
            plt.close()


if __name__ == '__main__':

    seed = 2020
    device = 'cuda:0'
    N, Na, Nr = 4, 25, 26

    th.random.manual_seed(seed)
    X = th.randn(N, Na, Nr, 2)

    net = CELM(Na, Nr, Qas=[2, 3], Conv=[[16, 11, 1, 1, 1, 0, 0, 1, 1, 1]])
    net = net.to(device)
    X = X.to(device)
    H = net.forward_feature(X)

    print(H.shape)
    # print(ca)

    net = BaggingECELMs(Na, Nr, Qas=[2, 3, 4, 5, 6, 7], Convs=[[[8, 11, 1, 1, 1, 0, 0, 1, 1, 1]], [[16, 7, 1, 1, 1, 0, 0, 1, 1, 1]], [[32, 5, 1, 1, 1, 0, 0, 1, 1, 1]]], seed=2020)
    net = net.to(device)


