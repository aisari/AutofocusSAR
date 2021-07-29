import os
import time
import argparse
import torchsar as ts
import torch as th
from pafnet import PAFnet
from dataset import readsamples, saveimage
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./pafnet.yaml')
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
parser.add_argument('--axismode', type=str, default='2fftfreq')

cfg = parser.parse_args()

device = cfg.device
# device = th.device('cpu')

seed = 2020
sizeBatch = cfg.size_batch
isplot = True
isplot = False
issaveimg = True
issaveimg = False
ftshift = True


datacfg = ts.loadyaml(cfg.datacfg)
modelcfg = ts.loadyaml(cfg.modelcfg)
solvercfg = ts.loadyaml(cfg.solvercfg)
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

keys = [['X', 'poly_ca', 'poly_cr']]

Xtest, catest, crtest = readsamples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

Ntest, Na, Nr, _ = Xtest.shape

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='2fftfreq')
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='2fftfreq')

if cfg.mkpetype in ['simpoly', 'SimPoly']:
    print("---Making polynominal phase error...")
    pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
    Xtest = ts.focus(Xtest, pa, None, isfft=True, ftshift=ftshift)

    carange = [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]]
    crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]
    print('~~~carange', carange)
    print('~~~crrange', crrange)
    ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
    catrain, crtrain = ppeg.mkpec(n=6000, seed=None)
    cavalid, crvalid = ppeg.mkpec(n=8000, seed=None)
    catest, crtest = ppeg.mkpec(n=Ntest, seed=None)
    pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
    Xtest = ts.defocus(Xtest, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

print("--->Test:", Xtest.shape, catest.shape, crtest.shape)
numSamples = Ntest

outfolder, weight = './snapshot/' + cfg.model + '/' + cfg.optimizer + '/' + cfg.snapshot_name, '/weights/current.pth.tar'

if cfg.modelfile is not None:
    modelfile = cfg.modelfile
else:
    modelfile = outfolder + weight

os.makedirs(outfolder + '/tests', exist_ok=True)

print(device)


if cfg.model in ['PAFnet', 'PAFnet']:
    net = PAFnet(Na, 1, Mas=modelcfg['Mas'], lpaf=modelcfg['lpaf'], ftshift=ftshift, seed=seed)

modelparams = th.load(modelfile, map_location=device)
# print(modelparams)
net.load_state_dict(modelparams['network'])
print(net)

net.to(device=device)
net.eval()

loss_mse_func = th.nn.MSELoss(reduction='mean')
loss_ent_func = ts.EntropyLoss('natural', reduction='mean')  # OK
loss_cts_func = ts.ContrastLoss('way1', reduction='mean')  # OK
loss_fro_func = ts.FrobeniusLoss(reduction='mean', p=1)
# loss_fro_func = ts.LogFrobenius(reduction='mean', p=1)
loss_tv_func = ts.TotalVariation(reduction='mean', axis=(2, 3))


tstart = time.time()

lossMSEv, lossENTv, lossCTSv, lossFROv, lossvtest = 0., 0., 0., 0., 0.
numBatch = int(numSamples / sizeBatch)
idx = list(range(numSamples))
for b in range(numBatch):
    i = idx[b * sizeBatch:(b + 1) * sizeBatch]
    xi, cai, cri = Xtest[i], catest[i], crtest[i]

    with th.no_grad():

        xi, cai, cri = xi.to(device), cai.to(device), cri.to(device)
        # print(cai, "cai")
        fi, casi = net.forward(xi)
        pai = ts.polype(cai, x=xa)
        pri = ts.polype(cri, x=xa)
        # fi, pa, pr = ts.focus(xi, pai, pri)

        # fi = ts.toimage(ts.absc(fi))
        lossENT = loss_ent_func(fi)
        lossCTS = loss_cts_func(fi)
        lossFRO = loss_fro_func(fi)
        # print(X.min(), X.max(), T.min(), T.max())

        # lossMSE = lossMSEa + lossMSEr
        # loss = lossMSE + lossENT
        # loss = lossENT + lossCTS

        # if k < 1000:
        #     loss = lossMSEa
        # else:
        #     loss = lossENT + lossFRO
        loss = lossENT
        # loss = lossMSEa + lossMSEr + lossENT

        lossvtest += loss.item()
        lossCTSv += lossCTS.item()
        lossENTv += lossENT.item()
        lossFROv += lossFRO.item()

    if issaveimg:
        saveimage(xi, fi, i, prefixname='test', outfolder=outfolder + '/tests/')

    if isplot:
        pai = ts.polype(c=cai, x=xa)
        print(casi.shape)
        ppai = ts.polype(c=casi, x=xa)
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

print(numBatch, sizeBatch, numSamples)
print("--->Test: loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
      (lossvtest, lossENTv, lossFROv, lossCTSv, tend - tstart))
