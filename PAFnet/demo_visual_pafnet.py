import os
import time
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

keys = [['SI', 'ca', 'cr']]
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

index = [89, 1994, 7884]

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

loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')  # OK
loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')  # OK
loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(-3, -2), keepcdim=True, reduction='mean')

tstart = time.time()

# net.visual_epoch(X, size_batch, loss_ent_func, loss_cts_func, device=device)
net.visual_features(X, idx=index, device=device)


tend = time.time()


