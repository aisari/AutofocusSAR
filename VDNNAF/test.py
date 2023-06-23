import os
import time
import argparse
import torch as th
import torchbox as tb
import torchsar as ts
from model import Net
from utils import saveimage

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--model', type=str, default='VDNN')

# params in Adam
parser.add_argument('--modelfile', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=800)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--scheduler', type=str, default='StepLR')
parser.add_argument('--snapshot_name', type=str, default='2020')

# misc
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--mkpetype', type=str, default='SimPoly',
                    help='make phase error(RealPE, SimPoly, SimSin...)')

cfg = parser.parse_args()

seed = cfg.seed
sizeBatch = cfg.size_batch
ftshift = True
issaveimage = True

cudaTF32, cudnnTF32 = False, False
# benchmark: if False, training is slow, if True, training is fast but
# conv results may be slight different for different cards.
benchmark, deterministic = True, True


datacfg = tb.loadyaml(cfg.datacfg)

if 'SAR_AF_DATA_PATH' in os.environ.keys():
    datafolder = os.environ['SAR_AF_DATA_PATH']
else:
    datafolder = datacfg['SAR_AF_DATA_PATH']

print(cfg)

fileTrain = [datafolder + datacfg['filenames'][i] for i in datacfg['trainid']]
fileValid = [datafolder + datacfg['filenames'][i] for i in datacfg['validid']]
fileTest = [datafolder + datacfg['filenames'][i] for i in datacfg['testid']]

modeTest = 'sequentially'
print("--->Test files sampling mode:", modeTest)
print(fileTest)

modelfile = './record/PE1d/weights/ValidSingleAverage204.pth.tar'

keys, ppeaxismode = [['X', 'poly_ca', 'poly_cr']], '2fftfreq'
keys, ppeaxismode = [['SI', 'ca', 'cr']], 'fftfreq'
keys, ppeaxismode = [['SI', 'mea_steplr_bare_pa', 'mea_steplr_bare_pr']], None
X, ca, cr = tb.read_samples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

N, Na, Nr, _ = X.size()

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=ppeaxismode)
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=ppeaxismode)

print("---Focusing...")
pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
F = ts.focus(X, pa, pr, isfft=True, ftshift=ftshift)
print("---Done.")

if cfg.mkpetype in ['simpoly', 'SimPoly']:
    print("---Making polynominal phase error...")
    xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='2fftfreq')
    xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='2fftfreq')
    carange = [[-16, -16, -16, -16, -16, -16], [16, 16, 16, 16, 16, 16]]
    crrange = [[-8, -8, -8, -8, -8, -8], [8, 8, 8, 8, 8, 8]]
    print('~~~carange', carange)
    print('~~~crrange', crrange)
    ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
    ca, cr = ppeg.mkpec(n=N, seed=seed + 3)
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
    X = ts.defocus(F, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

# # index = list(range(0, 200))
# # index = list(range(7800, 8000))
# # index = list(range(1980, 2031))
# index = [0, 162, 3460, 638, 7884, 2906]
# X, F, ca, cr = X[index], F[index], ca[index], cr[index]

numSamples = N = X.shape[0]

X, F = X.unsqueeze(1), F.unsqueeze(1)
print("--->Test:", X.shape, F.shape, ca.shape, cr.shape)
del ca, cr, pa, pr

X, F = th.view_as_complex(X), th.view_as_complex(F)

outfolder = './snapshot/' + cfg.model + '/' + cfg.mkpetype + '/' + cfg.optimizer + '/' + cfg.snapshot_name + '/tests/'

os.makedirs(outfolder, exist_ok=True)

device = th.device(cfg.device if th.cuda.is_available() else 'cpu')
devicename = 'CPU' if cfg.device == 'cpu' else th.cuda.get_device_name(int(str(device)[-1]))
# device = th.device('cpu')
print(device)

net = Net(in_channels=1, out_channels=1, seed=seed)

logdict = th.load(modelfile, map_location=device)
net.load_state_dict(logdict['network'])

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

loss_ent_func = tb.EntropyLoss('natural', dim=(2, 3), reduction='mean')  # OK
loss_cts_func = tb.NegativeContrastLoss('way1', dim=(2, 3), reduction='mean')  # OK

tb.setseed(seed)

tstart = time.time()

# =======================
# ---convert to real and sample patch
X = tb.ct2rt(X, dim=2).real

print("--->Test:", X.shape, F.shape)
numSamples = X.shape[0]
sizeBatch = min(sizeBatch, numSamples)

numBatch = int(numSamples / sizeBatch)
idx = list(range(numSamples))
Y = th.zeros_like(F)
lossentv, lossctsv = 0., 0.
with th.no_grad():
    for b in range(numBatch):
        i = idx[b * sizeBatch:(b + 1) * sizeBatch]
        xi = X[i]
        xi = xi.to(device)
        yi = net.forward(xi)
        yi = tb.rt2ct(yi, dim=2)
        lossentv += loss_ent_func(yi).item()
        lossctsv += loss_cts_func(yi).item()
        Y[i] = yi.cpu()
X = tb.rt2ct(X, dim=2)

tend = time.time()

lossentv /= numBatch
lossctsv /= numBatch

print("--->Test: entropy: %.4f, contrast: %.4f, time: %ss" % (lossentv, lossctsv, tend - tstart))

if issaveimage:
    X, F, Y = X.squeeze(1), F.squeeze(1), Y.squeeze(1)
    X, F, Y = th.view_as_real(X), th.view_as_real(F), th.view_as_real(Y)
    saveimage(X, F, Y, idx, prefixname='test', outfolder=outfolder)
