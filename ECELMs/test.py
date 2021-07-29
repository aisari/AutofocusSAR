import os
import argparse
import torch as th
import torchsar as ts
from ecelms import BaggingECELMs
from dataset import readsamples


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./ecelms.yaml')
parser.add_argument('--weightfile', type=str, default=None)
parser.add_argument('--cstrategy', type=str, default='Entropy', help='Entropy, AveragePhase')

# params in Adam
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=3)
parser.add_argument('--snapshot_name', type=str, default='2020')

# misc
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(RealPE, SimPoly, SimSin...)')
cfg = parser.parse_args()

isplot = True
isplot = False
ftshift = True
issaveimg = True
issaveimg = False
seed = cfg.seed
device = cfg.device
size_batch = cfg.size_batch
cudaTF32, cudnnTF32 = False, False
benchmark, deterministic = True, True

outfolder = './snapshot/tests/'

if cfg.weightfile is None:
    cfg.weightfile = './record/RealPE/DiffKernelSize/Entropy/weights/64CELMs.pth.tar'


datacfg = ts.loadyaml(cfg.datacfg)
modelcfg = ts.loadyaml(cfg.modelcfg)

if 'SAR_AF_DATA_PATH' in os.environ.keys():
    datafolder = os.environ['SAR_AF_DATA_PATH']
else:
    datafolder = datacfg['SAR_AF_DATA_PATH']

print(datacfg)
print(modelcfg)
nCELMs = len(modelcfg['Convs'])
# for i in range(nCEMLs):
#     modelcfg['Convs'][i][0][1] = 3
#     modelcfg['Convs'][i][0][1] = 17

fileTrain = [datafolder + datacfg['filenames'][i] for i in datacfg['trainid']]
fileValid = [datafolder + datacfg['filenames'][i] for i in datacfg['validid']]
fileTest = [datafolder + datacfg['filenames'][i] for i in datacfg['testid']]

modeTest = 'sequentially'
print("--->Test files sampling mode:", modeTest)
print(fileTest)

keys, ppeaxismode = [['SI', 'ca', 'cr']], 'fftfreq'

parts = [1. / nCELMs] * nCELMs
X, ca, cr = readsamples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, parts=None, seed=seed)

N, Na, Nr, _ = X.size()
numSamples = N

os.makedirs(outfolder + '/images/', exist_ok=True)

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=ppeaxismode)
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=ppeaxismode)

F = X
if cfg.mkpetype in ['simpoly', 'SimPoly']:
    print("---Focusing...")
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
    F = ts.focus(X, pa, None, isfft=True, ftshift=ftshift)
    print("---Done.")

    print("---Making polynominal phase error...")
    carange = [[-32, -32, -32, -32, -32, -32], [32, 32, 32, 32, 32, 32]]
    # carange = [[-64, -64, -64, -64, -64, -64], [64, 64, 64, 64, 64, 64]]
    # carange = [[-128, -128, -128, -128, -128, -128], [128, 128, 128, 128, 128, 128]]
    crrange = [[-32, -32, -32, -32, -32, -32], [32, 32, 32, 32, 32, 32]]
    # crrange = [[-64, -64, -64, -64, -64, -64], [64, 64, 64, 64, 64, 64]]
    # crrange = [[-128, -128, -128, -128, -128, -128], [128, 128, 128, 128, 128, 128]]
    print('~~~carange', carange)
    print('~~~crrange', crrange)
    ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
    ppeg.mkpec(n=6000, seed=None)  # train
    ppeg.mkpec(n=8000, seed=None)  # valid

    ca, cr = ppeg.mkpec(n=N, seed=None)
    pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)
    X = ts.defocus(F, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

# index = list(range(0, 200))
# # index = list(range(7800, 8000))
# # index = list(range(1980, 2031))
index = [89, 1994, 7884]
# index = [0, 1, 4, 19, 21, 51, 93, 140, 156, 162, 250, 2000, 1999, 7835, 7881, 7887]
X, F, ca, cr = X[index], F[index], ca[index], cr[index]

numSamples = N = X.shape[0]
N = X.shape[0]
size_batch = min(cfg.size_batch, N)

# device = th.device(cfg.device if th.cuda.is_available() else 'cpu')
devicename = 'E5 2696v3' if device == 'cpu' else th.cuda.get_device_name(int(str(device)[-1]))
print(device)
print(devicename)
print("Torch Version: ", th.__version__)
print("Torch CUDA Version: ", th.version.cuda)
print("CUDNN Version: ", th.backends.cudnn.version())
print("CUDA TF32: ", cudaTF32)
print("CUDNN TF32: ", cudnnTF32)
print("CUDNN Benchmark: ", benchmark)
print("CUDNN Deterministic: ", deterministic)

th.backends.cuda.matmul.allow_tf32 = cudaTF32
th.backends.cudnn.allow_tf32 = cudnnTF32
th.backends.cudnn.benchmark = benchmark
th.backends.cudnn.deterministic = deterministic

print("--->Orders for azimuth: ", modelcfg['Qas'])
print("--->Orders for range: ", modelcfg['Qrs'])
print("--->Convolution params: ", modelcfg['Convs'])

net = BaggingECELMs(Na, 1, Qas=modelcfg['Qas'], Convs=modelcfg['Convs'], xa=xa, ftshift=ftshift, seed=seed)

modelparamsaf = th.load(cfg.weightfile, map_location=device)
# print(modelparamsaf['network'].keys())
net.load_state_dict(modelparamsaf['network'])

net.to(device=device)
net.eval()

xa = ts.fftfreq(Na, Na, norm=True, shift=True).reshape(1, Na)

loss_mse_fn = th.nn.MSELoss(reduction='mean')
loss_ent_fn = ts.EntropyLoss('natural', reduction='mean')  # OK
loss_cts_fn = ts.ContrastLoss('way1', reduction='mean')  # OK
loss_fro_fn = ts.FrobeniusLoss(reduction='mean', p=1)
# loss_fro_fn = ts.LogFrobenius(reduction='mean', p=1)
loss_tv_fn = ts.TotalVariation(reduction='mean', axis=(2, 3))


lossvtest = net.ensemble_test(X, ca, cr, size_batch, loss_ent_fn, loss_cts_fn, loss_fro_fn, device, name='Test')
net.plot(X, ca, cr, xa, index, 'test', outfolder, device)
