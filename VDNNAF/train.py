import os
import argparse
import torch as th
import torchbox as tb
import torchsar as ts
from model import Net
from utils import sample

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
# th.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# th.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()

parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--model', type=str, default='VDNN')

# params in Adam
parser.add_argument('--modelfile', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=800)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--scheduler', type=str, default='StepLR')
parser.add_argument('--snapshot_name', type=str, default='2021')

# misc
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--mkpetype', type=str, default='SimPoly',
                    help='make phase error(RealPE, SimPoly, SimSin...)')

cfg = parser.parse_args()

# nTrain, nValid, nTest = 6000, 8000, 8000
nTrain, nValid, nTest = 192000, 64000, 64000
size = (64, 64)
seed = cfg.seed
size_batch = cfg.size_batch
ftshift = True

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

# fileTrain = [datafolder + datacfg['filenames'][0]]
# fileValid = [datafolder + datacfg['filenames'][1]]
# fileTest = [datafolder + datacfg['filenames'][0]]

modeTrain, modeValid, modeTest = 'sequentially', 'sequentially', 'sequentially'
print("--->Train files sampling mode:", modeTrain)
print(fileTrain)
print("--->Valid files sampling mode:", modeValid)
print(fileValid)
print("--->Test files sampling mode:", modeTest)
print(fileTest)

keys, ppeaxismode = [['X', 'poly_ca', 'poly_cr']], '2fftfreq'
keys, ppeaxismode = [['SI', 'ca', 'cr']], 'fftfreq'
keys, ppeaxismode = [['SI', 'mea_steplr_bare_pa', 'mea_steplr_bare_pr']], None
Xtrain, catrain, crtrain = tb.read_samples(fileTrain, keys=keys, nsamples=[1200], groups=[25], mode=modeTrain, seed=seed)
Xvalid, cavalid, crvalid = tb.read_samples(fileValid, keys=keys, nsamples=[4000], groups=[25], mode=modeValid, seed=seed)
Xtest, catest, crtest = tb.read_samples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

N, Na, Nr, _ = Xtrain.size()

Ntrain, Nvalid, Ntest = Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=ppeaxismode)
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=ppeaxismode)

print("---Focusing...")
pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
Ftrain = ts.focus(Xtrain, pa, pr, isfft=True, ftshift=ftshift)
pa, pr = ts.polype(cavalid, xa), ts.polype(crvalid, xr)
Fvalid = ts.focus(Xvalid, pa, pr, isfft=True, ftshift=ftshift)
pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
Ftest = ts.focus(Xtest, pa, pr, isfft=True, ftshift=ftshift)
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
    catrain, crtrain = ppeg.mkpec(n=Ntrain, seed=seed + 1)
    pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
    Xtrain = ts.defocus(Ftrain, pa, None, isfft=True, ftshift=ftshift)
    cavalid, crvalid = ppeg.mkpec(n=Nvalid, seed=seed + 2)
    pa, pr = ts.polype(cavalid, xa), ts.polype(crvalid, xr)
    Xvalid = ts.defocus(Fvalid, pa, None, isfft=True, ftshift=ftshift)
    catest, crtest = ppeg.mkpec(n=Ntest, seed=seed + 3)
    pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
    Xtest = ts.defocus(Ftest, pa, None, isfft=True, ftshift=ftshift)
    print("---Making polynominal phase error done.")

Xtrain, Ftrain = Xtrain.unsqueeze(1), Ftrain.unsqueeze(1)
Xvalid, Fvalid = Xvalid.unsqueeze(1), Fvalid.unsqueeze(1)
Xtest, Ftest = Xtest.unsqueeze(1), Ftest.unsqueeze(1)
print("--->Train:", Xtrain.shape, Ftrain.shape, catrain.shape, crtrain.shape)
print("--->Valid:", Xvalid.shape, Fvalid.shape, cavalid.shape, crvalid.shape)
print("--->Test:", Xtest.shape, Ftest.shape, catest.shape, crtest.shape)
del catrain, cavalid, catest, crtrain, crvalid, crtest, pa, pr

Xtrain, Ftrain = th.view_as_complex(Xtrain), th.view_as_complex(Ftrain)
Xvalid, Fvalid = th.view_as_complex(Xvalid), th.view_as_complex(Fvalid)
Xtest, Ftest = th.view_as_complex(Xtest), th.view_as_complex(Ftest)

idxtrain = list(range(0, Xtrain.shape[0], int(Xtrain.shape[0] / 32)))
idxvalid = list(range(0, Xvalid.shape[0], int(Xvalid.shape[0] / 32)))
idxtest = list(range(0, Xtest.shape[0], int(Xtest.shape[0] / 32)))

# =======================
# ---convert to real and sample patch
Xtest, Ftest = tb.ct2rt(Xtest, dim=2).real, tb.ct2rt(Ftest, dim=2).real
XtestPlot, FtestPlot = Xtest[idxtest], Ftest[idxtest]
Xtest, Ftest = sample(Xtest, Ftest, nsamples=nTest, size=size, index=None, seed=seed + 3)

Xvalid, Fvalid = tb.ct2rt(Xvalid, dim=2).real, tb.ct2rt(Fvalid, dim=2).real
XvalidPlot, FvalidPlot = Xvalid[idxvalid], Fvalid[idxvalid]
Xvalid, Fvalid = sample(Xvalid, Fvalid, nsamples=nValid, size=size, index=None, seed=seed + 2)

Xtrain, Ftrain = tb.ct2rt(Xtrain, dim=2).real, tb.ct2rt(Ftrain, dim=2).real
XtrainPlot, FtrainPlot = Xtrain[idxtrain], Ftrain[idxtrain]
Xtrain, Ftrain = sample(Xtrain, Ftrain, nsamples=nTrain, size=size, index=None, seed=seed + 1)

print("--->Train:", Xtrain.shape, Ftrain.shape)
print("--->Valid:", Xvalid.shape, Fvalid.shape)
print("--->Test:", Xtest.shape, Ftest.shape)
numSamples = N

outfolder = './snapshot/' + cfg.model + '/' + cfg.mkpetype + '/' + cfg.optimizer + '/' + cfg.snapshot_name
losslog = tb.LossLog(outfolder, filename='LossEpoch', lom='min')

os.makedirs(outfolder + '/images', exist_ok=True)
os.makedirs(outfolder + '/weights', exist_ok=True)

device = th.device(cfg.device if th.cuda.is_available() else 'cpu')
devicename = 'CPU' if cfg.device == 'cpu' else th.cuda.get_device_name(int(str(device)[-1]))
# device = th.device('cpu')
print(device)

net = Net(in_channels=1, out_channels=1, seed=seed)

checkpoint_path = outfolder + '/weights/'

lossfn = th.nn.MSELoss()

print(lossfn)

logdict = {}
sepoch = 1

last_epoch = sepoch - 2

optimizer = th.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
if cfg.scheduler in ['StepLR', 'steplr']:
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=last_epoch)
else:
    scheduler = None

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
print(optimizer)
print(scheduler)
net.to(device=device)

tb.device_transfer(optimizer, 'optimizer', device=device)

n1, n2 = 50, 10
tb.setseed(seed)
for k in range(sepoch, cfg.num_epochs):

    lossvtrain = net.train_epoch(Xtrain, Ftrain, size_batch, lossfn, k, optimizer, None, device)
    net.plot(XtrainPlot, FtrainPlot, idxtrain, 'train', outfolder, device)

    if cfg.scheduler is not None:
        scheduler.step()

    lossvvalid = net.valid_epoch(Xvalid, Fvalid, size_batch, lossfn, k, device)
    net.plot(XvalidPlot, FvalidPlot, idxvalid, 'valid', outfolder, device)

    lossvtest = net.test_epoch(Xtest, Ftest, size_batch, lossfn, k, device)
    net.plot(XtestPlot, FtestPlot, idxtest, 'test', outfolder, device)

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

    flag, proof = losslog.judge('train', n1=n1, n2=n2)
    if flag:
        th.save(logdict, checkpoint_path + 'Train' + proof + str(k) + '.pth.tar')

    flag, proof = losslog.judge('valid', n1=n1, n2=n2)
    if flag:
        th.save(logdict, checkpoint_path + 'Valid' + proof + str(k) + '.pth.tar')

    flag, proof = losslog.judge('test', n1=n1, n2=n2)
    if flag:
        th.save(logdict, checkpoint_path + 'Test' + proof + str(k) + '.pth.tar')

    th.save(logdict, checkpoint_path + 'current' + '.pth.tar')
