import os
import argparse
import torch as th
import torchbox as tb
import torchsar as ts
from ecelms import BaggingECELMs
from dataset import readsamples

isnetplot = False
isnetplot = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./ecelms.yaml')
parser.add_argument('--cstrategy', type=str, default='Entropy', help='Entropy, AverageCoef, AveragePhase')

# params in Adam
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=10)
parser.add_argument('--snapshot_name', type=str, default='2021')

# misc
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(RealPE, SimPoly, SimSin...)')
cfg = parser.parse_args()

seed = cfg.seed
device = cfg.device
size_batch = cfg.size_batch
ftshift = True

cudaTF32, cudnnTF32 = False, False
# benchmark: if False, training is slow, if True, training is fast but conv results may be slight different for different cards.
benchmark, deterministic = True, True

Cs = [1e-2, 1e-1, 1e0, 1e1, 1e2]

datacfg = tb.loadyaml(cfg.datacfg)
modelcfg = tb.loadyaml(cfg.modelcfg)

if 'SAR_AF_DATA_PATH' in os.environ.keys():
    datafolder = os.environ['SAR_AF_DATA_PATH']
else:
    datafolder = datacfg['SAR_AF_DATA_PATH']

for k in range(1, 64, 2):

    modelcfg['Convs'][0][0][1] = k
    print(cfg)
    print(datacfg)
    print(modelcfg)
    nCEMLs = len(modelcfg['Convs'])

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

    Xtrain, catrain, crtrain = readsamples(fileTrain, keys=keys, nsamples=[4000], groups=[25], mode=modeTrain, parts=None, seed=seed)
    Xvalid, cavalid, crvalid = readsamples(fileValid, keys=keys, nsamples=[4000], groups=[25], mode=modeValid, parts=None, seed=seed)
    Xtest, catest, crtest = readsamples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, parts=None, seed=seed)

    Ntrain, Nvalid, Ntest = Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]
    nsamples1 = 3000

    _, Na, Nr, _ = Xtrain.shape
    xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=ppeaxismode)
    xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=ppeaxismode)
    print(xa.min(), xa.max())
    if cfg.mkpetype in ['simpoly', 'SimPoly']:
        print("---Focusing...")
        pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
        Xtrain = ts.focus(Xtrain, pa, None, isfft=True, ftshift=ftshift)
        pa, pr = ts.polype(cavalid, xa), ts.polype(crvalid, xr)
        Xvalid = ts.focus(Xvalid, pa, None, isfft=True, ftshift=ftshift)
        pa, pr = ts.polype(catest, xa), ts.polype(crtest, xr)
        Xtest = ts.focus(Xtest, pa, None, isfft=True, ftshift=ftshift)
        print("---Done.")

        print("---Making polynominal phase error...")
        carange = [[-64, -64, -64, -64, -64, -64], [64, 64, 64, 64, 64, 64]]
        crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]
        print('~~~carange', carange)
        print('~~~crrange', crrange)
        ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)
        catrain, crtrain = ppeg.mkpec(n=Ntrain, seed=None)
        print(catrain.min(), catrain.max())
        pa, pr = ts.polype(catrain, xa), ts.polype(crtrain, xr)
        print(pa.min(), pa.max())
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

    outfolder = './snapshot/' + modelcfg['model'] + '/' + cfg.mkpetype + '/' + cfg.snapshot_name
    checkpoint_path = outfolder + '/weights/'
    os.makedirs(outfolder + '/images', exist_ok=True)
    os.makedirs(outfolder + '/weights', exist_ok=True)

    net = BaggingECELMs(Na, 1, Qas=modelcfg['Qas'], Convs=modelcfg['Convs'], xa=xa, xr=None, cstrategy=cfg.cstrategy, ftshift=ftshift, seed=seed)

    loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(-3, -2), reduction='mean')  # OK
    loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(-3, -2), reduction='mean')  # OK
    loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(-3, -2), reduction='mean')

    print("Orignal Entropy(Train, Valid, Test):", loss_ent_func(Xtrain).item(), loss_ent_func(Xvalid).item(), loss_ent_func(Xtest).item())
    print("Orignal Contrast(Train, Valid, Test):", loss_cts_func(Xtrain).item(), loss_cts_func(Xvalid).item(), loss_cts_func(Xtest).item())

    # if cfg.modelfile is not None:
    #     modelfile = cfg.modelfile
    # else:
    #     modelfile = './snapshot/EHAFnet/AdamW/2020/weights/current.pth.tar'
    # logdict = th.load(modelfile, map_location=device)
    # sepoch = logdict['epoch']

    logdict = {}
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

    print(net)
    net.to(device=device)

    xc = []
    ts.setseed(seed)
    idxtrain = list(range(0, Xtrain.shape[0], int(Xtrain.shape[0] / 16)))
    idxvalid = list(range(0, Xvalid.shape[0], int(Xvalid.shape[0] / 16)))
    idxtest = list(range(0, Xtest.shape[0], int(Xtest.shape[0] / 16)))

    with th.no_grad():
        cs = net.train_valid(Xtrain, catrain, crtrain, Xvalid, cavalid, crvalid, size_batch, nsamples1, loss_ent_func, loss_cts_func, loss_fro_func, Cs, device)

        print("---Best C", cs)

        lossvtrain = net.ensemble_test(Xtrain, catrain, crtrain, size_batch, loss_ent_func, loss_cts_func, loss_fro_func, device, name='Train')
        if isnetplot:
            net.plot(Xtrain[idxtrain], catrain[idxtrain], crtrain[idxtrain], xa, idxtrain, 'train', outfolder, device)

        lossvvalid = net.ensemble_test(Xvalid, cavalid, crvalid, size_batch, loss_ent_func, loss_cts_func, loss_fro_func, device, name='Valid')
        if isnetplot:
            net.plot(Xvalid[idxvalid], cavalid[idxvalid], crvalid[idxvalid], xa, idxvalid, 'valid', outfolder, device)

        lossvtest = net.ensemble_test(Xtest, catest, crtest, size_batch, loss_ent_func, loss_cts_func, loss_fro_func, device, name='Test')
        if isnetplot:
            net.plot(Xtest[idxtest], catest[idxtest], crtest[idxtest], xa, idxtest, 'test', outfolder, device)

        logdict['BalanceFactor'] = cs
        logdict['network'] = net.state_dict()

        th.save(logdict, checkpoint_path + str(k) + '.pth.tar')
