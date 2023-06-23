import os
import time
import argparse
import torch as th
import torchsar as ts
import torchbox as tb
from pafnet import PAFnet
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelafcfg', type=str, default='./afnet.yaml')
parser.add_argument('--modelpafcfg', type=str, default='./pafnet.yaml')
parser.add_argument('--solvercfg', type=str, default='./solver.yaml')

parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(SimPoly, SimSin...)')
parser.add_argument('--axismode', type=str, default='fftfreq')

cfg = parser.parse_args()

device = cfg.device
# device = th.device('cpu')

seed = 2020
isplot = True
# isplot = False
figsize = (3.7, 4)
fontsize = 12
fonttype = 'Times New Roman'
fontdict = {'family': fonttype, 'size': fontsize}

issaveimg = True
# issaveimg = False
ftshift = True
degpga, degmea = 7, 7
# degpga, degmea = 7, None
# degpga, degmea = None, None
outfolder = './snapshot/'

modelafcfg = tb.loadyaml(cfg.modelafcfg)
modelpafcfg = tb.loadyaml(cfg.modelpafcfg)

lpafaf, Masaf, Mrsaf = modelafcfg['lpaf'], modelafcfg['Mas'], None  # 52s
lpafhaf, Mashaf, Mrshaf = modelpafcfg['lpaf'], modelpafcfg['Mas'], None

modelfileaf = './record/RealPE/AFnet/AdamW/GaussianLR_nsamples6000/weights/best_valid_664.pth.tar'
modelfilehaf = './record/RealPE/PAFnet/AdamW/GaussianLR_6000Samples/weights/best_valid_655.pth.tar'

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

keys = [['SI', 'ca', 'cr']]
X, Ca, Cr = ts.read_samples(fileTest, keys=keys, nsamples=[4000], groups=[25], mode=modeTest, seed=seed)

N, Na, Nr, _ = X.shape

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode=cfg.axismode)
xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode=cfg.axismode)


N, Na, Nr, _ = X.size()
numSamples = N

os.makedirs(outfolder + '/tests', exist_ok=True)

print(device)

netaf = PAFnet(Na, 1, Mas=Masaf, lpaf=lpafaf, ftshift=ftshift, seed=seed)
nethaf = PAFnet(Na, 1, Mas=Mashaf, lpaf=lpafhaf, ftshift=ftshift, seed=seed)

modelparamsaf = th.load(modelfileaf, map_location=device)
modelparamshaf = th.load(modelfilehaf, map_location=device)
# print(modelparams)
netaf.load_state_dict(modelparamsaf['network'])
nethaf.load_state_dict(modelparamshaf['network'])

netaf.to(device=device)
netaf.eval()
nethaf.to(device=device)
nethaf.eval()

xa = tb.fftfreq(Na, Na, norm=True, shift=True).reshape(1, Na)

loss_mse_func = th.nn.MSELoss(reduction='mean')
loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(1, 2), reduction='mean')  # OK
loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(1, 2), reduction='mean')  # OK
loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(1, 2), reduction='mean')

# X, Ca, Cr = X[80:101], Ca[80:101], Cr[80:101]
# X, Ca, Cr = X[7880:7901], Ca[7880:7901], Cr[7880:7901]
# X, Ca, Cr = X[1994:2031], Ca[1994:2031], Cr[1994:2031]

# X, Ca, Cr = X[2998:3002], Ca[2998:3002], Cr[2998:3002]
# X, Ca, Cr = X[4498:4502], Ca[4498:4502], Cr[4498:4502]
# X, Ca, Cr = X[3490:3510], Ca[3490:3510], Cr[3490:3510]

idx = [89, 1994, 7884]
X, Ca, Cr = X[idx], Ca[idx], Cr[idx]
N = X.shape[0]

tstart = time.time()
for n in range(N):
    with th.no_grad():
        x, ca, cr = X[[n]], Ca[[n]], Cr[[n]]
        x, ca, cr = x.to(device), ca.to(device), cr.to(device)
        # print(cai, "cai")
        faf, caaf = netaf.forward(x)
        fhaf, cahaf = nethaf.forward(x)

        fpga, papga = ts.pgaf_sm(th.view_as_complex(x), 6785, nsub=None, windb=None, est='LUMV', deg=degpga, niter=20, tol=1.e-6, isplot=False, islog=False)
        fnmea, panmea = ts.meaf_sm(th.view_as_complex(x), phi=None, niter=200, tol=1e-4, eta=1., method='N-MEA', selscat=False, isunwrap=True, deg=degmea, ftshift=ftshift, islog=False)
        fpga = th.view_as_real(fpga)
        fnmea = th.view_as_real(fnmea)

        pa = ts.polype(ca, x=xa)
        paaf = ts.polype(caaf, x=xa)
        pahaf = ts.polype(cahaf, x=xa)

        lossENT, lossCTS = loss_ent_func(faf), loss_cts_func(faf)
        print('AFnet', lossENT, lossCTS)
        lossENT, lossCTS = loss_ent_func(fhaf), loss_cts_func(fhaf)
        print('PAFnet', lossENT, lossCTS)
        lossENT, lossCTS = loss_ent_func(fpga), loss_cts_func(fpga)
        print('PGA', lossENT, lossCTS)
        lossENT, lossCTS = loss_ent_func(fnmea), loss_cts_func(fnmea)
        print('NMEA', lossENT, lossCTS)

    if issaveimg:
        prefixname = 'test'
        if x.dim() == 3:
            x = x.unsqueeze(0)
            faf = faf.unsqueeze(0)
            fhaf = fhaf.unsqueeze(0)
            fpga = fpga.unsqueeze(0)
            fnmea = fnmea.unsqueeze(0)

        x = x.pow(2).sum(-1).sqrt()
        faf = faf.pow(2).sum(-1).sqrt()
        fhaf = fhaf.pow(2).sum(-1).sqrt()
        fpga = fpga.pow(2).sum(-1).sqrt()
        fnmea = fnmea.pow(2).sum(-1).sqrt()

        x, faf, fhaf = tb.mapping(x), tb.mapping(faf), tb.mapping(fhaf)
        fpga, fnmea = tb.mapping(fpga), tb.mapping(fnmea)
        x = x.cpu().detach().numpy()
        faf = faf.cpu().detach().numpy()
        fhaf = fhaf.cpu().detach().numpy()
        fpga = fpga.cpu().detach().numpy()
        fnmea = fnmea.cpu().detach().numpy()

        outfileX = outfolder + '/tests/' + prefixname + '_unfocused' + str(n) + '.tif'
        outfileFaf = outfolder + '/tests/' + prefixname + '_focused_afnet' + str(n) + '.tif'
        outfileFhaf = outfolder + '/tests/' + prefixname + '_focused_hafnet' + str(n) + '.tif'
        outfileFpga = outfolder + '/tests/' + prefixname + '_focused_pga' + str(n) + '.tif'
        outfileFnmea = outfolder + '/tests/' + prefixname + '_focused_fnmea' + str(n) + '.tif'
        tb.imsave(outfileX, x)
        tb.imsave(outfileFaf, faf)
        tb.imsave(outfileFpga, fpga)
        tb.imsave(outfileFnmea, fnmea)

    if isplot:

        pa = pa.detach().cpu().numpy()
        paaf = paaf.detach().cpu().numpy()
        pahaf = pahaf.detach().cpu().numpy()
        papga = papga.detach().cpu().numpy()
        panmea = panmea.detach().cpu().numpy()
        plt.figure(figsize=figsize)
        plt.plot(papga[0], ':k')
        plt.plot(pa[0], '-r')
        plt.plot(panmea[0], '-m', linewidth=2)
        plt.plot(paaf[0], '-.b')
        plt.plot(pahaf[0], '--g')
        plt.legend(['PGA', 'MEA', 'N-MEA', 'AFnet', 'PAFnet'])
        plt.grid()
        plt.xlabel('Aperture position / samples', fontdict=fontdict)
        plt.ylabel('Phase / rad', fontdict=fontdict)
        plt.xticks(fontproperties=fonttype, size=fontsize)
        plt.yticks(fontproperties=fonttype, size=fontsize)
        # plt.title('Estimated phase error')
        plt.subplots_adjust(left=0.16, bottom=0.12, right=0.995, top=0.94, wspace=0, hspace=0)
        plt.savefig(outfolder + '/tests/phase_error_azimuth' + str(n) + '.pdf')
        plt.show()
        plt.close()


tend = time.time()
