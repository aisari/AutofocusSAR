
import torch as th
import torchbox as tb
import torchsar as ts
from pafnet import PAFnet


datafile = '/mnt/e/DataSets/zhi/sar/alos/autofocus/ALPSRP020160970_L3600_T10000_25x8192x8192_Defocused_Vr(7128-7178)_8000x256x256.h5'

data = tb.loadh5(datafile)

print(data.keys())

X, ca, cr = data['SI'], data['ca'], data['cr']

N, Na, Nr, _ = X.shape
X = tb.r2c(th.from_numpy(X), cdim=-1)

xa = ts.ppeaxis(Na, norm=True, shift=True, mode='fftfreq')
xr = ts.ppeaxis(Nr, norm=True, shift=True, mode='fftfreq')

pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)

Y = ts.focus(X, pa=pa)

print(X.shape)
modelafcfg = tb.loadyaml('afnet.yaml')
modelpafcfg = tb.loadyaml('pafnet.yaml')
lpafaf, Masaf, Mrsaf = modelafcfg['lpaf'], modelafcfg['Mas'], None  # 52s
lpafhaf, Mashaf, Mrshaf = modelpafcfg['lpaf'], modelpafcfg['Mas'], None
modelfileaf = './record/RealPE/AFnet/AdamW/GaussianLR_nsamples6000/weights/best_valid_664.pth.tar'
modelfilehaf = 'record/RealPE/PAFnet/AdamW/GaussianLR_6000Samples/3Focuser4ConvLayers/weights/best_valid_655.pth.tar'

netaf = PAFnet(Na, 1, Mas=Masaf, lpaf=lpafaf, ftshift=True, seed=2020)
nethaf = PAFnet(Na, 1, Mas=Mashaf, lpaf=lpafhaf, ftshift=True, seed=2020)

modelparamsaf = th.load(modelfileaf, map_location='cpu')
modelparamshaf = th.load(modelfilehaf, map_location='cpu')
netaf.load_state_dict(modelparamsaf['network'])
nethaf.load_state_dict(modelparamshaf['network'])
netaf.eval()
nethaf.eval()

for n in range(61, X.shape[0]):
    Xn, Yn = X[n], Y[n]
    YnPGA, ppai = ts.pgaf_sm(Xn, 6785, nsub=None, windb=None, est='LUMV', deg=7, niter=20, tol=1.e-6, isplot=False, islog=False)
    YnPGA = YnPGA.squeeze(0)

    YnAFnet, _ = netaf.forward(tb.c2r(Xn, cdim=-1).unsqueeze(0))
    YnAFnet = YnAFnet.squeeze(0)
    YnAFnet = tb.r2c(YnAFnet)

    YnPAFnet, _ = nethaf.forward(tb.c2r(Xn, cdim=-1).unsqueeze(0))
    YnPAFnet = YnPAFnet.squeeze(0)
    YnPAFnet = tb.r2c(YnPAFnet)
    
    YnMEA = tb.loadmat('YnMEA.mat')['Y']
    YnMEA = tb.r2c(th.from_numpy(YnMEA), cdim=-1)

    Xn = tb.abs(Xn)
    Yn = tb.abs(Yn)
    YnPGA = tb.abs(YnPGA)
    YnMEA = tb.abs(YnMEA)
    YnAFnet = tb.abs(YnAFnet)
    YnPAFnet = tb.abs(YnPAFnet)

    tbrX = ts.tbr2(Xn, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)
    tbrY = ts.tbr2(Yn, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)
    tbrYPGA = ts.tbr2(YnPGA, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)
    tbrYMEA = ts.tbr2(YnMEA, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)
    tbrYnAFnet = ts.tbr2(YnAFnet, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)
    tbrYnPAFnet = ts.tbr2(YnPAFnet, tgrs=([[68, 140, 71, 143]]), subrs=([[60, 130, 80, 150]]), isshow=False)

    Xn = tb.mapping(Xn)
    Yn = tb.mapping(Yn)
    YnPGA = tb.mapping(YnPGA)
    YnMEA = tb.mapping(YnMEA)
    YnAFnet = tb.mapping(YnAFnet)
    YnPAFnet = tb.mapping(YnPAFnet)

    rects = [[60, 130, 80, 150]]
    Xn = tb.draw_rectangle(Xn, rects=rects, axes=[(0, 1)], linewidths=[2])
    Yn = tb.draw_rectangle(Yn, rects=rects, axes=[(0, 1)], linewidths=[2])
    YnPGA = tb.draw_rectangle(YnPGA, rects=rects, axes=[(0, 1)], linewidths=[2])
    YnMEA = tb.draw_rectangle(YnMEA, rects=rects, axes=[(0, 1)], linewidths=[2])
    YnAFnet = tb.draw_rectangle(YnAFnet, rects=rects, axes=[(0, 1)], linewidths=[2])
    YnPAFnet = tb.draw_rectangle(YnPAFnet, rects=rects, axes=[(0, 1)], linewidths=[2])

    tb.imsave('DefocusedTBR.png', Xn)
    tb.imsave('TruthTBR.png', Yn)
    tb.imsave('PGATBR.png', YnPGA)
    tb.imsave('MEATBR.png', YnMEA)
    tb.imsave('AFnetTBR.png', YnAFnet)
    tb.imsave('PAFnetTBR.png', YnPAFnet)

    print(n, tbrX, tbrY, tbrYPGA, tbrYMEA, tbrYnAFnet, tbrYnPAFnet)
    
    plt = tb.imshow([Xn, Yn, YnPGA, YnMEA, YnAFnet, YnPAFnet])
    plt.show()




