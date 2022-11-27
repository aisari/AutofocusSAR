
import torch as th
import torchbox as tb
import torchsar as ts


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

x, y = X[61, 69-32:69+32, 141], Y[61, 69-32:69+32, 141]
xx, yy = tb.fft(x, dim=0, shift=True), tb.fft(y, dim=0, shift=True)
plt = tb.plot([[x.abs()], [y.abs()], [xx.abs()], [yy.abs()]])
plt.show()

for n in range(60, X.shape[0]):
    Xn = tb.mapping(tb.abs(X[n]))
    Yn = tb.mapping(tb.abs(Y[n]))
    print(n)

    plt = tb.imshow([Xn, Yn])
    plt.show()





