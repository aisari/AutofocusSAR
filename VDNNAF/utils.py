import torch as th
import numpy as np
import torchbox as tb

# drange = [0, 1023]
drange = [0, 255]


def sample(A, B, nsamples, size, index, seed=None):

    A, B = A.numpy(), B.numpy()
    Ao, Bo = [], []
    N, _, Na, Nr = A.shape
    if nsamples < N:
        N = 1
    num_each = int(nsamples / N)
    tb.setseed(seed)
    for n in range(N):
        imgsize = A[n].shape[1:]  # C-H-W
        if index is None:
            ys = tb.randperm(0, imgsize[0] - size[0] + 1, num_each)
            xs = tb.randperm(0, imgsize[1] - size[1] + 1, num_each)
        else:
            ys, xs = index[0], index[1]

        for k in range(num_each):
            Ao.append(A[n, :, ys[k]:ys[k] + size[0], xs[k]:xs[k] + size[1]])
            Bo.append(B[n, :, ys[k]:ys[k] + size[0], xs[k]:xs[k] + size[1]])
    Ao = np.array(Ao)
    Bo = np.array(Bo)
    Ao = th.from_numpy(Ao)
    Bo = th.from_numpy(Bo)
    # nsamples-C-size
    return Ao, Bo


def saveimage(X, Y, F, idx, prefixname='train', outfolder='./snapshot/'):

    if th.is_complex(X):
        X = th.view_as_real(X)
    if th.is_complex(Y):
        Y = th.view_as_real(Y)
    if th.is_complex(F):
        F = th.view_as_real(F)
    if X.dim() == 3:
        X = X.unsqueeze(0)
    if Y.dim() == 3:
        Y = Y.unsqueeze(0)
    if F.dim() == 3:
        F = F.unsqueeze(0)

    X = X.pow(2).sum(-1).sqrt()
    Y = Y.pow(2).sum(-1).sqrt()
    F = F.pow(2).sum(-1).sqrt()

    X, Y, F = tb.mapping(X), tb.mapping(Y), tb.mapping(F)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    F = F.cpu().detach().numpy()

    for i, ii in zip(range(len(idx)), idx):
        outfileX = outfolder + prefixname + '_unfocused' + str(ii) + '.tif'
        outfileY = outfolder + prefixname + '_gtruth' + str(ii) + '.tif'
        outfileF = outfolder + prefixname + '_focused' + str(ii) + '.tif'
        tb.imsave(outfileX, X[i])
        tb.imsave(outfileY, Y[i])
        tb.imsave(outfileF, F[i])


if __name__ == "__main__":

    pass
