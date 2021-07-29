import torch as th
import numpy as np
import torchsar as ts

# drange = [0, 1023]
drange = [0, 255]


def readdata(datafile, key=['SI', 'ca', 'cr'], index=None):

    if datafile[datafile.rfind('.'):] == '.mat':
        data = ts.loadmat(datafile)
    if datafile[datafile.rfind('.'):] in ['.h5', '.hdf5']:
        data = ts.loadh5(datafile)
    X, pa, pr = data[key[0]], data[key[1]], data[key[2]]
    del data
    if index is not None:
        if index[1] == -1:
            index[1] = X.shape[0]
        if len(index) > 2:
            idx = slice(index[0], index[1], index[2])
        else:
            idx = slice(index[0], index[1])
        X, pa, pr = X[idx], pa[idx], pr[idx]
    else:
        pass

    return X, pa, pr


def readdatas(datafiles, keys=[['SI', 'ca', 'cr']], indexes=[None]):

    if type(datafiles) is str:
        return readdata(datafiles, key=keys[0], index=indexes[0])
    if type(datafiles) is tuple or list:
        Xs, pas, prs = th.tensor([]), th.tensor([]), th.tensor([])
        for datafile, key, index in zip(datafiles, keys, indexes):
            X, pa, pr = readdata(datafile, key=key, index=index)
            Xs = th.cat((Xs, th.from_numpy(X)), axis=0)
            pas = th.cat((pas, th.from_numpy(pa)), axis=0)
            prs = th.cat((prs, th.from_numpy(pr)), axis=0)

        return Xs, pas, prs


def readsamples(datafiles, keys=[['SI', 'ca', 'cr']], nsamples=[10], groups=[1], mode='sequentially', seed=None):

    nfiles = len(datafiles)
    if len(keys) == 1:
        keys = keys * nfiles
    if len(nsamples) == 1:
        nsamples = nsamples * nfiles
    if len(groups) == 1:
        groups = groups * nfiles

    Xs, pas, prs = th.tensor([]), th.tensor([]), th.tensor([])
    for datafile, key, n, group in zip(datafiles, keys, nsamples, groups):
        X, pa, pr = readdata(datafile, key=key, index=None)
        N = X.shape[0]
        M = int(N / group)  # each group has M samples
        m = int(n / group)  # each group has m sampled samples

        if (M < m):
            raise ValueError('The tensor does not has enough samples')

        idx = []
        if mode in ['sequentially', 'Sequentially']:
            for g in range(group):
                idx += list(range(int(M * g), int(M * g) + m))
        if mode in ['uniformly', 'Uniformly']:
            for g in range(group):
                idx += list(range(int(M * g), int(M * g + M), int(M / m)))[:m]
        if mode in ['randomly', 'Randomly']:
            ts.setseed(seed)
            for g in range(group):
                idx += ts.randperm(int(M * g), int(M * g + M), m)

        Xs = th.cat((Xs, th.from_numpy(X[idx])), axis=0)
        pas = th.cat((pas, th.from_numpy(pa[idx])), axis=0)
        prs = th.cat((prs, th.from_numpy(pr[idx])), axis=0)
    return Xs, pas, prs


def get_samples(datafile, nsamples=10000, region=None, size=(512, 512), index=None, seed=2020):
    if datafile[datafile.rfind('.'):] == '.mat':
        data = ts.loadmat(datafile)
    if datafile[datafile.rfind('.'):] in ['.h5', '.hdf5']:
        data = ts.loadh5(datafile)
    SI = data['SI']

    del data

    if np.ndim(SI) == 3:
        SI = SI[np.newaxis, :, :, :]

    if region is not None:
        SI = SI[:, region[0]:region[1], region[2]:region[3], :]

    SI = _sample(SI, nsamples, size, index, seed)
    return SI


def _sample(SI, nsamples, size, index, seed=None):

    To = []
    N, Na, Nr, _ = SI.shape
    if nsamples < N:
        N = 1
    num_each = int(nsamples / N)
    ts.setseed(seed)
    for n in range(N):
        imgsize = SI[n].shape  # H-W-2
        if index is None:
            ys = ts.randperm(0, imgsize[0] - size[0] + 1, num_each)
            xs = ts.randperm(0, imgsize[1] - size[1] + 1, num_each)
        else:
            ys, xs = index[0], index[1]
        for k in range(num_each):
            To.append(SI[n, ys[k]:ys[k] + size[0], xs[k]:xs[k] + size[1], :])

    To = np.array(To)

    # N-2-H-W
    SI = th.tensor(To, dtype=th.float32, requires_grad=False)
    return SI


def saveimage(X, Y, idx, prefixname='train', outfolder='./snapshot/'):

    if X.dim() == 3:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    X = X.pow(2).sum(-1).sqrt()
    Y = Y.pow(2).sum(-1).sqrt()

    X, Y = ts.mapping(X), ts.mapping(Y)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()

    for i, ii in zip(range(len(idx)), idx):
        outfileX = outfolder + prefixname + '_unfocused' + str(ii) + '.tif'
        outfileY = outfolder + prefixname + '_focused' + str(ii) + '.tif'
        ts.imsave(outfileX, X[i])
        ts.imsave(outfileY, Y[i])


if __name__ == "__main__":

    pass
