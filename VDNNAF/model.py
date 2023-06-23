import time
import torch as th
import torchbox as tb
from utils import saveimage


class Conv_Block(th.nn.Module):
    def __init__(self):
        super(Conv_Block, self).__init__()
        self.conv = th.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        th.nn.init.xavier_uniform_(self.conv.weight, gain=th.nn.init.calculate_gain('relu'))
        self.relu = th.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(th.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, seed=2020):
        super(Net, self).__init__()
        self.seed = seed
        tb.setseed(seed, target='torch')

        self.input = th.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        th.nn.init.xavier_uniform_(self.input.weight, gain=th.nn.init.calculate_gain('relu'))
        self.conv_layer = self.make_layer(Conv_Block, 18)
        self.output = th.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        th.nn.init.xavier_uniform_(self.output.weight, gain=th.nn.init.calculate_gain('relu'))
        self.relu = th.nn.ReLU(inplace=True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return th.nn.Sequential(*layers)

    def forward(self, x):
        res = x
        out = self.relu(self.input(x))
        out = self.conv_layer(out)
        out = self.output(out)
        out = out + res
        return out

    def withoutres(self, x):
        out = self.relu(self.input(x))
        out = self.conv_layer(out)
        out = self.output(out)
        out = out
        return out

    def train_epoch(self, X, F, sizeBatch, lossfn, epoch, optimizer, scheduler=None, device='cpu'):
        self.train()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = tb.randperm(0, numSamples, numSamples)
        lossvtrain = 0.
        for b in range(numBatch):
            i = idx[b * sizeBatch:(b + 1) * sizeBatch]
            xi, fi = X[i], F[i]
            xi, fi = xi.to(device), fi.to(device)

            optimizer.zero_grad()

            pfi = self.forward(xi)

            loss = lossfn(pfi, fi)

            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            lossvtrain += loss.item()

        tend = time.time()

        lossvtrain /= numBatch
        print("--->Train epoch: %d, loss: %.4f, time: %ss" % (epoch, lossvtrain, tend - tstart))

        return lossvtrain

    def valid_epoch(self, X, F, sizeBatch, lossfn, epoch, device):
        self.eval()

        tstart = time.time()

        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossvvalid = 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi, fi = X[i], F[i]
                xi, fi = xi.to(device), fi.to(device)

                pfi = self.forward(xi)

                loss = lossfn(pfi, fi)

                lossvvalid += loss.item()

        tend = time.time()

        lossvvalid /= numBatch

        print("--->Valid epoch: %d, loss: %.4f, time: %ss" % (epoch, lossvvalid, tend - tstart))

        return lossvvalid

    def test_epoch(self, X, F, sizeBatch, lossfn, epoch, device):
        self.eval()

        tstart = time.time()

        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossvtest = 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi, fi = X[i], F[i]
                xi, fi = xi.to(device), fi.to(device)

                pfi = self.forward(xi)

                loss = lossfn(pfi, fi)

                lossvtest += loss.item()

        tend = time.time()

        lossvtest /= numBatch

        print("--->Test epoch: %d, loss: %.4f, time: %ss" % (epoch, lossvtest, tend - tstart))

        return lossvtest

    def plot(self, xi, fi, idx, prefixname, outfolder, device):

        self.eval()
        with th.no_grad():
            xi = xi.to(device)

            pfi = self.forward(xi)

            xi, fi, pfi = tb.rt2ct(xi, axis=2), tb.rt2ct(fi, axis=2), tb.rt2ct(pfi, axis=2)
            xi, fi, pfi = xi.squeeze(1), fi.squeeze(1), pfi.squeeze(1)
            xi, fi, pfi = th.view_as_real(xi), th.view_as_real(fi), th.view_as_real(pfi)
        saveimage(xi, fi, pfi, idx, prefixname=prefixname, outfolder=outfolder + '/images/')


class SRCNN(th.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SRCNN, self).__init__()
        self.input = th.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, padding=9 // 2)
        self.conv = th.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=5 // 2)
        self.output = th.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=5, padding=5 // 2)
        self.relu = th.nn.ReLU(inplace=True)

        th.nn.init.xavier_uniform_(self.input.weight, gain=th.nn.init.calculate_gain('relu'))
        th.nn.init.xavier_uniform_(self.conv.weight, gain=th.nn.init.calculate_gain('relu'))
        th.nn.init.xavier_uniform_(self.output.weight, gain=th.nn.init.calculate_gain('relu'))

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.relu(self.conv(out))
        out = self.output(out)
        return out
