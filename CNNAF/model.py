import time
import torch as th
import torchsar as ts
from utils import saveimage


class Conv_Block(th.nn.Module):
    def __init__(self, in_channel=32, out_channel=64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=None, bias=False):
        super(Conv_Block, self).__init__()
        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2)) if padding is None else padding
        self.conv = th.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        th.nn.init.xavier_uniform_(self.conv.weight, gain=th.nn.init.calculate_gain('relu'))
        self.relu = th.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(th.nn.Module):
    def __init__(self, channels=[2, 32, 2], kernel_sizes=[(3, 3), (3, 3)], seed=2020):
        super(Net, self).__init__()
        self.seed = seed
        ts.setseed(seed, target='torch')

        in_channels, out_channels = channels[:-1], channels[1:]

        layers = []
        for in_channel, out_channel, kernel_size in zip(in_channels, out_channels, kernel_sizes):
            layers.append(Conv_Block(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size))
        self.conv_layer = th.nn.Sequential(*layers)
        self.relu = th.nn.ReLU(inplace=True)

        self.output1 = th.nn.Conv2d(in_channels=channels[-1], out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        th.nn.init.xavier_uniform_(self.output1.weight, gain=th.nn.init.calculate_gain('relu'))
        self.output2 = th.nn.Conv2d(in_channels=64, out_channels=channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        th.nn.init.xavier_uniform_(self.output2.weight, gain=th.nn.init.calculate_gain('relu'))

    def forward(self, x):
        out = self.conv_layer(x)
        out = out + x
        out = self.relu(self.output1(out))
        out = self.output2(out)
        return out

    def withoutres(self, x):
        out = self.conv_layer(x)
        out = self.output(out)
        return out

    def train_epoch(self, X, F, sizeBatch, lossfn, epoch, optimizer, scheduler=None, device='cpu'):
        self.train()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = ts.randperm(0, numSamples, numSamples)
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

            xi, fi, pfi = xi.permute(0, 2, 3, 1), fi.permute(0, 2, 3, 1), pfi.permute(0, 2, 3, 1)
            saveimage(xi, fi, pfi, idx, prefixname=prefixname, outfolder=outfolder + '/images/')
