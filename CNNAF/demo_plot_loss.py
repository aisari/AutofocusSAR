import torchsar as ts
import matplotlib.pyplot as plt


figsize = (4, 4)
Nsep = 2
nshots = 300
x = list(range(1, nshots + 1, Nsep))

fontsize = 12
fonttype = 'Times New Roman'
fontdict = {'family': fonttype, 'size': fontsize}
lines = ['--r', '-b']
linews = [1.5, 1, 1.5]
legend = ['Train', 'Valid']

logfile = './record/PE1d/train.log'

plt.figure(figsize=figsize)
plt.grid()

lossestrain = ts.readnum(logfile, pmain='--->Train', psub='loss: ', vfn=float, nshots=nshots)
lossesvalid = ts.readnum(logfile, pmain='--->Valid', psub='loss: ', vfn=float, nshots=nshots)

plt.plot(x, (lossestrain)[::Nsep], lines[0], linewidth=linews[0])
plt.plot(x, (lossesvalid)[::Nsep], lines[1], linewidth=linews[1])
plt.xlabel('Epoch', fontdict=fontdict)
plt.ylabel('Loss', fontdict=fontdict)
plt.xticks(fontproperties=fonttype, size=fontsize)
plt.yticks(fontproperties=fonttype, size=fontsize)
# plt.title('Testing loss versus epoch')
plt.legend(legend, prop=fontdict)
plt.subplots_adjust(left=0.14, bottom=0.12, right=0.995, top=0.96, wspace=0, hspace=0)
plt.savefig('./LossCNNAF.pdf')

plt.show()
