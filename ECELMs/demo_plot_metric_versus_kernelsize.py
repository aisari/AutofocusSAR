import torch as th
import torchsar as ts
import matplotlib.pyplot as plt


figsize = (4, 4)
Nsep = 1
nshots = 32
x = list(range(1, nshots * 2, 2))

fontsize = 12
fonttype = 'Times New Roman'
fontdict = {'family': fonttype, 'size': fontsize}
lines = ['-^r', '-ob', '-sg']
linews = [1.5, 1, 1.5]
legend = ['Train', 'Valid', 'Test']

logfile = './record/RealPE/1ECELMs/train_valid_test.log'

plt.figure(figsize=figsize)
plt.grid()

lossestrain = ts.readnum(logfile, pmain='--->Train ensemble', psub='entropy: ', vfn=float, nshots=nshots)
lossesvalid = ts.readnum(logfile, pmain='--->Valid ensemble', psub='entropy: ', vfn=float, nshots=nshots)
lossestest = ts.readnum(logfile, pmain='--->Test ensemble', psub='entropy: ', vfn=float, nshots=nshots)
print(lossestrain)
plt.plot(x, (lossestrain)[::Nsep], lines[0], linewidth=linews[0])
plt.plot(x, (lossesvalid)[::Nsep], lines[1], linewidth=linews[1])
plt.plot(x, (lossestest)[::Nsep], lines[2], linewidth=linews[2])
plt.xlabel('Kernel size', fontdict=fontdict)
plt.ylabel('Entropy', fontdict=fontdict)
plt.xticks(fontproperties=fonttype, size=fontsize)
plt.yticks(fontproperties=fonttype, size=fontsize)
# plt.title('Testing loss versus epoch')
plt.legend(legend, prop=fontdict)
# plt.subplots_adjust(left=0.14, bottom=0.09, right=0.995, top=0.995, wspace=0, hspace=0)  # (5, 5)
plt.subplots_adjust(left=0.175, bottom=0.115, right=0.995, top=0.995, wspace=0, hspace=0)  # (4, 4)
plt.savefig('./BaggingECELMsLossEntropyKernelSize.pdf')

plt.show()


plt.figure(figsize=figsize)
plt.grid()

lossestrain = ts.readnum(logfile, pmain='--->Train ensemble', psub='contrast: ', vfn=float, nshots=nshots)
lossesvalid = ts.readnum(logfile, pmain='--->Valid ensemble', psub='contrast: ', vfn=float, nshots=nshots)
lossestest = ts.readnum(logfile, pmain='--->Test ensemble', psub='contrast: ', vfn=float, nshots=nshots)

lossestrain = [-x for x in lossestrain]
lossesvalid = [-x for x in lossesvalid]
lossestest = [-x for x in lossestest]

plt.plot(x, (lossestrain)[::Nsep], lines[0], linewidth=linews[0])
plt.plot(x, (lossesvalid)[::Nsep], lines[1], linewidth=linews[1])
plt.plot(x, (lossestest)[::Nsep], lines[2], linewidth=linews[2])
plt.xlabel('Kernel size', fontdict=fontdict)
plt.ylabel('Contrast', fontdict=fontdict)
plt.xticks(fontproperties=fonttype, size=fontsize)
plt.yticks(fontproperties=fonttype, size=fontsize)
# plt.title('Testing loss versus epoch')
plt.legend(legend, prop=fontdict)
# plt.subplots_adjust(left=0.125, bottom=0.09, right=0.995, top=0.995, wspace=0, hspace=0)  # (5, 5)
plt.subplots_adjust(left=0.155, bottom=0.115, right=0.995, top=0.995, wspace=0, hspace=0)  # (4, 4)
plt.savefig('./BaggingECELMsLossContrastKernelSize.pdf')

plt.show()
