#!/bin/bash
#

loss_type='Entropy'
mkpetype='RealPE'
# mkpetype='SimPoly'

optimizer='AdamW'
scheduler='StepLR'
size_batch=8
lr=1e-2
device='cuda:0'
num_epochs=1000
snapshot_name='2020'

snapshotdir='./snapshot/PAFnet/'$mkpetype'/'$loss_type'/'$optimizer'/'$scheduler'/'$snapshot_name'/'

mkdir -p $snapshotdir

python3 train.py \
--datacfg='./data.yaml' \
--modelcfg='./pafnet.yaml' \
--solvercfg='./solver.yaml' \
--loss_type=$loss_type \
--mkpetype=$mkpetype \
--optimizer=$optimizer \
--scheduler=$scheduler \
--size_batch=$size_batch \
--lr=$lr \
--device=$device \
--num_epochs=$num_epochs \
--snapshot_name=$snapshot_name \
>> $snapshotdir'/training.log'

