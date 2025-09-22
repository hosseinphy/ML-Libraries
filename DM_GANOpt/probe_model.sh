#!/bin/bash

source /home/maps/envs/ccus-env/bin/activate

scriptdir=`pwd`
instance_uuid=$1
batch_size=$2

cd instances/${instance_uuid}
python ${scriptdir}/probe_model.py ${instance_uuid} ${batch_size}
