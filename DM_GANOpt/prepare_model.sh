#!/bin/bash

source /home/maps/envs/ccus-env/bin/activate

scriptdir=`pwd`
instance_uuid=$1

cd instances/${instance_uuid}

python ${scriptdir}/prepare_model.py ${instance_uuid}

