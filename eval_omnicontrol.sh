#!/bin/sh
model_path=$1
joint=$2
density=$3
# eval root joint
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint ${joint} --density ${density}
