#!/usr/bash

cd ~/dev/dist/scripts
nohup bash local.sh server 2 2 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm &
