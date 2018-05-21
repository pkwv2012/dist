#!/usr/bash

# cd ~/dev/dist/scripts
# pwd
# ifconfig
# nohup bash local.sh server 2 2 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm &
nohup bash local.sh server 1 1 ../build/dist_xlearn_train ../data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ../data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm -r 1e-5 -k 8 > server.out 2>&1 &
