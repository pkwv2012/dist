#!/usr/bin

python ../ps-lite/tracker/dmlc_ssh.py -n 2 -s 2 -H hosts "../build/dist_xlearn_train ../data/libffm_toy/criteo.tr.r100.gbdt0.ffm.small -v ../data/libffm_toy/criteo.va.r100.gbdt0.ffm.small -r 0.0001 -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm"