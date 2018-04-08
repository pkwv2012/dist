#! /bin/sh

echo "=====================================BigDataLR================================="
bash local.sh 1 3 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#sh local.sh 1 3 ../build/dist_xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p adagrad -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#sh local.sh 1 3 ../build/dist_xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFM================================="
#sh local.sh 1 1 ../build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFFM==============================="
#sh local.sh 1 1 ../build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
