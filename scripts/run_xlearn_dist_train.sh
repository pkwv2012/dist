#! /bin/sh

echo "=====================================BigDataLR================================="
# nohup bash local.sh scheduler 2 2 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm &
nohup bash local.sh scheduler 2 2 ../build/dist_xlearn_train ../data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ../data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm &
server_list="10.0.0.12 10.0.0.14"
for server in $server_list
do
    echo $server
    ssh zhengpeikai@$server < start_server.sh &
done

worker_list="10.0.0.12 10.0.0.14"

for worker in $worker_list
do
    echo $worker
    ssh zhengpeikai@$worker < start_worker.sh &
done
# bash local.sh server 1 3 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
# bash local.sh worker 1 3 ../build/dist_xlearn_train ../demo/classification/criteo_ctr/small_train.txt -v ../demo/classification/criteo_ctr/small_train.txt -s 0 -x auc --dis-es -p sgd -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#sh local.sh 1 3 ../build/dist_xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p adagrad -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#sh local.sh 1 3 ../build/dist_xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFM================================="
#sh local.sh 1 1 ../build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFFM==============================="
#sh local.sh 1 1 ../build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
