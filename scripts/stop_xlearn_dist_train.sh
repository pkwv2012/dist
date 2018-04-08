#! /bin/sh
#
# stop_xlearn.sh
# Copyright (C) 2017 wangxiaoshu <2012wxs@gmail.com>
#
# Distributed under terms of the MIT license.
#


ps aux | grep dist_xlearn_train | awk '{ print $2 }' | sudo xargs kill -9
