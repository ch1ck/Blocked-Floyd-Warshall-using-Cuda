#!/bin/sh
# this script is use to link the testcase to the real cases

DIR=/home/pp20/share/.testcase/hw4

for dir in $DIR/*; do
	echo $dir
	echo $(basename "${dir}")
	ln -s $dir $(basename "${dir}")
done
