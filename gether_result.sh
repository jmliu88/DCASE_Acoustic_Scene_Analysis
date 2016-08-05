#!/bin/sh
path=$1
log=$2
find $path -maxdepth 2 -mindepth 2 -type d -exec sh -c "echo {}>>$log; python do_system_evaluation.py {}>>$log" \;
