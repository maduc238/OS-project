#! /bin/bash

# Variables:
file_name='pipeline.py'
user='master'
rank=0
master_addr='localhost'
interface='lo'
split=8

taskset -c 0 python3 $file_name --rank=0 --world_size=2 --interface=$interface --master_addr=$master_addr --master_port=29500 --split=$split &
taskset -c 1 python3 $file_name --rank=1 --world_size=2 --interface=$interface --master_addr=$master_addr --master_port=29500 --split=$split
