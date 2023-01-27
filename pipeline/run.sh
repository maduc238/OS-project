#! /bin/bash
python3 pipeline.py --rank=0 --world_size=2 --interface=lo --master_addr=localhost --master_port=29500 &
python3 pipeline.py --rank=1 --world_size=2 --interface=lo --master_addr=localhost --master_port=29500
