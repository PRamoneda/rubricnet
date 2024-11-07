#!/bin/sh
# Set CUDA_VISIBLE_DEVICES to an empty value to disable GPU usage
export CUDA_VISIBLE_DEVICES=""

# extract raw data
rm fatal_err.log
mkdir -p cipi
python raw_data_cipi_extractor.py -a  ../all.json --base ../ -o split_raw_data.json --debug -f ../ -s -i -t 100 -e fatal_err.log

# calculate difficulty features
python difficulty_calculators.py -i cipi -r split_raw_data.json -o split_difficulties.json --debug
