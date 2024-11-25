#!/bin/sh
# extract raw data
rm fatal_err.log
mkdir -p cipi
python raw_data_cipi_extractor.py -a  ../../../../CIPI_symbolic/index.json --base ../../../../CIPI_symbolic/ -o cipi_raw_data.json --debug -f cipi -s -i -t 100 -e fatal_err.log

# calculate difficulty features
python difficulty_calculators.py -i cipi -r cipi_raw_data.json -o cipi_difficulties.json --debug
