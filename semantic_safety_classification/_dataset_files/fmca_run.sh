#!/bin/bash

python calibration.py
python evaluation.py True >> $1
python evaluation.py False >> $1
