#!/bin/bash

for i in {1..5}
do
    python context_drone_dataset_generation.py $i
done
