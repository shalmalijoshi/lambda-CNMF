#!/bin/bash

for s in 0.1 0.3 0.5 0.7 0.9 0.0
do
    echo "$s"
    python main_constrainednmf_folds.py -t <tau> -s $s -b <bias> -i <foldid> -l sparse_poisson  -f <data pickle folder path>
done

