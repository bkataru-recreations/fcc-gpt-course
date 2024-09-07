#!/usr/bin/env bash

git clone https://huggingface.co/datasets/Skylion007/openwebtext

SUBSETS_DIR=openwebtext/subsets
DATASET_DIR=openwebtext/dataset

for file in $SUBSETS_DIR/*.tar;
do 
    echo "$file"
    tar xvf "$file" --strip-components=1 --one-top-level=$DATASET_DIR
done

# rm -rf $SUBSETS_DIR