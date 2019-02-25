#!/bin/usr/env bash

kaggle competitions download -c human-protein-atlas-image-classification -f train.zip
kaggle competitions download -c human-protein-atlas-image-classification -f test.zip
mkdir -p data/raw
unzip train.zip -d data/raw/train
unzip test.zip -d data/raw/test