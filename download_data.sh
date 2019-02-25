#!/usr/bin/env bash

kaggle competitions download -c humpback-whale-identification -f train.zip
kaggle competitions download -c humpback-whale-identification -f test.zip
mkdir -p data/input/raw
unzip train.zip -d data/input/raw/train
unzip test.zip -d data/input/raw/test
