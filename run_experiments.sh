#!/bin/bash

python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init xavier -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init xavier -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init kaiming -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init kaiming -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init normal -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init normal -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init lecun -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init lecun -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init ikun_v1 -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init ikun_v1 -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt sgd -lr 0.1 -channels 128 -subset-size 20000 -init ikun_v2 -save-es ./logs
python train_csnn.py -T 4 -device cuda:0 -b 128 -epochs 64 -j 8 -data-dir ../data/ -out-dir ./logs -opt adam -lr 0.001 -channels 128 -subset-size 20000 -init ikun_v2 -save-es ./logs
