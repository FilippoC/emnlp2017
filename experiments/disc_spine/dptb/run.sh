#!/bin/sh

../../../bin/spine-build-dict \
    ./train \
    ./model

/home/filippo/repos/jparser/bin/spine-train-tagger \
    ./train \
    ./model \
    --eval-on-dev=true \
    --dev-path=./dev \
    --probabilistic=true \
    --iteration=10 \
    --n-stack=1 \
    --n-layer=2 \
    --word-dim=100 \
    --pos-dim=50 \
    --dynet-mem 1024MB \
    --lstm-dim=256

/home/filippo/repos/jparser/bin/spine-train-parser\
    ./train \
    ./model \
    --eval-on-dev=true \
    --dev-path=./dev \
    --best-head=true \
    --probabilistic=true \
    --iteration=10 \
    --n-stack=1 \
    --n-layer=2 \
    --word-dim=100 \
    --pos-dim=50 \
    --lstm-dim=256 \
    --dynet-mem 1024MB \
    --activation-function=tanh

