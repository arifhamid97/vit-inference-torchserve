#!/bin/bash
torch-model-archiver \
    --model-name vision_transformer \
    --version 1.0 \
    --serialized-file torch_model/pytorch_model.bin \
    --export-path model_store \
    --force \
    --handler handler.py \
    --extra-files torch_model/config.json,torch_model/preprocessor_config.json
