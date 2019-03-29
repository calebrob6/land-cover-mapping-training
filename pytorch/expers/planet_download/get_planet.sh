#!/usr/bin/env bash
API_KEY="201625d504ff440c981e7156e7d0628e"
DIR="/mnt/blobfuse/planet"

python -u pytorch/expers/planet_download/planet_downloads.py \
    --api_key ${API_KEY} \
    --directory ${DIR}