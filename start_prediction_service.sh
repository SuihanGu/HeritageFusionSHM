#!/bin/bash

echo "========================================"
echo "启动测缝计预测服务"
echo "========================================"
echo ""

cd "$(dirname "$0")"
python prediction_service.py

