@echo off
echo ========================================
echo 启动测缝计预测服务
echo ========================================
echo.

cd /d %~dp0
python prediction_service.py

pause

