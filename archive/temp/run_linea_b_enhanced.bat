@echo off
cd /d "F:\code_manager\Menglong Cao\sealine_detection"

set PYTHONUNBUFFERED=1
set PYTHONPATH=F:\code_manager\Menglong Cao\sealine_detection;%PYTHONPATH%

echo ============================== >> "train_linea_b_enhanced.log"
echo Start time: %date% %time% >> "train_linea_b_enhanced.log"

"F:\Anaconda\envs\cml\python.exe" -u "LINEA\main.py" -c "stage1_linea_entropy\configs\linea_entropy_b_enhanced_musid.py" >> "train_linea_b_enhanced.log" 2>&1