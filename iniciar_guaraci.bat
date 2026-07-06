@echo off
title GUARACI - Inteligencia Quimiometrica
cd /d "%~dp0"
rem Pacote em ./src (nao exige `pip install -e .`): PYTHONPATH torna `guaraci` importavel.
set PYTHONPATH=%~dp0src
python -m guaraci.guaraci
pause
