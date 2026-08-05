@echo off
title GUARACI - Inteligencia Quimiometrica
cd /d "%~dp0"

rem Pacote em ./src (nao exige `pip install -e .`): PYTHONPATH torna `guaraci`
rem importavel a partir da arvore de fontes.
set PYTHONPATH=%~dp0src

rem O ambiente virtual fica FORA do OneDrive de proposito. Dois motivos, ambos
rem verificados na pratica:
rem   1) o sync do OneDrive segura handles de arquivo durante instalacao de
rem      pacote e corrompe o venv (quebrou numpy e jsonschema em 2026-08-04);
rem   2) o venv precisa ser criado com o CPython OFICIAL (python.org). O
rem      CPython proprio do `uv` nao e' assinado e o Smart App Control do
rem      Windows bloqueia sua execucao ("politica de Controle de Aplicativo").
rem Recriar com:  python -m venv %USERPROFILE%\.venvs\guaraci
rem               %USERPROFILE%\.venvs\guaraci\Scripts\pip install -e ".[all]"
set "GUARACI_PY=%USERPROFILE%\.venvs\guaraci\Scripts\python.exe"

if not exist "%GUARACI_PY%" (
    echo.
    echo  [ERRO] Ambiente do GUARACI nao encontrado em:
    echo         %GUARACI_PY%
    echo.
    echo  Crie com os 2 comandos abaixo ^(leva alguns minutos^):
    echo    python -m venv "%USERPROFILE%\.venvs\guaraci"
    echo    "%USERPROFILE%\.venvs\guaraci\Scripts\pip" install -e ".[all]"
    echo.
    pause
    exit /b 1
)

"%GUARACI_PY%" -m guaraci.guaraci
pause
