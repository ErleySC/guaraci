@echo off
title Plataforma Quimiometrica - Iniciando...
cd /d "C:\Users\erley\OneDrive\Documentos\ERLEY\Pibic\code py"
echo.
echo  ==========================================
echo   Plataforma Quimiometrica (PLS-DA / PIBIC)
echo  ==========================================
echo  Abrindo interface em http://localhost:8501
echo  Pressione Ctrl+C para encerrar o servidor.
echo.
streamlit run app_quimiometria.py --server.runOnSave=false --server.port=8501
pause
