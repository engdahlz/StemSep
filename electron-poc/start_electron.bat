@echo off
set ELECTRON_RUN_AS_NODE=
set SCRIPT_DIR=%~dp0
call "%SCRIPT_DIR%node_modules\.bin\electron.cmd" .
