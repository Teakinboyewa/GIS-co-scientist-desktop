@echo off
REM GIS Co-Scientist desktop app, with a console showing live server logs.
REM For everyday use, double-click "GIS Co-Scientist.lnk" instead (no console).
if "%GIS_DESKTOP_VENV%"=="" set GIS_DESKTOP_VENV=%USERPROFILE%\venvs\agm_venv

if not exist "%GIS_DESKTOP_VENV%\Scripts\python.exe" (
    echo Could not find Python at "%GIS_DESKTOP_VENV%\Scripts\python.exe".
    echo Set GIS_DESKTOP_VENV to the virtual environment that runs the WebUI.
    pause
    exit /b 1
)

"%GIS_DESKTOP_VENV%\Scripts\python.exe" "%~dp0desktop\launcher.py"
pause
