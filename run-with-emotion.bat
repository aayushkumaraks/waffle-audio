@echo off
REM ==============================================================================
REM Waffle Audio — One-Click Emotional Voice Assistant Script (Windows)
REM ==============================================================================

echo.
echo ==============================================================
echo   WAFFLE AUDIO — Emotional Voice-to-Voice AI Assistant (CLI)
echo ==============================================================
echo.

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [1/4] uv package manager not found.
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
) else (
    echo [1/4] uv package manager found.
)

echo [2/4] Synchronizing Python environment...
uv sync

echo [3/4] Verifying emotion voice models...
set "MISSING=0"
for %%E in (neutral happy sad angry) do (
    if not exist "cloned_voices\%%E_female.safetensors" set "MISSING=1"
    if not exist "cloned_voices\%%E_male.safetensors" set "MISSING=1"
)
if "%MISSING%"=="1" (
    echo Emotion voice safetensors missing. Generating now from voices\output\...
    uv run python generate_emotion_voices.py
)
echo All emotion voice models verified.

setlocal enabledelayedexpansion
set "GENDER_OPT="

echo.
echo Select Assistant Voice Gender:
echo   [1] Female (Emotions: neutral, happy, sad, angry)
echo   [2] Male   (Emotions: neutral, happy, sad, angry)
set /p choice="Enter choice [1-2, default: 1 (Female)]: "
if defined choice (
    if "%choice%"=="2" (
        set "GENDER_OPT=--gender male"
        echo Using Male emotion voice models.
    ) else if /i "%choice%"=="male" (
        set "GENDER_OPT=--gender male"
        echo Using Male emotion voice models.
    ) else (
        set "GENDER_OPT=--gender female"
        echo Using Female emotion voice models.
    )
) else (
    set "GENDER_OPT=--gender female"
    echo Using Female emotion voice models.
)

echo.
echo [4/4] Launching Waffle Audio Emotional Assistant CLI...
uv run python -m services.conversation.emotion_cli %* %GENDER_OPT%
