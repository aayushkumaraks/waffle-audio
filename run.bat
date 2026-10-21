@echo off
REM ==============================================================================
REM Waffle Audio — One-Click Install & Run Script (Windows)
REM ==============================================================================

echo.
echo ===================================================
echo   WAFFLE AUDIO — Voice-to-Voice AI Assistant (CLI)
echo ===================================================
echo.

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [1/3] uv package manager not found.
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
) else (
    echo [1/3] uv package manager found.
)

echo [2/3] Synchronizing Python environment...
uv sync

if not exist ".env" (
    echo.
    echo [Setup] Launching interactive configuration setup...
    uv run python setup.py
)

setlocal enabledelayedexpansion
set "VOICE_OPT="

for %%A in (%*) do (
    if "%%A"=="--trigger-mode" (
        echo [Trigger Mode] Active: Assistant will only respond when 'Hey ASSISTANT_NAME' is spoken.
    )
    if "%%A"=="--setup" (
        uv run python setup.py --force
    )
)

REM Check if cloned voices exist
if exist "cloned_voices\*.safetensors" (
    echo.
    echo Available Cloned Voices:
    echo   [0] Default voice
    set /a idx=0
    for %%F in (cloned_voices\*.safetensors) do (
        set /a idx+=1
        echo   [!idx!] %%~nF ^(cloned^)
        set "voice_!idx!=%%~nF"
    )
    set /p choice="Would you like to use a cloned voice for your assistant? [0-!idx!, default: 0]: "
    if defined choice (
        if not "!choice!"=="0" (
            for /f "tokens=1" %%C in ("!choice!") do (
                if defined voice_%%C (
                    for %%V in (!voice_%%C!) do (
                        echo Using cloned voice: %%V
                        set "VOICE_OPT=--voice %%V"
                    )
                )
            )
        )
    )
)

echo.
echo [3/3] Launching Waffle Audio CLI...
uv run python -m services.conversation.cli %* %VOICE_OPT%
