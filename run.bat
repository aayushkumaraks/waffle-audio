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

echo [3/3] Launching Waffle Audio CLI...
uv run python -m services.conversation.cli %*
