$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir
Set-Location $AppDir
if (-not (Test-Path (Join-Path $AppDir '.venv\Scripts\python.exe'))) { throw 'Dependencies are not installed. Run scripts\install.ps1 first.' }
$envFile = Join-Path $AppDir '.env'
if (Test-Path $envFile) { Get-Content $envFile | ForEach-Object { if ($_ -match '^(VOICECHAT_TTS_BACKEND|VOICECHAT_TTS_VOICE|VOICECHAT_API_HOST|VOICECHAT_API_PORT|VOICECHAT_CORS_ALLOW_ORIGINS)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } } }
if (-not $env:VOICECHAT_TTS_BACKEND) { $env:VOICECHAT_TTS_BACKEND = 'pocket' }
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null } catch {
  Write-Warning 'Ollama is not reachable. Start Ollama and ensure qwen3.5:4b is installed.'
  if (-not ((Read-Host 'Start the Waffle Audio API anyway? [y/N]') -match '^(y|yes)$')) { exit 1 }
}
& uv run python -m services.api.app
exit $LASTEXITCODE
