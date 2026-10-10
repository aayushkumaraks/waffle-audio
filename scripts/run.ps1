$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir
Set-Location $AppDir
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Desktop is required. Install it from https://www.docker.com/products/docker-desktop/.' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }
Write-Host "`nWaffle Audio`n============`n1) CLI voice assistant (microphone/speakers; host audio passthrough varies)`n2) Web UI + API (use your browser microphone)"
$mode = Read-Host 'Choose a mode [1/2]'
switch ($mode) {
  '1' { Write-Host 'Starting CLI container. Native Windows microphone passthrough is not provided by Docker Desktop; see README.'; docker compose --profile cli run --rm waffle-audio-cli }
  '2' { docker compose --profile web up --build -d waffle-audio; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; $port = if ($env:VOICECHAT_API_PORT) { $env:VOICECHAT_API_PORT } else { '3000' }; Write-Host "Web UI: http://localhost:$port/ui`nAPI docs: http://localhost:$port/docs`nLogs: docker compose logs -f waffle-audio" }
  default { Write-Host "No mode selected. Scripts are located at $ScriptDir" }
}
