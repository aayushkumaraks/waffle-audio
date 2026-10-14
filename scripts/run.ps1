$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir
Set-Location $AppDir
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Desktop is required. Install it from https://www.docker.com/products/docker-desktop/.' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }

Write-Host ''
Write-Host 'Waffle Audio'
Write-Host '============'
Write-Host 'Docker Desktop on Windows does not provide portable microphone passthrough to Linux containers.'
Write-Host 'Use Web UI + API mode; the browser captures microphone audio and plays assistant speech.'
$start = Read-Host 'Start Web UI + API? [Y/n]'
if ($start -match '^(n|no)$') { Write-Host "No changes made. Scripts are located at $ScriptDir"; exit 0 }

docker compose --profile web up --build -d waffle-audio
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose failed to build or start the web service.' }
$port = if ($env:VOICECHAT_API_PORT) { $env:VOICECHAT_API_PORT } else { '3000' }
$webPort = if ($env:VOICECHAT_WEBAPP_PORT) { $env:VOICECHAT_WEBAPP_PORT } else { '5173' }
Write-Host "Web UI: http://localhost:$port/ui"
Write-Host "Webapp (Vite): http://localhost:$webPort"
Write-Host "API docs: http://localhost:$port/docs"
Write-Host 'Logs: docker compose logs -f waffle-audio'
