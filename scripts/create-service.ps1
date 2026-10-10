$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir
Set-Location $AppDir
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Desktop is required.' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required.' }
Write-Host 'This starts the web/API container in the background with restart: unless-stopped.'
if ((Read-Host 'Continue? [y/N]') -notmatch '^(y|yes)$') { Write-Host "No changes made. Scripts are in $ScriptDir"; exit 0 }
docker compose --profile web up --build -d waffle-audio
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$port = if ($env:VOICECHAT_API_PORT) { $env:VOICECHAT_API_PORT } else { '3000' }
Write-Host "Web UI: http://localhost:$port/ui`nStop: docker compose --profile web down"
