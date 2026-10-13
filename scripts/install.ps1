$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/aayushkumaraks/waffle-audio.git'
$AppDir = if ($env:WAFFLE_AUDIO_DIR) { $env:WAFFLE_AUDIO_DIR } else { Join-Path $HOME 'waffle-audio' }

function Read-EnvValue([string]$Key) {
    $processValue = [Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($processValue)) { return $processValue }
    if (Test-Path '.env') {
        foreach ($line in Get-Content '.env') {
            if ($line -match ('^' + [regex]::Escape($Key) + '=(.*)$')) { return $Matches[1] }
        }
    }
    return ''
}

function Set-EnvValue([string]$Key, [string]$Value) {
    $lines = @()
    if (Test-Path '.env') {
        $lines = @(Get-Content '.env' | Where-Object { $_ -notmatch ('^' + [regex]::Escape($Key) + '=') })
    }
    $lines += "$Key=$Value"
    [System.IO.File]::WriteAllLines((Join-Path (Get-Location) '.env'), [string[]]$lines, [System.Text.UTF8Encoding]::new($false))
}

function Get-OrPrompt([string]$Key, [string]$Prompt, [string]$Default) {
    $value = Read-EnvValue $Key
    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = Read-Host "$Prompt [$Default]"
        if ([string]::IsNullOrWhiteSpace($value)) { $value = $Default }
    }
    return $value
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install Docker Desktop first: https://docs.docker.com/get-docker/' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }
if (Test-Path (Join-Path $AppDir '.git')) {
    git -C $AppDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Could not update repository; check local changes.' }
} elseif (Test-Path $AppDir) {
    throw "$AppDir exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another location."
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required to clone the project.' }
    git clone $RepoUrl $AppDir
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed.' }
}
Set-Location $AppDir

Write-Host ''
Write-Host 'Configure Ollama. The URL must be reachable FROM the Docker container.'
$ollamaUrl = Get-OrPrompt 'OLLAMA_BASE_URL' 'Ollama base URL' 'http://host.docker.internal:11434'
if ($ollamaUrl -notmatch '^https?://\S+$') { throw 'OLLAMA_BASE_URL must be an http(s) URL without spaces.' }
$ollamaModel = Get-OrPrompt 'OLLAMA_MODEL' 'Ollama model name' 'qwen3.5:4b'
if ($ollamaModel -notmatch '^[A-Za-z0-9._:/-]+$') { throw 'OLLAMA_MODEL contains unsupported characters.' }
$voice = Get-OrPrompt 'VOICECHAT_TTS_VOICE' 'Pocket TTS voice' 'alba'
if ($voice -notmatch '^[A-Za-z0-9._/-]+$') { throw 'Invalid voice identifier.' }
$backend = Read-EnvValue 'VOICECHAT_TTS_BACKEND'
if ([string]::IsNullOrWhiteSpace($backend)) { $backend = 'pocket' }

Set-EnvValue 'OLLAMA_BASE_URL' $ollamaUrl
Set-EnvValue 'OLLAMA_MODEL' $ollamaModel
Set-EnvValue 'VOICECHAT_TTS_BACKEND' $backend
Set-EnvValue 'VOICECHAT_TTS_VOICE' $voice

Write-Host ''
Write-Host 'Building Docker image. This validates dependency installation before launch.'
docker compose build waffle-audio
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed. Review the build output above.' }

Write-Host ''
Write-Host 'Checking Ollama connectivity and running a minimal chat request from inside the image.'
docker compose run --rm --no-deps --entrypoint python waffle-audio-cli scripts/validate-ollama.py
$validationExit = $LASTEXITCODE
if ($validationExit -eq 2) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        $pull = Read-Host "Model '$ollamaModel' is missing. Pull it on this host now? [y/N]"
        if ($pull -match '^(y|yes)$') {
            ollama pull $ollamaModel
            if ($LASTEXITCODE -ne 0) { throw 'ollama pull failed.' }
            docker compose run --rm --no-deps --entrypoint python waffle-audio-cli scripts/validate-ollama.py
            if ($LASTEXITCODE -ne 0) { throw 'Ollama validation still failed.' }
        } else {
            throw "Install the model with: ollama pull $ollamaModel, then rerun the installer."
        }
    } else {
        throw "Install the model on the host with: ollama pull $ollamaModel, then rerun the installer."
    }
} elseif ($validationExit -ne 0) {
    throw 'Ollama validation failed. Check that Ollama is running and OLLAMA_BASE_URL is reachable from Docker.'
}

Write-Host ''
Write-Host 'Configuration and Docker build validated.'
Write-Host 'Docker Desktop on Windows does not provide portable host microphone passthrough to Linux containers. Use web mode and your browser microphone.'
Write-Host "One-click scripts: $(Join-Path $AppDir 'scripts')"
if ((Read-Host 'Run Waffle Audio now? [y/N]') -match '^(y|yes)$') { & '.\scripts\run.ps1'; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; exit 0 }
if ((Read-Host 'Start web/API container in background with Docker restart policy? [y/N]') -match '^(y|yes)$') { & '.\scripts\create-service.ps1'; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; exit 0 }
Write-Host "No action taken. Run later: cd '$AppDir'; .\scripts\run.ps1"
