$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/aayushkumaraks/waffle-audio.git'
$AppDir = if ($env:WAFFLE_AUDIO_DIR) { $env:WAFFLE_AUDIO_DIR } else { Join-Path $HOME 'waffle-audio' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install Docker Desktop first: https://docs.docker.com/get-docker/' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }
if (Test-Path (Join-Path $AppDir '.git')) { git -C $AppDir pull --ff-only; if ($LASTEXITCODE -ne 0) { throw 'Could not update repository; check local changes.' } }
elseif (Test-Path $AppDir) { throw "$AppDir exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another location." }
else { if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required to clone the project.' }; git clone $RepoUrl $AppDir; if ($LASTEXITCODE -ne 0) { throw 'Git clone failed.' } }
Set-Location $AppDir
Write-Host @'

Pocket TTS voices (default: alba)
English: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette,
         eponine, eve, fantine, george, jane, jean, javert, marius, mary,
         michael, paul, peter_yearsley, stuart_bell, vera
French: estelle | German: juergen | Italian: giovanni
Portuguese: rafael | Spanish: lola
Catalog and licenses: https://huggingface.co/kyutai/tts-voices
'@
$voice = Read-Host 'Pocket TTS voice [alba]'; if ([string]::IsNullOrWhiteSpace($voice)) { $voice = 'alba' }
if ($voice -notmatch '^[A-Za-z0-9._/-]+$') { throw 'Invalid voice identifier.' }
$settings = @{}; if (Test-Path '.env') { Get-Content '.env' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { if ($matches[1] -notin @('VOICECHAT_TTS_BACKEND','VOICECHAT_TTS_VOICE')) { $settings[$matches[1]] = $matches[2] } } } }
$settings['VOICECHAT_TTS_BACKEND'] = 'pocket'; $settings['VOICECHAT_TTS_VOICE'] = $voice
$settings.GetEnumerator() | Sort-Object Name | ForEach-Object { '{0}={1}' -f $_.Key, $_.Value } | Set-Content -Encoding utf8 '.env'
Write-Host 'Docker builds the runtime and installs Python/system dependencies inside the image.'
Write-Host 'Install and start Ollama on the host, then run: ollama pull qwen3.5:4b'
Write-Host "One-click scripts: $(Join-Path $AppDir 'scripts')"
if ((Read-Host 'Run Waffle Audio now? [y/N]') -match '^(y|yes)$') { & '.\scripts\run.ps1'; exit $LASTEXITCODE }
if ((Read-Host 'Start web/API container in background with Docker restart policy? [y/N]') -match '^(y|yes)$') { & '.\scripts\create-service.ps1'; exit $LASTEXITCODE }
Write-Host "No action taken. Run later: cd '$AppDir'; .\scripts\run.ps1"
