$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/aayushkumaraks/waffle-audio.git'
$AppDir = if ($env:WAFFLE_AUDIO_DIR) { $env:WAFFLE_AUDIO_DIR } else { Join-Path $HOME 'waffle-audio' }
$Model = 'qwen3.5:4b'
function Info($Text) { Write-Host "`n[Waffle Audio] $Text" -ForegroundColor Cyan }
function Ask($Text) { $answer = Read-Host "$Text [y/N]"; return $answer -match '^(y|yes)$' }

Info 'Checking prerequisites (Python 3.11, Git, uv, Ollama, audio support)...'
$py = Get-Command py -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue
$PythonExe = $null
if ($py) { try { & py -3.11 --version *> $null; if ($LASTEXITCODE -eq 0) { $PythonExe = 'py -3.11' } } catch {} }
if (-not $PythonExe -and $python) { try { $v = & python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'; if ($v -eq '3.11') { $PythonExe = 'python' } } catch {} }
if (-not $PythonExe) { throw 'Python 3.11 is required. Install from https://www.python.org/downloads/windows/ and enable Add Python to PATH, then rerun.' }
Info "Python 3.11 found ($PythonExe)."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required. Install Git for Windows from https://git-scm.com/download/win.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  if (Ask 'Install uv using the official PowerShell installer?') { irm https://astral.sh/uv/install.ps1 | iex; $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path" }
  else { throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/.' }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv was not found on PATH after installation. Open a new PowerShell window and rerun.' }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Warning 'Ollama is required for local LLM responses. Install from https://ollama.com/download/windows.'
  if (-not (Ask 'Continue setup without Ollama for now?')) { throw 'Install Ollama and rerun this installer.' }
} else { Info (& ollama --version) }

if (Test-Path (Join-Path $AppDir '.git')) { Info "Updating existing repository at $AppDir"; git -C $AppDir pull --ff-only; if ($LASTEXITCODE -ne 0) { throw 'Could not update repository; check local changes.' } }
elseif (Test-Path $AppDir) { throw "$AppDir exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another location." }
else { Info "Cloning Waffle Audio into $AppDir"; git clone $RepoUrl $AppDir; if ($LASTEXITCODE -ne 0) { throw 'Git clone failed.' } }
Set-Location $AppDir
Info 'Installing dependencies with uv sync (first install can take several minutes)...'
uv sync
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Review output and rerun.' }

Write-Host @'

Pocket TTS voices (default: alba)
English: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette,
         eponine, eve, fantine, george, jane, jean, javert, marius, mary,
         michael, paul, peter_yearsley, stuart_bell, vera
French: estelle | German: juergen | Italian: giovanni
Portuguese: rafael | Spanish: lola
Catalog and voice licenses: https://huggingface.co/kyutai/tts-voices
'@
$voice = Read-Host 'Pocket TTS voice [alba]'
if ([string]::IsNullOrWhiteSpace($voice)) { $voice = 'alba' }
if ($voice -notmatch '^[A-Za-z0-9._/-]+$') { throw 'Invalid voice identifier.' }
$envPath = Join-Path $AppDir '.env'
$settings = @{}
if (Test-Path $envPath) { Get-Content $envPath | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { $settings[$matches[1]] = $matches[2] } } }
$settings['VOICECHAT_TTS_BACKEND'] = 'pocket'
$settings['VOICECHAT_TTS_VOICE'] = $voice
$settings.GetEnumerator() | Sort-Object Name | ForEach-Object { '{0}={1}' -f $_.Key, $_.Value } | Set-Content -Encoding utf8 $envPath
Info "Selected Pocket TTS voice: $voice"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
  try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null } catch { Write-Warning 'Ollama is not reachable. Start Ollama before running the app.' }
  $models = & ollama list 2>$null
  if (($models -join "`n") -notmatch [regex]::Escape($Model)) { if (Ask "Pull required Ollama model '$Model' now?") { & ollama pull $Model; if ($LASTEXITCODE -ne 0) { Write-Warning "Run ollama pull $Model before starting." } } }
}
Info "Installation complete: $AppDir"
Info "One-click scripts are located at: $(Join-Path $AppDir 'scripts')"
if (Ask 'Start Waffle Audio now?') { & (Join-Path $AppDir 'scripts\run.ps1'); exit $LASTEXITCODE }
if (Ask 'Configure Waffle Audio to start on login?') { & (Join-Path $AppDir 'scripts\create-service.ps1'); exit $LASTEXITCODE }
Info "No action taken. Scripts remain at $(Join-Path $AppDir 'scripts'). Run scripts\run.ps1 when ready."
