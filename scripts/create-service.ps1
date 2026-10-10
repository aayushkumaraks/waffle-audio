$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir
$RunScript = Join-Path $AppDir 'scripts\run.ps1'
$TaskName = 'Waffle Audio'
$PowerShell = (Get-Process -Id $PID).Path
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`"" -WorkingDirectory $AppDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 0) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description 'Starts the local Waffle Audio API at user login.' -Force | Out-Null
Write-Host "Login task '$TaskName' installed. Manage it in Task Scheduler."
Write-Host "Remove with: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
