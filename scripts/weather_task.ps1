<#
.SYNOPSIS
Install, show or remove the one scheduled task of the weather collector.

.DESCRIPTION
One task, "thermo-fr weather-collect", every day at 08:00 and 09:45 Paris time, converted to this computer's time zone when
the task is installed (07:00 and 08:45 in London). Each run saves what the public weather sources offer at that moment for
the 00 UTC runs: NOAA GFS and its 31-member ensemble, the ECMWF ensemble, and the arrival times of the original files. The
08:00 run shows what a summer morning would have had, when the gate is an hour earlier in UTC. The 09:45 run is the last
look before 10:05.

The task writes only under the collector's own folder. It reads nothing of the spread logger or of the forward test, changes
no other task, publishes nothing and needs no key. It runs when you are logged on, like the other tasks, and only when a
network is available. If the computer changes time zone, install the task again.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\weather_task.ps1 -Install
powershell -ExecutionPolicy Bypass -File scripts\weather_task.ps1 -Show
powershell -ExecutionPolicy Bypass -File scripts\weather_task.ps1 -Remove
#>
param(
    [switch]$Install,
    [switch]$Show,
    [switch]$Remove,
    [string]$Python = "C:\Users\gerar\Projects\thermo-fr-forward\.venv\Scripts\python.exe"
)

$name = "thermo-fr weather-collect"
$snapshot = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $snapshot
$script = Join-Path $snapshot "scripts\weather_collect.py"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    Write-Host "Removed $name (if it was there)."
}

if ($Install) {
    if (-not (Test-Path $Python)) { throw "No Python at $Python." }
    if (-not (Test-Path $script)) { throw "No collector at $script." }
    Write-Host "Self-test (no network, a second) ..."
    & $Python $script selftest | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "The self-test failed. Nothing was installed." }
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $paris = [TimeZoneInfo]::FindSystemTimeZoneById("Romance Standard Time")
    $triggers = @()
    $local = @()
    foreach ($time in @("08:00", "09:45")) {
        $wall = [DateTime]::SpecifyKind([DateTime]::Parse((Get-Date -Format "yyyy-MM-dd") + " " + $time), [DateTimeKind]::Unspecified)
        $here = [TimeZoneInfo]::ConvertTime($wall, $paris, [TimeZoneInfo]::Local).ToString("HH:mm")
        $local += $here
        $triggers += New-ScheduledTaskTrigger -Daily -At $here
    }
    $argument = "`"$script`" collect"
    $action = New-ScheduledTaskAction -Execute $Python -Argument $argument -WorkingDirectory $root
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings `
        -Description "Weather collector: saves the 00 UTC public forecasts of wind at the German and French points each morning. Writes only under $root." | Out-Null
    Write-Host "Installed ${name}: every day at $($local -join ' and ') on this computer's clock (08:00 and 09:45 Paris), running $script."
}

if ($Show -or -not ($Install -or $Remove)) {
    Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo | Format-List TaskName, LastRunTime, LastTaskResult, NextRunTime
}
