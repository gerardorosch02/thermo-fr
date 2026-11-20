<#
.SYNOPSIS
Install, show or remove the one scheduled task of the spread logger (docs/prereg/spread_w_amendment_2.md).

.DESCRIPTION
One task, "thermo-fr spread-issue", every day at 10:00 and 10:30 Paris time, converted to this computer's time zone when the
task is installed (09:00 and 09:30 in London). The first run fetches what is public from 10:00 Paris, the deadline of the
load forecast, and writes the record of the next delivery day a few minutes later. If a source fails it writes nothing. The
second run does nothing if the record exists, and otherwise issues with what there is, before the auction closes at 12:00
Paris. The logger itself refuses to run before 10:00 Paris or after the auction, whatever the computer's clock zone.

The task reads the production folder and writes only under the logger's own folder. It changes no other task, publishes
nothing and touches neither git nor the production store. It runs when you are logged on, like the other tasks, and only
when a network is available. If the computer changes time zone, install the task again.

The code runs from the snapshot this script sits in, with the Python of the forward checkout's virtual environment.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\spread_task.ps1 -Install
powershell -ExecutionPolicy Bypass -File scripts\spread_task.ps1 -Show
powershell -ExecutionPolicy Bypass -File scripts\spread_task.ps1 -Remove
#>
param(
    [switch]$Install,
    [switch]$Show,
    [switch]$Remove,
    [switch]$SkipSelfTest,
    [string]$Python = "C:\Users\gerar\Projects\thermo-fr-forward\.venv\Scripts\python.exe"
)

$name = "thermo-fr spread-issue"
$snapshot = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $snapshot
$launcher = Join-Path $snapshot "scripts\run_spread.py"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    Write-Host "Removed $name (if it was there)."
}

if ($Install) {
    if (-not (Test-Path $Python)) { throw "No Python at $Python." }
    if (-not (Test-Path $launcher)) { throw "No launcher at $launcher." }
    if (-not [Environment]::GetEnvironmentVariable("ENTSOE_API_KEY", "User") -and -not [Environment]::GetEnvironmentVariable("ENTSOE_API_KEY", "Machine")) {
        throw "ENTSOE_API_KEY is not set at user level, so the task could not fetch. Nothing was installed."
    }
    if (-not $SkipSelfTest) {
        Write-Host "Self-test (no network, about a minute) ..."
        & $Python $launcher selftest | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "The self-test failed. Nothing was installed." }
    }
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 25) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $paris = [TimeZoneInfo]::FindSystemTimeZoneById("Romance Standard Time")
    $triggers = @()
    $local = @()
    foreach ($time in @("10:00", "10:30")) {
        $wall = [DateTime]::SpecifyKind([DateTime]::Parse((Get-Date -Format "yyyy-MM-dd") + " " + $time), [DateTimeKind]::Unspecified)
        $here = [TimeZoneInfo]::ConvertTime($wall, $paris, [TimeZoneInfo]::Local).ToString("HH:mm")
        $local += $here
        $triggers += New-ScheduledTaskTrigger -Daily -At $here
    }
    $argument = "`"$launcher`" issue"
    $action = New-ScheduledTaskAction -Execute $Python -Argument $argument -WorkingDirectory $root
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings `
        -Description "Spread logger: writes the forecasts of H, HW, G, GK, GM and W for the next delivery day before the auction. Writes only under $root." | Out-Null
    Write-Host "Installed ${name}: every day at $($local -join ' and ') on this computer's clock (10:00 and 10:30 Paris), running $launcher."
}

if ($Show -or -not ($Install -or $Remove)) {
    Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo | Format-List TaskName, LastRunTime, LastTaskResult, NextRunTime
}
