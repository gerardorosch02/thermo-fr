<#
.SYNOPSIS
Install, show or remove the one scheduled task of the forward live log of Q (docs/prereg/forward_q_prereg.md).

.DESCRIPTION
One task, "thermo-fr forward-issue", on weekdays at 09:22 and 09:52 local time (London intended: 10:22 and 10:52 Paris),
a few minutes after the 09:15 and 09:45 morning runs and before the market window opens at 11:15 Paris. It reads the
production folder (the store, the inputs and the saved morning windows) and writes only under data\forward there. It
changes no other task, publishes nothing and touches neither git nor the store.

The code runs from a second checkout on branch forward-q with its own virtual environment, so the production checkout
stays on main.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\forward_task.ps1 -Install
powershell -ExecutionPolicy Bypass -File scripts\forward_task.ps1 -Show
powershell -ExecutionPolicy Bypass -File scripts\forward_task.ps1 -Remove
#>
param(
    [switch]$Install,
    [switch]$Show,
    [switch]$Remove,
    [string]$Production = "C:\Users\gerar\Projects\thermo-fr\thermo-fr",
    [string]$Forward = "C:\Users\gerar\Projects\thermo-fr-forward"
)

$name = "thermo-fr forward-issue"
$python = Join-Path $Forward ".venv\Scripts\python.exe"
$weekdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

if ($Remove) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    Write-Host "Removed $name (if it was there)."
}

if ($Install) {
    if (-not (Test-Path $python)) { throw "No Python at $python. Create the forward checkout and its virtual environment first." }
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $triggers = @()
    foreach ($time in @("09:22", "09:52")) { $triggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $time }
    $argument = "-m thermo_fr.forecast.forward_q issue --repo `"$Forward`""
    $action = New-ScheduledTaskAction -Execute $python -Argument $argument -WorkingDirectory $Production
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings `
        -Description "Forward live log of Q: issues L0, P, Q, S1 and S1-Q beside the pre-market honest_v2 version. Writes only under data\forward." | Out-Null
    Write-Host "Installed $name: weekdays 09:22 and 09:52."
}

if ($Show -or -not ($Install -or $Remove)) {
    Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo | Format-List TaskName, LastRunTime, LastTaskResult, NextRunTime
}
