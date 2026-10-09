<#
.SYNOPSIS
Install or remove the Windows Task Scheduler entries for the thermo-fr forecast jobs.

.DESCRIPTION
Creates two tasks that run as the current user, with "run task as soon as
possible after a scheduled start is missed" and "wake the computer to run this
task" turned on. Waking only works if Windows allows wake timers (Power Options,
Sleep, Allow wake timers: Enable, for both plugged in and on battery) and the
machine is asleep or hibernating rather than shut down:

  thermo-fr morning-run   weekdays at 07:00, 08:00, 09:00, 10:00, 10:45 and 11:30
                          (local machine time; the machine is expected to be on
                          London time, which is 08:00 to 12:30 Paris)
  thermo-fr settle        every day at 14:00 local time

Each task starts python -m thermo_fr <command> --kind scheduled in the repository
directory. The jobs read ENTSOE_API_KEY from the user's environment, so the
variable must be set at user level (setx ENTSOE_API_KEY ...), never in this file.
Logs are written by the jobs themselves under logs/.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Install
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Remove
powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Show
#>
param(
    [switch]$Install,
    [switch]$Remove,
    [switch]$Show,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Python -eq "") { $Python = (Get-Command python).Source }
$morningName = "thermo-fr morning-run"
$settleName = "thermo-fr settle"

function Remove-IfPresent($name) {
    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed task '$name'"
    }
}

if ($Remove) {
    Remove-IfPresent $morningName
    Remove-IfPresent $settleName
}

if ($Install) {
    Remove-IfPresent $morningName
    Remove-IfPresent $settleName
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $weekdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

    $morningTriggers = @()
    foreach ($time in @("07:00", "08:00", "09:00", "10:00", "10:45", "11:30")) {
        $morningTriggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $time
    }
    $morningAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr morning-run --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $morningName -Action $morningAction -Trigger $morningTriggers -Settings $settings `
        -Description "thermo-fr: fetch tomorrow's day-ahead inputs, log their timing and store the price forecasts" | Out-Null
    Write-Host "Installed task '$morningName' (weekdays 07:00, 08:00, 09:00, 10:00, 10:45, 11:30 local time)"

    $settleTrigger = New-ScheduledTaskTrigger -Daily -At "14:00"
    $settleAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr settle --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $settleName -Action $settleAction -Trigger $settleTrigger -Settings $settings `
        -Description "thermo-fr: fetch actual day-ahead prices and score the stored forecasts" | Out-Null
    Write-Host "Installed task '$settleName' (daily 14:00 local time)"
    Write-Host "Python: $Python"
    Write-Host "Working directory: $repo"
    Write-Host "Machine time zone: $((Get-TimeZone).Id) (the times above are local; London time is intended)"
}

if ($Show -or $Install) {
    foreach ($name in @($morningName, $settleName)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -eq $task) { Write-Host "Task '$name' is not installed"; continue }
        $info = Get-ScheduledTaskInfo -TaskName $name
        Write-Host ("{0}: state {1}, next run {2}, last run {3} (result {4}), wake to run {5}, start when available {6}" -f `
            $name, $task.State, $info.NextRunTime, $info.LastRunTime, $info.LastTaskResult, $task.Settings.WakeToRun, $task.Settings.StartWhenAvailable)
        foreach ($trigger in $task.Triggers) {
            $start = ([datetime]$trigger.StartBoundary).ToString("HH:mm")
            $days = if ($trigger.DaysOfWeek) { $trigger.DaysOfWeek } else { "daily" }
            Write-Host ("    trigger at {0} ({1})" -f $start, $days)
        }
    }
}

if (-not ($Install -or $Remove -or $Show)) {
    Write-Host "Use -Install, -Remove or -Show."
}
