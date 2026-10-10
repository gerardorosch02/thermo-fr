<#
.SYNOPSIS
Install or remove the Windows Task Scheduler entries for the thermo-fr forecast jobs.

.DESCRIPTION
Creates five tasks that run as the current user, with "run task as soon as
possible after a scheduled start is missed" and "wake the computer to run this
task" turned on. Waking only works if Windows allows wake timers (Power Options,
Sleep, Allow wake timers: Enable, for both plugged in and on battery) and the
machine is asleep or hibernating rather than shut down:

  thermo-fr morning-run   weekdays at 07:00, 08:00, 09:15, 09:45, 10:30 and 11:30
                          (local machine time; the machine is expected to be on
                          London time, so 08:00 to 12:30 Paris: the 09:15 and 09:45
                          runs land at 10:15 and 10:45 Paris, after the 10:00 load
                          forecast deadline and before the 11:15 market window)
  thermo-fr settle        every day at 14:00 local time
  thermo-fr market fetch  weekdays at 11:20 and 11:35 local time (12:20 and 12:35 Paris):
                          collects the EEX day-ahead window for tomorrow (Saturday
                          to Monday on a Friday) with the local, git-ignored
                          collector module; the second trigger is the single
                          retry and stores nothing the first one already has.
                          This task runs here only, never in GitHub Actions.
  thermo-fr outage snapshot  weekdays at 09:00 and Saturday and Sunday at 10:00 local
                          time: saves the ENTSO-E unavailability notices of French
                          units (raw, paged) under data/entsoe/outage_snapshots/,
                          before the pre-market runs, so that planned nuclear
                          availability can later be rebuilt as of any time.
                          Laptop only.
  thermo-fr fuel snapshot  every day at 22:15 local time (after the 22:00 CET evening
                          publication all year): the private collector local\fuel_fetch.py
                          saves the day's gas and carbon prices as a dated snapshot
                          under data/ (git-ignored). Laptop only; the module is not
                          part of the repository.

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
$marketName = "thermo-fr market fetch"
$outageName = "thermo-fr outage snapshot"
$fuelName = "thermo-fr fuel snapshot"

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
    Remove-IfPresent $marketName
    Remove-IfPresent $outageName
    Remove-IfPresent $fuelName
}

if ($Install) {
    Remove-IfPresent $morningName
    Remove-IfPresent $settleName
    Remove-IfPresent $marketName
    Remove-IfPresent $outageName
    Remove-IfPresent $fuelName
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $weekdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

    $morningTriggers = @()
    foreach ($time in @("07:00", "08:00", "09:15", "09:45", "10:30", "11:30")) {
        $morningTriggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $time
    }
    $morningAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr morning-run --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $morningName -Action $morningAction -Trigger $morningTriggers -Settings $settings `
        -Description "thermo-fr: fetch tomorrow's day-ahead inputs, log their timing and store the price forecasts" | Out-Null
    Write-Host "Installed task '$morningName' (weekdays 07:00, 08:00, 09:15, 09:45, 10:30, 11:30 local time)"

    $settleTrigger = New-ScheduledTaskTrigger -Daily -At "14:00"
    $settleAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr settle --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $settleName -Action $settleAction -Trigger $settleTrigger -Settings $settings `
        -Description "thermo-fr: fetch actual day-ahead prices and score the stored forecasts" | Out-Null
    Write-Host "Installed task '$settleName' (daily 14:00 local time)"

    $marketTriggers = @()
    foreach ($time in @("11:20", "11:35")) {
        $marketTriggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $time
    }
    $marketAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr market fetch" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $marketName -Action $marketAction -Trigger $marketTriggers -Settings $settings `
        -Description "thermo-fr: collect the EEX day-ahead trading window for tomorrow with the local collector (laptop only)" | Out-Null
    Write-Host "Installed task '$marketName' (weekdays 11:20 and 11:35 local time, 12:20 and 12:35 Paris)"

    $outageTriggers = @(
        (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "09:00"),
        (New-ScheduledTaskTrigger -Weekly -DaysOfWeek @("Saturday", "Sunday") -At "10:00")
    )
    $outageAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr outage-snapshot" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $outageName -Action $outageAction -Trigger $outageTriggers -Settings $settings `
        -Description "thermo-fr: save today's ENTSO-E unavailability notices of French units (raw snapshot, laptop only)" | Out-Null
    Write-Host "Installed task '$outageName' (weekdays 09:00, weekends 10:00 local time)"

    $fuelScript = Join-Path $repo "local\fuel_fetch.py"
    if (Test-Path $fuelScript) {
        $fuelTrigger = New-ScheduledTaskTrigger -Daily -At "22:15"
        $fuelAction = New-ScheduledTaskAction -Execute $Python -Argument "`"$fuelScript`"" -WorkingDirectory $repo
        Register-ScheduledTask -TaskName $fuelName -Action $fuelAction -Trigger $fuelTrigger -Settings $settings `
            -Description "thermo-fr: private collector, daily snapshot of gas and carbon prices (laptop only)" | Out-Null
        Write-Host "Installed task '$fuelName' (daily 22:15 local time)"
    } else {
        Write-Host "Skipped task '$fuelName': no local\fuel_fetch.py on this machine"
    }
    Write-Host "Python: $Python"
    Write-Host "Working directory: $repo"
    Write-Host "Machine time zone: $((Get-TimeZone).Id) (the times above are local; London time is intended)"
}

if ($Show -or $Install) {
    foreach ($name in @($morningName, $settleName, $marketName, $outageName, $fuelName)) {
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
