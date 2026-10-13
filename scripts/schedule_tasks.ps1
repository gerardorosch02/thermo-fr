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
  thermo-fr dispatch forecast  weekdays at 09:05 and 09:30 local time (10:05 and 10:30 Paris,
                          all year, since London and Paris change clocks together): runs
                          `gh workflow run` so the GitHub Actions pre-market forecast starts
                          on time; GitHub's own morning crons were delivered hours late or
                          dropped and were removed from the workflow on 2026-10-13. gh uses
                          its stored login, no secret is on the command line, and each
                          dispatch is recorded as a run in the local store (red on the
                          dashboard when it fails).
  thermo-fr dispatch settle  every day at 14:20 local time: runs `gh workflow run` for the
                          settle step on main (same login, logging and single retry), so the
                          public record is scored the same afternoon; the 13:15 UTC settle
                          cron stays as a backup, and a second settle of the same day changes
                          nothing. It runs between the laptop's own 14:00 settle and the 14:30
                          market publish, which fetches main before scoring.
  thermo-fr dispatch refit  the 1st, 2nd and 3rd of each month at 07:00 local time: runs
                          `gh workflow run` for the refit step on main, on the first weekday
                          of the month only (the other days are recorded as skipped), with the
                          same stored login, logging and single retry as the forecast dispatch.
                          The GitHub refit cron stays as a backup; the refit itself runs once
                          per month (`--once-per-month` in the workflow), so a second trigger
                          exits without committing. Task Scheduler's PowerShell cmdlets have
                          no monthly trigger, so the task is registered with a placeholder and
                          its trigger is replaced through the Task Scheduler COM API.
  thermo-fr settle        every day at 14:00 local time
  thermo-fr market publish  every day at 14:30 local time, after the settle: writes
                          published/market.json from the local store and the traded
                          prices, commits that one file and pushes it (pull with
                          rebase first, one retry, never a force push). It works in a
                          dedicated worktree of origin/main (.publish-worktree, git-ignored,
                          created on first use), so the main working tree is never used,
                          whatever branch or uncommitted state it is in. The workflow
                          never writes market.json, so the two publishers touch
                          different files. The commit carries the identity of the
                          repository's local git config, skips the pre-commit hook
                          (a one-file data commit, not code) and its outcome is
                          recorded as a run in the local store, shown on the
                          dashboard's market panel; failures are logged there too.
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
$marketPublishName = "thermo-fr market publish"
$dispatchName = "thermo-fr dispatch forecast"
$refitDispatchName = "thermo-fr dispatch refit"
$settleDispatchName = "thermo-fr dispatch settle"

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
    Remove-IfPresent $marketPublishName
    Remove-IfPresent $dispatchName
    Remove-IfPresent $refitDispatchName
    Remove-IfPresent $settleDispatchName
}

if ($Install) {
    Remove-IfPresent $morningName
    Remove-IfPresent $settleName
    Remove-IfPresent $marketName
    Remove-IfPresent $outageName
    Remove-IfPresent $fuelName
    Remove-IfPresent $marketPublishName
    Remove-IfPresent $dispatchName
    Remove-IfPresent $refitDispatchName
    Remove-IfPresent $settleDispatchName
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

    $dispatchTriggers = @()
    foreach ($time in @("09:05", "09:30")) {
        $dispatchTriggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $time
    }
    $dispatchAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr dispatch --step morning-run --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $dispatchName -Action $dispatchAction -Trigger $dispatchTriggers -Settings $settings `
        -Description "thermo-fr: trigger the GitHub Actions pre-market forecast run (gh workflow run; no cron backs it up since 2026-10-13)" | Out-Null
    Write-Host "Installed task '$dispatchName' (weekdays 09:05 and 09:30 local time, 10:05 and 10:30 Paris)"

    $settleDispatchTrigger = New-ScheduledTaskTrigger -Daily -At "14:20"
    $settleDispatchAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr dispatch --step settle --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $settleDispatchName -Action $settleDispatchAction -Trigger $settleDispatchTrigger -Settings $settings `
        -Description "thermo-fr: trigger the GitHub Actions settle (gh workflow run; the 13:15 UTC cron is a backup, a second settle changes nothing)" | Out-Null
    Write-Host "Installed task '$settleDispatchName' (daily 14:20 local time, between the local settle and the market publish)"

    # Monthly trigger on the 1st to the 3rd at 07:00: New-ScheduledTaskTrigger has no monthly form, so the task is registered with a
    # placeholder daily trigger and its triggers are replaced through the Task Scheduler COM API (trigger type 4, TASK_TRIGGER_MONTHLY,
    # days and months as bitmasks). The command itself dispatches on the first weekday of the month only.
    $refitAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr dispatch --step refit --kind scheduled --first-weekday-only" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $refitDispatchName -Action $refitAction -Trigger (New-ScheduledTaskTrigger -Daily -At "07:00") -Settings $settings `
        -Description "thermo-fr: trigger the GitHub Actions monthly refit on the first weekday of the month (gh workflow run; the GitHub cron is a backup)" | Out-Null
    $service = New-Object -ComObject Schedule.Service
    $service.Connect()
    $folder = $service.GetFolder("\")
    $definition = $folder.GetTask($refitDispatchName).Definition
    $definition.Triggers.Clear()
    $monthly = $definition.Triggers.Create(4)
    $monthly.DaysOfMonth = 7          # bitmask: days 1, 2 and 3
    $monthly.MonthsOfYear = 4095      # every month
    $monthly.StartBoundary = (Get-Date -Format "yyyy-MM-dd") + "T07:00:00"
    $monthly.Enabled = $true
    $folder.RegisterTaskDefinition($refitDispatchName, $definition, 4, $null, $null, 3) | Out-Null   # 4 = TASK_UPDATE, 3 = interactive token
    Write-Host "Installed task '$refitDispatchName' (the 1st, 2nd and 3rd of each month at 07:00 local time; dispatches on the first weekday only)"

    $settleTrigger = New-ScheduledTaskTrigger -Daily -At "14:00"
    $settleAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr settle --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $settleName -Action $settleAction -Trigger $settleTrigger -Settings $settings `
        -Description "thermo-fr: fetch actual day-ahead prices and score the stored forecasts" | Out-Null
    Write-Host "Installed task '$settleName' (daily 14:00 local time)"

    $marketPublishTrigger = New-ScheduledTaskTrigger -Daily -At "14:30"
    $marketPublishAction = New-ScheduledTaskAction -Execute $Python -Argument "-m thermo_fr market publish --kind scheduled" -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $marketPublishName -Action $marketPublishAction -Trigger $marketPublishTrigger -Settings $settings `
        -Description "thermo-fr: write published/market.json from the local store and push that one file (rebase, never force)" | Out-Null
    Write-Host "Installed task '$marketPublishName' (daily 14:30 local time, after the settle; pushes published/market.json only)"

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
    foreach ($name in @($morningName, $dispatchName, $settleDispatchName, $refitDispatchName, $settleName, $marketName, $outageName, $fuelName, $marketPublishName)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -eq $task) { Write-Host "Task '$name' is not installed"; continue }
        $info = Get-ScheduledTaskInfo -TaskName $name
        $next = $info.NextRunTime
        if ($null -eq $next -or $next.Year -lt 2000) {
            # Get-ScheduledTaskInfo reports no next run for a monthly trigger registered through the COM API; schtasks computes it
            $row = schtasks /query /tn $name /fo csv /v | ConvertFrom-Csv | Select-Object -First 1
            $next = "$($row.'Next Run Time') (from schtasks)"
        }
        Write-Host ("{0}: state {1}, next run {2}, last run {3} (result {4}), wake to run {5}, start when available {6}" -f `
            $name, $task.State, $next, $info.LastRunTime, $info.LastTaskResult, $task.Settings.WakeToRun, $task.Settings.StartWhenAvailable)
        $xml = [string](Export-ScheduledTask -TaskName $name)
        foreach ($trigger in $task.Triggers) {
            $start = ([datetime]$trigger.StartBoundary).ToString("HH:mm")
            $days = if ($trigger.DaysOfWeek) { $trigger.DaysOfWeek }
                    elseif ($xml -match "<ScheduleByMonth>") { "monthly, the 1st to the 3rd" }
                    else { "daily" }
            Write-Host ("    trigger at {0} ({1})" -f $start, $days)
        }
    }
}

if (-not ($Install -or $Remove -or $Show)) {
    Write-Host "Use -Install, -Remove or -Show."
}
