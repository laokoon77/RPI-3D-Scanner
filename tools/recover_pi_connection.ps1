param(
    [string]$Pi = "user@100.106.34.100",
    [string]$Password = "ok",
    [string]$HostKey = "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4",
    [int]$LocalPort = 18000,
    [int]$RemotePort = 8000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RemoteProjectPath = "/home/user/RPI 3D Scanner"
$RemoteLogPath = "/tmp/scanner_uvicorn.log"

function Write-Marker {
    param([string]$Text)
    Write-Host $Text
}

function Invoke-PlinkCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteCommand,
        [switch]$NoBatch
    )

    $args = @("-ssh")
    if (-not $NoBatch) {
        $args += "-batch"
    }
    $args += @("-pw", $Password, "-hostkey", $HostKey, $Pi, $RemoteCommand)

    $output = & $script:PlinkPath @args 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).TrimEnd()

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output   = $text
    }
}

function ConvertTo-ShDoubleQuotedLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $escaped = $Value
    $escaped = $escaped.Replace('\', '\\')
    $escaped = $escaped.Replace('"', '\"')
    $escaped = $escaped.Replace('$', '\$')
    return '"' + $escaped + '"'
}

function Get-LocalTunnelPlinkProcesses {
    $needle = "-L $LocalPort`:127.0.0.1:$RemotePort"
    Get-CimInstance Win32_Process -Filter "Name='plink.exe'" |
        Where-Object { $_.CommandLine -like "*$needle*" }
}

function Show-Diagnostics {
    Write-Marker "DIAG_BEGIN"

    try {
        Write-Marker "DIAG_REMOTE_PS_BEGIN"
        $psDiag = Invoke-PlinkCommand -RemoteCommand "sh -lc 'ps -ef | grep -E \"uvicorn|webapp:app|python.*uvicorn\" | grep -v grep || true'"
        Write-Host $psDiag.Output
        Write-Marker "DIAG_REMOTE_PS_END"
    } catch {
        Write-Host "DIAG_REMOTE_PS_ERROR=$($_.Exception.Message)"
    }

    try {
        Write-Marker "DIAG_REMOTE_SS_BEGIN"
        $ssDiag = Invoke-PlinkCommand -RemoteCommand "sh -lc 'ss -tlnp | grep :$RemotePort || true'"
        Write-Host $ssDiag.Output
        Write-Marker "DIAG_REMOTE_SS_END"
    } catch {
        Write-Host "DIAG_REMOTE_SS_ERROR=$($_.Exception.Message)"
    }

    try {
        Write-Marker "DIAG_REMOTE_LOG_BEGIN"
        $logDiag = Invoke-PlinkCommand -RemoteCommand "sh -lc 'tail -n 80 $RemoteLogPath 2>/dev/null || echo NO_REMOTE_LOG'"
        Write-Host $logDiag.Output
        Write-Marker "DIAG_REMOTE_LOG_END"
    } catch {
        Write-Host "DIAG_REMOTE_LOG_ERROR=$($_.Exception.Message)"
    }

    Write-Marker "DIAG_LOCAL_PLINK_BEGIN"
    $localPlinks = Get-CimInstance Win32_Process -Filter "Name='plink.exe'" |
        Select-Object ProcessId, CommandLine
    if ($localPlinks) {
        $localPlinks | Format-Table -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "NO_LOCAL_PLINK"
    }
    Write-Marker "DIAG_LOCAL_PLINK_END"

    Write-Marker "DIAG_END"
}

function Fail-Recovery {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason
    )

    Write-Marker "RECOVER_FAIL:$Reason"
    Show-Diagnostics
    exit 1
}

Write-Marker "RECOVER_START"
Write-Marker "DEFAULTS:PI=$Pi;HOSTKEY=$HostKey;REMOTE_PATH=$RemoteProjectPath;LOCAL_PORT=$LocalPort;REMOTE_PORT=$RemotePort"

$plinkCommand = Get-Command plink.exe -ErrorAction SilentlyContinue
if (-not $plinkCommand) {
    Write-Marker "RECOVER_FAIL:PLINK_NOT_FOUND"
    Write-Host "Install PuTTY and ensure plink.exe is on PATH."
    exit 1
}
$script:PlinkPath = $plinkCommand.Source
Write-Marker "PLINK_OK:$PlinkPath"

Write-Marker "SSH_AUTH_CHECK_BEGIN"
$auth = Invoke-PlinkCommand -RemoteCommand "echo PI_OK"
if ($auth.ExitCode -ne 0 -or $auth.Output -notmatch "PI_OK") {
    Write-Host $auth.Output
    Fail-Recovery -Reason "SSH_AUTH_FAILED"
}
Write-Marker "SSH_AUTH_OK:PI_OK"

$remoteSetupTemplate = @'
sh -lc 'set -e;
cd __REMOTE_PROJECT_PATH__;
PYBIN="./.venv/bin/python";
if [ ! -x "$PYBIN" ]; then PYBIN="python3"; fi;
"$PYBIN" -m py_compile webapp.py calibration_intrinsics.py;
echo REMOTE_COMPILE_OK;
pkill -f "uvicorn webapp:app" >/dev/null 2>&1 || true;
pkill -f "python3 -m uvicorn webapp:app" >/dev/null 2>&1 || true;
pkill -f ".venv/bin/python -m uvicorn webapp:app" >/dev/null 2>&1 || true;
nohup "$PYBIN" -m uvicorn webapp:app --host 127.0.0.1 --port __REMOTE_PORT__ >__REMOTE_LOG_PATH__ 2>&1 < /dev/null &
sleep 3;
ss -tlnp | grep -q ":__REMOTE_PORT__";
echo REMOTE_LISTEN_OK;
REMOTE_HTTP_CODE=$(curl -sS -m 8 -o /dev/null -w "%{http_code}" "http://127.0.0.1:__REMOTE_PORT__/api/system/mode");
echo REMOTE_HTTP_CODE=$REMOTE_HTTP_CODE;
curl -sS -m 8 "http://127.0.0.1:__REMOTE_PORT__/api/system/mode";
echo'
'@

$remoteSetup = $remoteSetupTemplate.Replace("__REMOTE_PROJECT_PATH__", (ConvertTo-ShDoubleQuotedLiteral -Value $RemoteProjectPath))
$remoteSetup = $remoteSetup.Replace("__REMOTE_PORT__", [string]$RemotePort)
$remoteSetup = $remoteSetup.Replace("__REMOTE_LOG_PATH__", (ConvertTo-ShDoubleQuotedLiteral -Value $RemoteLogPath))

Write-Marker "REMOTE_RECOVERY_BEGIN"
$remote = Invoke-PlinkCommand -RemoteCommand $remoteSetup
if ($remote.ExitCode -ne 0) {
    Write-Host $remote.Output
    Fail-Recovery -Reason "REMOTE_SETUP_FAILED"
}
Write-Host $remote.Output
if ($remote.Output -notmatch "REMOTE_COMPILE_OK") {
    Fail-Recovery -Reason "REMOTE_COMPILE_CHECK_FAILED"
}
if ($remote.Output -notmatch "REMOTE_LISTEN_OK") {
    Fail-Recovery -Reason "REMOTE_LISTENER_CHECK_FAILED"
}
if ($remote.Output -notmatch "REMOTE_HTTP_CODE=200") {
    Fail-Recovery -Reason "REMOTE_API_CHECK_FAILED"
}
Write-Marker "REMOTE_RECOVERY_OK"

Write-Marker "LOCAL_TUNNEL_CLEAN_BEGIN"
$stale = Get-LocalTunnelPlinkProcesses
if ($stale) {
    $staleIds = $stale.ProcessId
    Stop-Process -Id $staleIds -Force -ErrorAction SilentlyContinue
    Write-Marker "LOCAL_TUNNEL_CLEANED:$($staleIds -join ',')"
} else {
    Write-Marker "LOCAL_TUNNEL_NONE"
}

$tunnelArgs = @(
    "-ssh",
    "-N",
    "-batch",
    "-pw", $Password,
    "-hostkey", $HostKey,
    "-L", "$LocalPort`:127.0.0.1:$RemotePort",
    $Pi
)

Write-Marker "LOCAL_TUNNEL_START_BEGIN"
$tunnelProc = Start-Process -FilePath $PlinkPath -ArgumentList $tunnelArgs -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2
if ($tunnelProc.HasExited) {
    Write-Marker "LOCAL_TUNNEL_START_EXITED:$($tunnelProc.ExitCode)"
    Fail-Recovery -Reason "LOCAL_TUNNEL_START_FAILED"
}
Write-Marker "LOCAL_TUNNEL_START_OK:PID=$($tunnelProc.Id)"

Write-Marker "LOCAL_API_CHECK_BEGIN"
$localUrl = "http://127.0.0.1:$LocalPort/api/system/mode"
$localCode = (& curl.exe -sS -m 8 -o NUL -w "%{http_code}" $localUrl).Trim()
if ($localCode -ne "200") {
    Write-Marker "LOCAL_HTTP_CODE=$localCode"
    try {
        $localBody = & curl.exe -sS -m 8 $localUrl
        Write-Host "LOCAL_RESPONSE=$localBody"
    } catch {
        Write-Host "LOCAL_RESPONSE_ERROR=$($_.Exception.Message)"
    }
    Fail-Recovery -Reason "LOCAL_API_CHECK_FAILED"
}

$localBodyOk = (& curl.exe -sS -m 8 $localUrl).Trim()
Write-Marker "LOCAL_HTTP_CODE=200"
Write-Host "LOCAL_RESPONSE=$localBodyOk"

Write-Marker "RECOVER_OK"
exit 0
