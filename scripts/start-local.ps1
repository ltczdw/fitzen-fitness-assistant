param(
    [string]$BackendHost = '',
    [int]$BackendPort = 0,
    [int]$FrontendPort = 5173,
    [switch]$WithMcp,
    [switch]$BackendOnly,
    [switch]$DisableChatRecords,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $projectRoot 'backend'
$frontend = Join-Path $projectRoot 'frontend'
$python = Join-Path $backend '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create backend/.venv and install backend/requirements.txt first.' }
if (!(Test-Path -LiteralPath (Join-Path $backend '.env'))) { throw 'Copy backend/.env.example to backend/.env and configure DATABASE_URL first.' }
Push-Location $backend
try {
    $configText = & $python -c 'import json; from app.core.config import settings as s; print(json.dumps(dict(host=s.backend_host,port=s.backend_port,mcp_host=s.mcp_host,mcp_port=s.mcp_port)))'
    if ($LASTEXITCODE -ne 0) { throw 'Backend configuration is invalid.' }
    $config = $configText | ConvertFrom-Json
} finally { Pop-Location }
if (!$BackendHost) { $BackendHost = $config.host }
if (!$BackendPort) { $BackendPort = $config.port }
if ($BackendPort -lt 1 -or $BackendPort -gt 65535 -or $FrontendPort -lt 1 -or $FrontendPort -gt 65535) { throw 'Ports must be 1..65535.' }
$ports = @($BackendPort)
if (!$BackendOnly) { $ports += $FrontendPort }
if ($WithMcp) { $ports += $config.mcp_port }
if (($ports | Select-Object -Unique).Count -ne $ports.Count) { throw 'Backend, frontend and MCP ports must differ.' }
$locks = @()
try {
    # Protect simultaneous runs of this script, then reject existing listeners.
    foreach ($port in ($ports | Sort-Object)) {
        $mutex = [Threading.Mutex]::new($false, "Localitzen-start-$port")
        if (!$mutex.WaitOne(0)) { $mutex.Dispose(); throw "Another startup is in progress on port $port." }
        $locks += $mutex
        $occupied = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        if ($occupied) { throw "Port $port is occupied (PID $($occupied[0].OwningProcess)). No second instance was started." }
    }
    if (!$BackendOnly) {
        $node = (Get-Command node.exe -ErrorAction Stop).Source
        $vite = Join-Path $frontend 'node_modules/vite/bin/vite.js'
        if (!(Test-Path -LiteralPath $vite)) { throw 'Run npm ci in frontend first.' }
    }
    if ($CheckOnly) { Write-Output 'Configuration, dependencies and requested ports are ready.'; return }
    $logs = Join-Path $projectRoot 'runtime-logs'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $previousWriteFlag = $env:REACT_WRITE_ENABLED
    try {
        $env:REACT_WRITE_ENABLED = if ($DisableChatRecords) { 'false' } else { 'true' }
        $apiProcess = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','app.main:app','--host',$BackendHost,'--port',"$BackendPort") -WorkingDirectory $backend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'backend.out.log') -RedirectStandardError (Join-Path $logs 'backend.err.log')
    } finally { $env:REACT_WRITE_ENABLED = $previousWriteFlag }
    $probeHost = if ($BackendHost -eq '0.0.0.0') { '127.0.0.1' } else { $BackendHost }
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($apiProcess.HasExited) { throw "Backend exited; inspect runtime-logs/backend.err.log." }
        try { $health = Invoke-RestMethod "http://${probeHost}:${BackendPort}/api/health" -TimeoutSec 2; if ($health.ready) { $ready = $true; break } } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (!$ready) { throw "Backend readiness timed out; inspect logs. PID $($apiProcess.Id) was not silently restarted." }
    Write-Output "Backend PID $($apiProcess.Id): http://${probeHost}:${BackendPort}"
    if (!$DisableChatRecords -and !$health.capabilities.chat_record.available) {
        Write-Warning 'Chat records are enabled but not ready; inspect /api/health capabilities. No fallback was enabled.'
    }
    if ($WithMcp) {
        $mcpProcess = Start-Process -FilePath $python -ArgumentList @('-m','app.services.mcp.server') -WorkingDirectory $backend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'mcp.out.log') -RedirectStandardError (Join-Path $logs 'mcp.err.log')
        Write-Output "MCP PID $($mcpProcess.Id); inspect runtime-logs/mcp.err.log for readiness."
    }
    if (!$BackendOnly) {
        $previousTarget = $env:VITE_API_TARGET
        try {
            $env:VITE_API_TARGET = "http://${probeHost}:${BackendPort}"
            $viteArgument = '"' + $vite + '"'
            $webProcess = Start-Process -FilePath $node -ArgumentList @($viteArgument,'--host','127.0.0.1','--port',"$FrontendPort",'--strictPort') -WorkingDirectory $frontend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'frontend.out.log') -RedirectStandardError (Join-Path $logs 'frontend.err.log')
            Write-Output "Frontend PID $($webProcess.Id): http://127.0.0.1:${FrontendPort}"
        } finally { $env:VITE_API_TARGET = $previousTarget }
    }
} finally {
    foreach ($mutex in $locks) { $mutex.ReleaseMutex(); $mutex.Dispose() }
}
