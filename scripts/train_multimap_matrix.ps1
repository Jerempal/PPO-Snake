[CmdletBinding()]
param([int[]]$Seeds = @(42, 43, 44), [switch]$Execute)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    $python = Join-Path $projectRoot ".venv/Scripts/python.exe"
    foreach ($seed in $Seeds) {
        foreach ($arm in @("direct", "trajectory")) {
            if (Test-Path -LiteralPath "runs/multimap-$arm-$seed") {
                throw "Run already exists: multimap-$arm-$seed"
            }
        }
    }
    foreach ($seed in $Seeds) {
        foreach ($arm in @("direct", "trajectory")) {
            $flag = if ($arm -eq "direct") { "--no-trajectory-restart" } else { "--trajectory-restart" }
            $arguments = @("-m", "snake_rl.train", "--config", "configs/multimap.toml",
                "--seed", "$seed", $flag, "--run-name", "multimap-$arm-$seed")
            Write-Host ("python " + ($arguments -join " "))
            if ($Execute) {
                & $python @arguments
                if ($LASTEXITCODE -ne 0) { throw "Failed: multimap-$arm-$seed" }
            }
        }
    }
} finally { Pop-Location }
