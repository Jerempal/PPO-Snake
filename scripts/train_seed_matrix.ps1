[CmdletBinding()]
param(
    [string]$Config = "configs/ppo_8.toml",

    [int[]]$Seeds = @(42, 43, 44),

    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",

    [string]$Prefix = "ppo",

    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    foreach ($seed in $Seeds) {
        $runName = "$Prefix-seed$seed"
        if (Test-Path -LiteralPath (Join-Path "runs" $runName)) {
            throw "Run already exists: $runName"
        }
        $arguments = @("run", "snake-train", "--config", $Config,
            "--device", $Device, "--seed", "$seed", "--run-name", $runName)
        Write-Host ("uv " + (($arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join " "))
        if ($Execute) {
            & uv @arguments
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $runName." }
        }
    }
} finally {
    Pop-Location
}
