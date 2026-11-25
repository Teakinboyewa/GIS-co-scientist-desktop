# Builds dist\GIS-CoScientist-Setup-<version>.exe
#   1. stages a self-contained Python + app into build\stage (stage_app.py)
#   2. compiles desktop\build\installer.iss with the portable Inno Setup in desktop\tools
# Usage: powershell -ExecutionPolicy Bypass -File desktop\build\build_installer.ps1 [-Version 0.1.2] [-SkipStage]
param(
    [string]$Version = "0.1.2",
    [string]$Venv = "$env:USERPROFILE\venvs\agm_venv",
    [switch]$SkipStage
)
$ErrorActionPreference = "Stop"
$buildDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$desk = Split-Path -Parent (Split-Path -Parent $buildDir)

if (-not $SkipStage) {
    $env:GIS_DESKTOP_VENV = $Venv
    & "$Venv\Scripts\python.exe" "$buildDir\stage_app.py"
    if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
}

$iscc = Get-ChildItem -Recurse "$desk\desktop\tools" -Filter ISCC.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $iscc) { throw "ISCC.exe not found under desktop\tools. See README_DESKTOP.md for how to fetch Inno Setup." }

# The OneDrive path is so deep that some staged files exceed Windows' 260-character limit,
# which Inno Setup can't read. Compile through a temporary drive letter mapped to this folder.
$used = (Get-PSDrive -PSProvider FileSystem).Name
$letter = @('G','K','Q','W','X','Y') | Where-Object { $used -notcontains $_ } | Select-Object -First 1
if (-not $letter) { throw "No free drive letter for the short build path" }
subst "${letter}:" "$desk"
try {
    $isccShort = $iscc.FullName.Replace($desk, "${letter}:")
    & $isccShort "/DAppVersion=$Version" "/Q" "${letter}:\desktop\build\installer.iss"
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
} finally {
    subst "${letter}:" /D
}

$exe = Get-Item "$desk\dist\GIS-CoScientist-Setup-$Version.exe"
"Built $($exe.FullName) ($([math]::Round($exe.Length / 1MB)) MB)"
