# Creates "GIS Co-Scientist.lnk" in the GIS_Co_Scientist_Desk folder: a no-console
# launcher with the app icon. Re-run it if the folder or the venv moves.
param(
    [string]$Venv = "$env:USERPROFILE\venvs\agm_venv"
)
$desktopDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $desktopDir
$pythonw = Join-Path $Venv "Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found at $pythonw" }

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $root "GIS Co-Scientist.lnk"))
$lnk.TargetPath = $pythonw
$lnk.Arguments = '"' + (Join-Path $desktopDir "launcher.py") + '"'
$lnk.WorkingDirectory = $root
$lnk.IconLocation = (Join-Path $desktopDir "assets\app.ico") + ",0"
$lnk.Description = "GIS Co-Scientist desktop app"
$lnk.Save()
Write-Output "Created $(Join-Path $root 'GIS Co-Scientist.lnk')"
