; GIS Co-Scientist desktop installer (Inno Setup 6).
; Build with desktop\build\build_installer.ps1, which stages the app first.
; Installs per user (no admin) to %LOCALAPPDATA%\Programs\GIS Co-Scientist, so the app can
; keep its conversations, uploads and outputs beside the code as it does today.

#define AppName "GIS Co-Scientist"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef StageDir
  #define StageDir "..\..\build\stage"
#endif
#define AppUserModelID "GIBD.GISCoScientist.Desktop"
; Single-quoted so the preprocessor keeps the doubled quotes that [Icons]/[Run] need around the path
#define Launch '-E -s ""{app}\app\desktop\launcher.py""'

[Setup]
AppId={{6B0E3F5C-2C1F-4D8B-9E77-3A4F6C1D2B90}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=GIBD Lab, Penn State
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\..\dist
OutputBaseFilename=GIS-CoScientist-Setup-{#AppVersion}
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\app\desktop\assets\app.ico
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[InstallDelete]
; Upgrades: replace the runtime and code folders wholesale so removed packages and modules
; don't linger. WebUI is overwritten in place because it also holds the user's data.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app\agents"
Type: filesandordirs; Name: "{app}\app\utils"
Type: filesandordirs; Name: "{app}\app\skills"
Type: filesandordirs; Name: "{app}\app\spatial_kb"
Type: filesandordirs; Name: "{app}\app\mapper"

[Files]
; Never ship anything a local test run of the stage may have written.
Source: "{#StageDir}\runtime\*"; DestDir: "{app}\runtime"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StageDir}\app\*"; DestDir: "{app}\app"; Excludes: "__pycache__,\desktop\logs,\desktop\profile,*.db,\WebUI\uploads,\WebUI\conversations,\WebUI\conversation_DBs,\WebUI\Outputs,\WebUI\cache"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#Launch}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\desktop\assets\app.ico"; AppUserModelID: "{#AppUserModelID}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#Launch}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\desktop\assets\app.ico"; AppUserModelID: "{#AppUserModelID}"; Tasks: desktopicon

[Run]
; Precompile Python once so the first launch is fast (the download ships source only).
Filename: "{app}\runtime\python.exe"; Parameters: "-E -s -m compileall -q -j 0 ""{app}\runtime\Lib"" ""{app}\app"""; StatusMsg: "Preparing {#AppName} for first launch (about a minute)..."; Flags: runhidden waituntilterminated
Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#Launch}"; WorkingDir: "{app}\app"; Description: "Launch {#AppName}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Compiled caches created after install. Conversations, uploads and outputs under
; {app}\app\WebUI and the window profile are kept, so reinstalling restores them.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app\agents"
Type: filesandordirs; Name: "{app}\app\utils"
Type: filesandordirs; Name: "{app}\app\mapper"
Type: filesandordirs; Name: "{app}\app\desktop\logs"

[Code]
function WebView2Installed(): Boolean;
var
  Version: String;
begin
  Result :=
    (RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0'));
end;

function InitializeSetup(): Boolean;
var
  ErrorCode: Integer;
begin
  Result := True;
  if not WebView2Installed() then
  begin
    if MsgBox('{#AppName} needs the Microsoft Edge WebView2 Runtime, which is not installed on this PC.' + #13#10#13#10 +
              'It is free and built into Windows 11. Open the download page now? You can run this installer again afterwards.',
              mbConfirmation, MB_YESNO) = IDYES then
      ShellExec('open', 'https://developer.microsoft.com/microsoft-edge/webview2/', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
    Result := False;
  end;
end;
