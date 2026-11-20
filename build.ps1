param(
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$finder = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (!(Test-Path -LiteralPath $finder)) {
    throw '找不到 vswhere.exe，请检查 Visual Studio Installer。'
}

$msbuild = & $finder -latest -products * -requires Microsoft.Component.MSBuild `
    -find 'MSBuild\Current\Bin\amd64\MSBuild.exe' | Select-Object -First 1
if (!$msbuild) {
    throw '请安装 Visual Studio 的 .NET 桌面开发与 C++ 桌面开发工作负载。'
}

$solution = Join-Path $PSScriptRoot 'PrintInspector.sln'
& $msbuild $solution /restore /m:1 /nr:false "/p:Configuration=$Configuration" /p:Platform=x64 /v:minimal
exit $LASTEXITCODE
