param(
    [ValidateSet("debug", "release")]
    [string]$Configuration = "release"
)

$ErrorActionPreference = "Stop"

$target    = "i686-pc-windows-msvc"
$root      = $PSScriptRoot
$buildDir  = Join-Path $root "build"
$dll       = Join-Path $root "target\$target\$Configuration\falke.dll"
$assets    = Join-Path $root "assets"

if ($Configuration -eq "release") {
    cargo build --release --target $target
} else {
    cargo build --target $target
}
if ($LASTEXITCODE -ne 0) {
    throw "cargo build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $dll)) {
    throw "falke.dll not found at $dll after a successful build"
}

# Start from an empty build folder so a stale file cannot ship.
if (Test-Path $buildDir) {
    Get-ChildItem -Path $buildDir -Force | Remove-Item -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Copy-Item -Path (Join-Path $root "CHANGELOG.md") -Destination $buildDir -Force
Copy-Item -Path (Join-Path $root "README.md")    -Destination $buildDir -Force

foreach ($game in @("thugpro", "better4")) {
    $gameDir = Join-Path $buildDir $game
    New-Item -ItemType Directory -Path (Join-Path $gameDir "plugins") -Force | Out-Null

    Copy-Item -Path $dll -Destination (Join-Path $gameDir "plugins\falke.asi") -Force
    Copy-Item -Path (Join-Path $assets "dinput8.dll") -Destination (Join-Path $gameDir "dinput8.dll") -Force
}

# Better4 needs the d3d8 wrapper, THUG Pro does not.
Copy-Item -Path (Join-Path $assets "d3d8.dll") -Destination (Join-Path $buildDir "better4\d3d8.dll") -Force

Write-Host "Packaged $Configuration build into $buildDir"
Get-ChildItem -Path $buildDir -Recurse -File | ForEach-Object {
    Write-Host ("  " + $_.FullName.Substring($buildDir.Length + 1))
}
