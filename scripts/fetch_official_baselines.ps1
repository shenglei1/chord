$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ExternalRoot = Join-Path $ProjectRoot "external"
$MohitoRoot = Join-Path $ExternalRoot "mohito-public"
$MohitoCommit = "105ddccbacc92953a7aa6bd0d19f8ec299f7476d"
$Marker = Join-Path $MohitoRoot "UPSTREAM_SOURCE.txt"
$RequiredPackage = Join-Path $MohitoRoot "wildfire\mohito\mohito"

if ((Test-Path -LiteralPath $Marker) -and (Test-Path -LiteralPath $RequiredPackage)) {
    $MarkerText = Get-Content -LiteralPath $Marker -Raw
    if ($MarkerText -notmatch [regex]::Escape($MohitoCommit)) {
        throw "The vendored MOHITO source marker does not match $MohitoCommit."
    }
    Write-Host "MOHITO source is already pinned at $MohitoCommit"
    exit 0
}

if (Test-Path -LiteralPath $MohitoRoot) {
    throw "A MOHITO directory exists without a valid source marker. Remove or archive it before fetching."
}

New-Item -ItemType Directory -Force -Path $ExternalRoot | Out-Null
$FetchRoot = Join-Path $ExternalRoot (".mohito-fetch-" + [guid]::NewGuid().ToString("N"))

try {
    git clone https://github.com/oasys-mas/mohito-public.git $FetchRoot
    git -c "safe.directory=$($FetchRoot.Replace('\','/'))" -C $FetchRoot checkout --detach $MohitoCommit
    Remove-Item -LiteralPath (Join-Path $FetchRoot ".git") -Recurse -Force
    @(
        "Repository: https://github.com/oasys-mas/mohito-public.git"
        "Commit: $MohitoCommit"
        "Vendoring: complete upstream working tree with version-control metadata removed"
    ) | Set-Content -LiteralPath (Join-Path $FetchRoot "UPSTREAM_SOURCE.txt") -Encoding UTF8
    Move-Item -LiteralPath $FetchRoot -Destination $MohitoRoot
    Write-Host "MOHITO source vendored at $MohitoCommit"
}
finally {
    if (Test-Path -LiteralPath $FetchRoot) {
        Remove-Item -LiteralPath $FetchRoot -Recurse -Force
    }
}
