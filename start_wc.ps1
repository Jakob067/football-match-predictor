$env:FOOTBALL_DATA_API_TOKEN = [Environment]::GetEnvironmentVariable(
    "FOOTBALL_DATA_API_TOKEN",
    "User"
)

if (-not $env:FOOTBALL_DATA_API_TOKEN) {
    Write-Error "FOOTBALL_DATA_API_TOKEN ist nicht als Benutzerumgebungsvariable gesetzt."
    exit 1
}

py wc_live_website.py
