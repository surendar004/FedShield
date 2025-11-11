param(
    [int]$Count = 3
)

$venv = "C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield\fedshield_env\Scripts\python.exe"
$root = "C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield"
$logdir = Join-Path $root "logs"
if (!(Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }

for ($i=1; $i -le $Count; $i++) {
    $id = "flwr_client_$i"
    $out = Join-Path $logdir "flwr_client_$i.out"
    $err = Join-Path $logdir "flwr_client_$i.err"
    Start-Process -FilePath $venv -ArgumentList "`"$root\client\start_flwr_client.py`" $id" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
    Write-Output "Started flwr client $id (logs: $out, $err)"
}
