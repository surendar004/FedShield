$venv = "C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield\fedshield_env\Scripts\python.exe"
$root = "C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield"
$logdir = Join-Path $root "logs"
if (!(Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$out = Join-Path $logdir "flwr_server.out"
$err = Join-Path $logdir "flwr_server.err"
Start-Process -FilePath $venv -ArgumentList "`"$root\server\federated_server.py`"" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
Write-Output "Started federated server (logs: $out, $err)"