# Single training run: all .txt files in data/ (Shakespeare + 8 novels), 20k iters.
# Run detached so closing the chat/terminal doesn't kill it:
#   Start-Process powershell -ArgumentList '-NoExit -File run_training.ps1'
Set-Location -Path $PSScriptRoot

Write-Host "=== Training: books combo 20k (log: train.log) ==="
$extra = if (Test-Path "checkpoints\model.pt") { "--resume" } else { "" }
python -u train.py --data data --max-iters 20000 --checkpoint-dir checkpoints $extra *> train.log

Write-Host "=== Done. Sample: ==="
python generate.py --checkpoint checkpoints/model.pt --prompt "It was a bright" --max-new-tokens 300
