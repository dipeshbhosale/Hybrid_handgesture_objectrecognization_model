Get-ChildItem -File | Where-Object {$_.Length -eq 0} | Move-Item -Destination "empty_files_cleanup" -Force
