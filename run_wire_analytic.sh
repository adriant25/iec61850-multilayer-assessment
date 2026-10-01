#!/bin/bash
# Re-run every analytical study with the on-wire frame sizes (config.ETH_WIRE_OVERHEAD).
set -e
PY=./.venv/Scripts/python.exe
for step in "simulation.py" "sensitivity.py" "event_scope.py" "congestion_threshold.py" "maintainability.py" "robustness.py t1" "robustness.py cap" "nc_bound.py"; do
  echo "=== $step $(date +%T)"
  $PY -u $step > "logs_wire_$(echo $step | tr ' .' '__').log" 2>&1
done
echo "=== done $(date +%T)"
