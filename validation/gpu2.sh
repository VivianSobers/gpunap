#!/bin/bash
# Cross-driver subset on gpu2 (driver 580, torch 2.14 venv). Run from ~/gpunap_validation.
export PY=~/.venv/bin/python
$PY bench_size.py 1 4 16 > logs/bench_size.out 2>&1
$PY faults.py errors short_far short_near handoff killmid ctlkill_restore sigint sigint_control sigstop \
    state_blocking child_cuda two_jobs cgroup leak uvm ipc > logs/faults_main.out 2>&1
for c in "--kind mlp --pauses 20" "--kind amp_tf --pauses 20" "--kind compile --pauses 20 --early" \
         "--kind cudagraph --pauses 20" "--kind loader --pauses 20" "--kind cnn --pauses 20" "--kind threads --pauses 20" \
         "--kind mlp --pauses 20 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --tag mlp_expandable_p20"; do
  eval "$PY run_case.py $c" >> logs/matrix.out 2>&1
done
echo W2 DONE >> logs/matrix.out
