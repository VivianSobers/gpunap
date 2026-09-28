#!/bin/bash
# Correctness matrix: three lanes in parallel; each case = two reference runs + one paused run.
PY=${PY:-python3}
lane() { for c in "$@"; do eval "$PY run_case.py $c"; done; }
lane "--kind mlp --pauses 1" "--kind mlp --pauses 20" "--kind mlp --pauses 20 --seed 1" \
     "--kind mlp --pauses 20 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --tag mlp_expandable_p20" \
     "--kind mlp --pauses 20 --env PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync --tag mlp_mallocasync_p20" \
     "--kind cudagraph --pauses 1" "--kind cudagraph --pauses 20" > logs/matrix_lane1.out 2>&1 &
lane "--kind amp_tf --pauses 1" "--kind amp_tf --pauses 20" "--kind amp_tf --pauses 20 --seed 1" \
     "--kind amp_tf --pauses 20 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --tag amp_tf_expandable_p20" \
     "--kind amp_tf --pauses 5 --ballast_gb 12 --steps 300 --tag amp_tf_ballast12_p5" > logs/matrix_lane2.out 2>&1 &
lane "--kind cnn --pauses 1" "--kind cnn --pauses 20" "--kind compile --pauses 1 --early" \
     "--kind compile --pauses 20 --early" "--kind loader --pauses 1" "--kind loader --pauses 20" \
     "--kind threads --pauses 1" "--kind threads --pauses 20" "--kind hf --steps 40 --pauses 1" \
     "--kind hf --steps 40 --pauses 10" > logs/matrix_lane3.out 2>&1 &
wait
echo MATRIX DONE
