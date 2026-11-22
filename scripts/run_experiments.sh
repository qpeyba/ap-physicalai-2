#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/pandapickcube-matplotlib

experiments=(max_iters_500 box_target_strong qpos_strong learning_rate_3e4)
test -s results/hypotheses.md
mkdir -p results logs
for experiment in "${experiments[@]}"; do
    if [[ -e "logs/$experiment" || -e "results/$experiment" ]]; then
        printf 'Выходной каталог уже существует: %s\n' "$experiment" >&2
        exit 1
    fi
done

python3 -V > results/python_version.txt
python3 -m pip freeze > results/requirements-observed.txt
diff -u <(LC_ALL=C sort -f requirements-lock.txt) \
    <(LC_ALL=C sort -f results/requirements-observed.txt) > results/dependency_diff.txt || true
python3 -m pip check > results/pip_check.txt || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu.csv
python3 - <<'PY'
import jax
import torch

assert torch.cuda.is_available(), "PyTorch CUDA unavailable"
assert jax.devices("gpu"), "JAX GPU unavailable"
print(torch.cuda.get_device_name(), jax.devices())
PY

failed=0
for experiment in "${experiments[@]}"; do
    max_iters=1001
    override=()
    case "$experiment" in
        max_iters_500) max_iters=500 ;;
        box_target_strong) override=(--box_target 40.0) ;;
        qpos_strong) override=(--robot_target_qpos 30.0) ;;
        learning_rate_3e4) override=(--learning_rate 0.0003) ;;
    esac

    log_dir="logs/$experiment"
    result_dir="results/$experiment"
    mkdir -p "$log_dir" "$result_dir"
    train=(python3 train.py --device cuda:0 --num_envs 2048 --max_iters "$max_iters"
           --seed 1 --save_interval 50 --exp_name "$experiment" "${override[@]}")
    inference=(python3 inference.py --device cuda:0
               --checkpoint "$log_dir/model_$((max_iters - 1)).pt"
               --env_config "$log_dir/env_config.json"
               --episodes 10 --seed 42 --obs_noise 0.0 --record "$result_dir/video.mp4")

    {
        printf 'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 MUJOCO_GL=egl\n'
        printf '%q ' "${train[@]}"
        printf '> %s 2>&1\n' "$log_dir/training.log"
        printf '%q ' "${inference[@]}"
        printf '> %s 2>&1\n' "$result_dir/evaluation.log"
    } > "$result_dir/commands.sh"
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$experiment: обучение"
    if ! "${train[@]}" > "$log_dir/training.log" 2>&1; then
        printf '%s\n' "$experiment: ошибка обучения" >&2
        tail -30 "$log_dir/training.log" >&2
        failed=1
        continue
    fi
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$experiment: инференс"
    if ! "${inference[@]}" > "$result_dir/evaluation.log" 2>&1; then
        printf '%s\n' "$experiment: ошибка инференса" >&2
        tail -30 "$result_dir/evaluation.log" >&2
        failed=1
    fi
done
exit "$failed"
