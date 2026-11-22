export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 MUJOCO_GL=egl
python3 train.py --device cuda:0 --num_envs 2048 --max_iters 500 --seed 1 --save_interval 50 --exp_name max_iters_500 > logs/max_iters_500/training.log 2>&1
python3 inference.py --device cuda:0 --checkpoint logs/max_iters_500/model_499.pt --env_config logs/max_iters_500/env_config.json --episodes 10 --seed 42 --obs_noise 0.0 --record results/max_iters_500/video.mp4 > results/max_iters_500/evaluation.log 2>&1
