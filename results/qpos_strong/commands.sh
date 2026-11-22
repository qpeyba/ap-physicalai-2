export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 MUJOCO_GL=egl
python3 train.py --device cuda:0 --num_envs 2048 --max_iters 1001 --seed 1 --save_interval 50 --exp_name qpos_strong --robot_target_qpos 30.0 > logs/qpos_strong/training.log 2>&1
python3 inference.py --device cuda:0 --checkpoint logs/qpos_strong/model_1000.pt --env_config logs/qpos_strong/env_config.json --episodes 10 --seed 42 --obs_noise 0.0 --record results/qpos_strong/video.mp4 > results/qpos_strong/evaluation.log 2>&1
