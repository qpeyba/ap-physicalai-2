"""Проверка конфигов, экспорт TensorBoard-метрик и сравнение пяти запусков."""

import csv
import json
from collections import deque
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
RESULTS = ROOT / "results"
RUNS = ("baseline", "max_iters_500", "box_target_strong", "qpos_strong", "learning_rate_3e4")
REWARD = "Train/mean_reward"
LEARNING_RATE = "Loss/learning_rate"
COLORS = dict(zip(RUNS, ("#1e293b", "#007d85", "#9b4529", "#6b46ac", "#2457ad")))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_scalars(directory):
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    scalars = {}
    for tag in sorted(events.Tags()["scalars"]):
        if not tag.endswith("/time"):
            series = {event.step: event.value for event in events.Scalars(tag)}
            if not all(isfinite(value) for value in series.values()):
                raise ValueError(f"Нечисловые значения в {directory}: {tag}")
            scalars[tag] = dict(sorted(series.items()))
    return scalars


def rolling_mean(series, width=25):
    window = deque(maxlen=width)
    smoothed = {}
    for index, value in series.items():
        if window and index != previous_index + 1:
            window.clear()
        window.append(value)
        if len(window) == width:
            smoothed[index] = mean(window)
        previous_index = index
    return smoothed


def tail_band_start(smoothed, center):
    """Первая точка полосы ±5% от среднего хвоста с 50 точками подряд внутри."""
    streak = 0
    previous = None
    for index, value in smoothed.items():
        if abs(value - center) > abs(center) * 0.05:
            streak = 0
        elif previous is None or index == previous + 1:
            streak += 1
        else:
            streak = 1
        if streak == 50:
            return index - 49
        previous = index
    return None


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_scalars(path, scalars):
    indices = sorted({step for series in scalars.values() for step in series})
    write_csv(path, [{"iteration": step, **{tag: series.get(step) for tag, series in scalars.items()}}
                     for step in indices])


def verify_configs(name, baseline_info, baseline_ppo, run_dir, evaluation):
    info = read_json(run_dir / "run_info.json")
    env_config = read_json(run_dir / "env_config.json")
    ppo_config = read_json(run_dir / "config.json")
    expected_env = baseline_info["environment_config"]
    expected_ppo = baseline_ppo
    if name in ("box_target_strong", "qpos_strong"):
        key, weight = ("box_target", 40.0) if name == "box_target_strong" else ("robot_target_qpos", 30.0)
        expected_env = {
            **expected_env,
            "reward_config": {"scales": {**expected_env["reward_config"]["scales"], key: weight}},
        }
    if name == "learning_rate_3e4":
        expected_ppo = {**expected_ppo, "algorithm": {**expected_ppo["algorithm"], "learning_rate": 0.0003}}
    if name == "max_iters_500":
        expected_ppo = {**expected_ppo, "max_iterations": 500}

    assert env_config == expected_env == evaluation["environment_config"], name
    assert ppo_config == expected_ppo, name
    assert info["num_envs"] == baseline_info["num_envs"], name
    assert info["training_seed"] == baseline_info["seed"] == ppo_config["seed"], name
    assert info["max_iterations"] == expected_ppo["max_iterations"], name
    assert info["steps_per_env_per_iteration"] == ppo_config["num_steps_per_env"], name
    assert info["device"] == "cuda:0" and info["resume"] is None, name
    final_index = info["max_iterations"] - 1
    expected_indices = set(range(0, info["max_iterations"], ppo_config["save_interval"])) | {final_index}
    actual_indices = {int(path.stem.split("_")[1]) for path in run_dir.glob("model_*.pt") if path.stat().st_size}
    # The submission includes the final checkpoint for each policy.
    assert actual_indices in ({final_index}, expected_indices), (name, actual_indices, expected_indices)
    assert evaluation["checkpoint"] == f"logs/{name}/model_{final_index}.pt", name
    return info


def episode_metrics(evaluation, name):
    episodes = evaluation["episodes"]
    assert [episode["seed"] for episode in episodes] == list(range(42, 52)), name
    assert evaluation["environment"] == "PandaPickCube" and evaluation["seed"] == 42, name
    assert evaluation["obs_noise"] == 0 and evaluation["cube_x"] is None and evaluation["cube_y"] is None, name
    assert evaluation["success_distance_threshold_m"] == 0.05, name
    assert evaluation["grasp_z_threshold_m"] == 0.20, name
    for episode in episodes:
        error = episode["final_position_error_m"]
        assert isfinite(error) and error >= 0 and isfinite(episode["reward"]), (name, episode)
        assert episode["success"] == (error < 0.05), (name, episode)
        if "max_cube_z_m" in episode:
            assert episode["grasped"] == (episode["max_cube_z_m"] > 0.2), (name, episode)
    assert evaluation["success_rate"] == sum(episode["success"] for episode in episodes) / len(episodes), name
    assert evaluation["grasp_rate"] == sum(episode["grasped"] for episode in episodes) / len(episodes), name
    assert abs(evaluation["mean_reward"] - mean(episode["reward"] for episode in episodes)) < 1e-6, name
    return episodes


def summarize(name, samples, evaluation, iterations, seconds):
    rewards = samples[REWARD]
    last = list(rewards.values())[-100:]
    smoothed = rolling_mean(rewards)
    assert max(samples[LEARNING_RATE]) == max(rewards) == iterations - 1, name
    episodes = episode_metrics(evaluation, name)
    return {
        "experiment": name,
        "iterations": iterations,
        "last_iteration": iterations - 1,
        "environment_steps": 2048 * 40 * iterations,
        "final_training_mean_reward": rewards[iterations - 1],
        "successes": sum(episode["success"] for episode in episodes),
        "grasped": sum(episode["grasped"] for episode in episodes),
        "episodes": len(episodes),
        "seconds": seconds,
        "evaluation_mean_reward": evaluation["mean_reward"],
        "mean_final_error_m": mean(episode["final_position_error_m"] for episode in episodes),
        "tail_100_reward_mean": mean(last),
        "tail_100_reward_stdev": stdev(last),
        "tail_50_minus_previous_50_mean": mean(last[50:]) - mean(last[:50]),
        "tail_band_start": tail_band_start(smoothed, mean(last)),
        "rolling_mean_25_at_1000_iteration": next((i for i, v in smoothed.items() if v >= 1000), None),
        "first_logged_learning_rate": next(iter(samples[LEARNING_RATE].values())),
        "final_logged_learning_rate": samples[LEARNING_RATE][iterations - 1],
    }


def format_axis(ax, title, ylabel):
    ax.set(title=title, xlabel="Итерация PPO (с 0)", ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22)


def plot_reward(ax, name, series):
    color = COLORS[name]
    average = rolling_mean(series)
    ax.plot(list(series), list(series.values()), color=color, alpha=0.20, linewidth=0.8)
    ax.plot(list(average), list(average.values()), color=color, linewidth=2, label=name)


def plot_comparison(series):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    for name in ("baseline", "max_iters_500", "learning_rate_3e4"):
        plot_reward(ax, name, series[name][REWARD])
    ax.axhline(1000, color="#667085", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    format_axis(ax, "Стандартные веса награды", "Train/mean_reward")

    for ax, name in ((axes[0, 1], "box_target_strong"), (axes[1, 0], "qpos_strong")):
        plot_reward(ax, name, series[name][REWARD])
        format_axis(ax, name, "Train/mean_reward")

    ax = axes[1, 1]
    for name in ("baseline", "learning_rate_3e4"):
        values = series[name][LEARNING_RATE]
        ax.plot(list(values), list(values.values()), label=name, color=COLORS[name], linewidth=1)
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)
    format_axis(ax, "Адаптивный шаг оптимизации", "Loss/learning_rate")
    fig.savefig(RESULTS / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    baseline_info = read_json(BASELINE / "run_info.json")
    baseline_ppo = read_json(BASELINE / "config.json")
    baseline_summary = read_json(BASELINE / "summary.json")
    scalars_by_run = {}
    observations = []
    all_episodes = []

    for name in RUNS:
        if name == "baseline":
            scalars = read_scalars(BASELINE / "tensorboard")
            evaluation = read_json(BASELINE / "video_final.json")
            iterations = baseline_info["max_iterations"]
            seconds = baseline_summary["training_seconds_recorded"]
            assert scalars[REWARD][iterations - 1] == baseline_summary["final_training_mean_reward"]
        else:
            run_dir = ROOT / "logs" / name
            output_dir = RESULTS / name
            evaluation = read_json(output_dir / "video.json")
            run_info = verify_configs(name, baseline_info, baseline_ppo, run_dir, evaluation)
            iterations = run_info["max_iterations"]
            seconds = run_info["training_seconds"]
            scalars = read_scalars(run_dir)
            export_scalars(output_dir / "training_metrics.csv", scalars)
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            plot_reward(ax, name, scalars[REWARD])
            format_axis(ax, name, "Train/mean_reward")
            fig.savefig(output_dir / "learning_curve.png", dpi=150)
            plt.close(fig)

        scalars_by_run[name] = scalars
        observations.append(summarize(name, scalars, evaluation, iterations, seconds))
        for episode in evaluation["episodes"]:
            all_episodes.append({"experiment": name, **episode, "max_cube_z_m": episode.get("max_cube_z_m")})

    baseline = observations[0]
    for row in observations:
        row["delta_training_mean_reward"] = row["final_training_mean_reward"] - baseline["final_training_mean_reward"]
        row["delta_training_mean_reward_pct"] = 100 * row["delta_training_mean_reward"] / baseline["final_training_mean_reward"]
        row["delta_sr_percentage_points"] = 100 * (row["successes"] - baseline["successes"]) / 10
        row["delta_seconds"] = row["seconds"] - baseline["seconds"]
        row["delta_seconds_pct"] = 100 * row["delta_seconds"] / baseline["seconds"]
    write_csv(RESULTS / "comparison.csv", observations)
    write_csv(RESULTS / "evaluation_metrics.csv", all_episodes)
    plot_comparison(scalars_by_run)


if __name__ == "__main__":
    main()
