import numpy as np


def l2_norm_metrics(traj: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    num_steps = traj.shape[0]
    if num_steps < 2:
        nan = float("nan")
        metrics = {
            "total_distance": nan,
            "displacement": nan,
            "average_speed": nan,
            "maximum_speed": nan,
            "minimum_speed": nan,
            "speed_variance": nan,
            "velocity": nan,
            "directness": nan,
            "circularity": nan,
        }
        return metrics

    # 1. Step Distances & Path Length
    steps = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    distance = np.sum(steps)
    displacement = np.linalg.norm(traj[-1] - traj[0])

    # 2. Circularity (Shape)
    if num_steps < 3:
        circularity = float("nan")
    else:
        centroids = np.mean(traj, axis=0)
        radii = np.linalg.norm(traj - centroids, axis=1)
        circularity = np.std(radii) / (np.mean(radii) + 1e-9)

    # 3. Assemble Metrics
    metrics = {
        f"{prefix}_num_steps": num_steps,
        f"{prefix}_distance": distance,
        f"{prefix}_displacement": displacement,
        f"{prefix}_speed": np.mean(steps),
        f"{prefix}_speed_std": np.std(steps),
        f"{prefix}_velocity": displacement / num_steps,
        f"{prefix}_directness": displacement / (distance + 1e-9),
        f"{prefix}_circularity": circularity,
    }

    return {k: float(v) for k, v in metrics.items()}