import numpy as np


def l2_norm_metrics(traj: np.ndarray) -> dict[str, float | int]:
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
    distances = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    total_distance = np.sum(distances)
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
        "total_distance": total_distance,
        "displacement": displacement,
        "average_speed": np.mean(distances),
        "maximum_speed": np.max(distances),
        "minimum_speed": np.min(distances),
        "speed_variance": np.var(distances),
        "velocity": displacement / num_steps,
        "directness": displacement / (total_distance + 1e-9),
        "circularity": circularity,
    }

    return {k: float(v) for k, v in metrics.items()}