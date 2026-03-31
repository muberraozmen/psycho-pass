import numpy as np


def l2_norm_metrics(traj: np.ndarray) -> dict[str, float | int]:
    num_steps = traj.shape[0]

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
        "num_steps": int(num_steps),
        "distance": float(distance),
        "displacement": float(displacement),
        "speed": float(np.mean(steps)),
        "speed_std": float(np.std(steps)),
        "velocity": float(displacement / num_steps),
        "directness": float(displacement / (distance + 1e-9)),
        "circularity": float(circularity),
    }

    return metrics
