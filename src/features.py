import numpy as np
import pandas as pd
import pycatch22


__all__ = ["l2norm_features", "catch22_features"]


def l2norm_features(traj: np.ndarray) -> dict[str, float | int]:
    # 0. Calculate the steps 
    num_steps = traj.shape[0]
    steps = np.linalg.norm(traj[1:] - traj[:-1], axis=1)

    # 1. Step Distances & Path Length
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
        "distance": float(distance),
        "displacement": float(displacement),
        "speed": float(np.mean(steps)),
        "speed_std": float(np.std(steps)),
        "velocity": float(displacement / num_steps),
        "directness": float(displacement / (distance + 1e-9)),
        "circularity": float(circularity),
    }

    return metrics


def catch22_features(traj: np.ndarray, top_k: int = 100) -> dict[str, float | int]:
    # 0. Compute feature names
    res_sample = pycatch22.catch22_all(traj[:, 0], short_names=True)
    feature_names = res_sample['short_names']
    
    # 1. Variance Pre-filtering (keep only most informative dimensions)
    variances = np.var(traj, axis=0)
    top_dims = np.argsort(variances)[-top_k:] 
    
    # 2. Extract features only for those dimensions
    raw_data = [pycatch22.catch22_all(traj[:, d])['values'] for d in top_dims]
    
    # 3. Aggregate across dimensions (Factor Compression)
    df = pd.DataFrame(raw_data, columns=feature_names)
    stats = df.describe().drop("count").T

    # 4. Convert to flattened dict with keys
    stats = stats.stack().to_dict()
    stats = {f"{a}_{b}": v for (a, b), v in stats.items()}

    return stats