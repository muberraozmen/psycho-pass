import numpy as np
import pandas as pd
import pycatch22

__all__ = ["l2norm_features", "catch22_features"]


_CATCH22_COLUMNS = (
    "outlier_timing_pos",
    "outlier_timing_neg",
    "high_fluctuation",
    "stretch_decreasing",
    "stretch_high",
    "trev",
)


def l2norm_features(traj: np.ndarray) -> dict[str, float | int]:
    # Calculate the steps
    num_steps = traj.shape[0]
    steps = np.linalg.norm(traj[1:] - traj[:-1], axis=1)

    # Step Distances & Path Length
    distance = np.sum(steps)
    displacement = np.linalg.norm(traj[-1] - traj[0])

    # Circularity (Shape)
    if num_steps < 3:
        circularity = float("nan")
    else:
        centroids = np.mean(traj, axis=0)
        radii = np.linalg.norm(traj - centroids, axis=1)
        circularity = np.std(radii) / (np.mean(radii) + 1e-9)

    # Assemble Metrics
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
    # Find the embedding dimensions with the highest variance
    variances = np.var(traj, axis=0)
    top_dims = np.argsort(variances)[-top_k:]

    # Get the feature names and indices
    probe = pycatch22.catch22_all(traj[:, top_dims[0]], short_names=True)
    name_to_i = {n: i for i, n in enumerate(probe["short_names"])}
    col_idx = [name_to_i[name] for name in _CATCH22_COLUMNS]

    # Calculate features for the top dimensions
    raw_data = []
    for d in top_dims:
        res = pycatch22.catch22_all(traj[:, d], short_names=True)
        raw_data.append([res["values"][j] for j in col_idx])
    df = pd.DataFrame(raw_data, columns=list(_CATCH22_COLUMNS))

    # Aggregate features over the top dimensions
    agg = df.agg(["mean", "max", "min", "std"])
    metrics = {
        f"{col}_{stat}": float(agg.loc[stat, col])
        for stat in ("mean", "max", "min", "std")
        for col in _CATCH22_COLUMNS
    }
    return metrics
