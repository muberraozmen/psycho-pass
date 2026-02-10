import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional


def _velocity(traj: np.ndarray) -> Dict[str, float]:
    # Calculate difference between consecutive steps
    steps = traj[1:] - traj[:-1]
    
    # Step Magnitudes: ||v_t||
    step_distances = np.linalg.norm(steps, axis=1)
    
    return {
        "avg_velocity": float(np.mean(step_distances)),
        "max_velocity": float(np.max(step_distances)),
        "step_variance": float(np.var(step_distances))
    }

def _distance(traj: np.ndarray) -> Dict[str, float]:
    steps = traj[1:] - traj[:-1]
    step_distances = np.linalg.norm(steps, axis=1)
    
    total_length = np.sum(step_distances)
    displacement = np.linalg.norm(traj[-1] - traj[0])
    
    return {
        "total_path_length": float(total_length),
        "final_displacement": float(displacement)
    }

def _directness(traj: np.ndarray) -> Dict[str, float]:
    steps = traj[1:] - traj[:-1]
    step_distances = np.linalg.norm(steps, axis=1)
    total_length = np.sum(step_distances)
    displacement = np.linalg.norm(traj[-1] - traj[0])
    
    directness = 0.0
    if total_length > 1e-9:
        directness = displacement / total_length
        
    return {
        "directness_index": float(directness)
    }

def evaluate_trajectory(embeddings_path: str, metrics_path: str) -> pd.DataFrame:

    # Path checks
    if not Path(embeddings_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)    


    # Load embeddings
    embeddings_df = pd.read_parquet(embeddings_path)

    # Evaluate trajectories
    results = []
    for _, row in embeddings_df.iterrows():
        traj = np.array([conversation for conversation in row.get("embeddings")], dtype=float)

        # At least 2 turns to calculate velocity/distance
        if traj is not None and traj.shape[0] >= 2:

            metrics = {
                "conversation_id": row.get("conversation_id", "unknown"),
                "is_success": row.get("is_success", False),
                "is_failure": row.get("is_failure", not row.get("is_success", False)),
                "turn_count": int(traj.shape[0])
            }

            metrics.update(_velocity(traj))
            metrics.update(_distance(traj))
            metrics.update(_directness(traj))
            
            results.append(metrics)

    metrics_df = pd.DataFrame(results)
    
    # Save to the correct output directory
    metrics_df.to_csv(metrics_path, index=False)
    
    return metrics_df

def log_comparative_stats(metrics_csv_path: str):
    """
    Reads a metrics.csv file and logs a comparative table of averages 
    between successful and failed attacks.
    """
    path = Path(metrics_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    df = pd.read_csv(path)
    
    # Split groups
    success_df = df[df["is_success"] == True]
    failure_df = df[df["is_success"] == False]
    
    # Define metrics to compare
    metrics_to_compare = [
        ("turn_count", "Avg Turns"),
        ("avg_velocity", "Avg Velocity"),
        ("max_velocity", "Max Velocity"),
        ("step_variance", "Step Variance"),
        ("total_path_length", "Total Path Length"),
        ("final_displacement", "Net Displacement"),
        ("directness_index", "Directness (0-1)")
    ]

    # Header Construction
    s_count = len(success_df)
    f_count = len(failure_df)
    
    report_lines = []
    report_lines.append("\n" + "="*80)
    report_lines.append(f"TRAJECTORY ANALYSIS REPORT | {path.parent.name}")
    report_lines.append("="*80)
    report_lines.append(f"{'METRIC':<25} | {'SUCCESS (N=' + str(s_count) + ')':<18} | {'FAILURE (N=' + str(f_count) + ')':<18} | {'DELTA':<10}")
    report_lines.append("-" * 80)

    # Helper for safe mean calculation
    def get_mean(d, col):
        return d[col].mean() if not d.empty else 0.0

    for col, label in metrics_to_compare:
        s_val = get_mean(success_df, col)
        f_val = get_mean(failure_df, col)
        
        # Calculate Delta (%)
        if f_val != 0:
            delta = ((s_val - f_val) / f_val) * 100
            delta_str = f"{delta:+.1f}%"
        elif s_val != 0:
            delta_str = "+Inf"
        else:
            delta_str = "0.0%"
            
        report_lines.append(f"{label:<25} | {s_val:<18.4f} | {f_val:<18.4f} | {delta_str}")

    report_lines.append("-" * 80)
    
    # Interpretation Footer
    report_lines.append("INTERPRETATION:")
    report_lines.append("  * Velocity: High avg velocity often implies the attacker is drastically changing strategies.")
    report_lines.append("  * Directness: High directness (near 1.0) means the attack was efficient/surgical.")
    report_lines.append("  * Displacement: High displacement means the conversation ended far from where it started.")
    report_lines.append("="*80)

    return "\n".join(report_lines)