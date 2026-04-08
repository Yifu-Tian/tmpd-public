import numpy as np


def format_time_to_success_summary(df, include_final_topo_energy=True):
    agg_spec = {
        "Success_Rate": lambda x: f"{np.mean(x) * 100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.mean(x) * 100:.1f}%",
        "Avg_Seg_Time": lambda x: f"{np.nanmean(x):.2f}s ± {np.nanstd(x):.2f}",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x) * 1000:.3f} ± {np.nanstd(x) * 1000:.3f}",
    }
    if include_final_topo_energy and "Final_Topo_Energy" in df.columns:
        agg_spec["Final_Topo_Energy"] = lambda x: f"{np.nanmean(x):.2f} ± {np.nanstd(x):.2f}"
    return df.groupby("Method").agg(agg_spec)


def format_single_pairs_summary(df):
    return df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x) * 100:.1f}%",
        "Topo_Energy": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x) * 1000:.3f} ± {np.nanstd(x) * 1000:.3f}",
        "Time": lambda x: f"{np.nanmean(x):.2f}s",
    })
