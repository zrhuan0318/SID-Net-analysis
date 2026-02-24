import json
import numpy as np
import pandas as pd

from glv_simulator import simulate_glv_cohort

# 确定规模
N_SPECIES = 200
N_SAMPLES = 500   # 每站点时间点数（n_time）
N_SITES   = 5
ALPHAS    = [1.0]

GLOBAL_SEED = 123


# 避免 overflow
DT = 0.01
CONNECTANCE = 0.08
NEG_POS_RATIO = 0.60

BASE_WEIGHT_SCALE = 0.08     
MODULAR = False
N_BLOCKS = 3

SELF_LIM = -2.0
MARGIN = 0.2
HOLLING_H = 0.1

PROCESS_NOISE_SD = 0.02
OBS_LOGN_SD = 0.0
SITE_VARIABILITY = 0.00

DEPTH_MEAN = 20000
DEPTH_SD   = 3000


def clr_transform(counts_df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    X = counts_df.astype(float).to_numpy()
    X = X + pseudocount
    logX = np.log(X)
    logX_centered = logX - logX.mean(axis=1, keepdims=True)
    return pd.DataFrame(logX_centered, columns=counts_df.columns)


def discretize_tertiles_per_taxon(counts_df: pd.DataFrame) -> pd.DataFrame:
    disc = pd.DataFrame(index=counts_df.index)
    for col in counts_df.columns:
        x = counts_df[col].astype(float)
        try:
            bins = pd.qcut(x, q=3, labels=[0, 1, 2], duplicates="drop")
            if bins.isna().any():
                bins = bins.astype("float").fillna(1).astype(int)
            disc[col] = bins.astype(int)
        except Exception:
            disc[col] = 1
    return disc


def generate_one_alpha(alpha: float) -> None:
    weight_scale = float(BASE_WEIGHT_SCALE) * float(alpha)

    stacked, meta = simulate_glv_cohort(
        n_species=N_SPECIES,
        n_time=N_SAMPLES,
        dt=DT,
        n_sites=N_SITES,
        seed=GLOBAL_SEED,
        site_variability=SITE_VARIABILITY,
        connectance=CONNECTANCE,
        neg_pos_ratio=NEG_POS_RATIO,
        weight_scale=weight_scale,
        modular=MODULAR,
        n_blocks=N_BLOCKS,
        self_lim=SELF_LIM,
        margin=MARGIN,
        holling_h=HOLLING_H,
        process_noise_sd=PROCESS_NOISE_SD,
        obs_logn_sd=OBS_LOGN_SD,
        return_counts=True,
        depth_mean=DEPTH_MEAN,
        depth_sd=DEPTH_SD,
    )

    species_cols = [c for c in stacked.columns if c.startswith("S")]

    Xcheck = stacked[species_cols].to_numpy(dtype=float)
    if not np.isfinite(Xcheck).all():
        raise RuntimeError("Non-finite values in generated trajectory (NaN/inf). Try smaller DT/weight_scale or larger self_lim/margin.")

    # 只物种列
    counts_df = stacked[species_cols].copy()

    # 样品名写入meta
    sample_names = [
        f"{site}__t{t_idx:04d}__time{time_val:.3f}"
        for t_idx, (site, time_val) in enumerate(zip(stacked["site"].tolist(), stacked["time"].tolist()))
    ]

    # CLR与离散化
    clr_df = clr_transform(counts_df, pseudocount=1.0)
    disc_df = discretize_tertiles_per_taxon(counts_df)

    out_prefix = f"sid_input_alpha{alpha}"
    counts_path = f"{out_prefix}_counts.tsv"
    clr_path    = f"{out_prefix}_clr.tsv"
    disc_path   = f"{out_prefix}_disc.tsv"

    counts_df.to_csv(counts_path, sep="\t", index=False)
    clr_df.to_csv(clr_path, sep="\t", index=False)
    disc_df.to_csv(disc_path, sep="\t", index=False)

    glv_meta = {
        "species": [f"S{i+1}" for i in range(N_SPECIES)],
        "alpha": float(alpha),
        "generation": {
            "N_SPECIES": N_SPECIES,
            "N_SAMPLES_per_site": N_SAMPLES,
            "N_SITES": N_SITES,
            "DT": DT,
            "GLOBAL_SEED": GLOBAL_SEED,
            "CONNECTANCE": CONNECTANCE,
            "NEG_POS_RATIO": NEG_POS_RATIO,
            "BASE_WEIGHT_SCALE": BASE_WEIGHT_SCALE,
            "WEIGHT_SCALE_USED": weight_scale,
            "MODULAR": MODULAR,
            "N_BLOCKS": N_BLOCKS,
            "SELF_LIM": SELF_LIM,
            "MARGIN": MARGIN,
            "HOLLING_H": HOLLING_H,
            "PROCESS_NOISE_SD": PROCESS_NOISE_SD,
            "OBS_LOGN_SD": OBS_LOGN_SD,
            "SITE_VARIABILITY": SITE_VARIABILITY,
            "DEPTH_MEAN": DEPTH_MEAN,
            "DEPTH_SD": DEPTH_SD,
        },
        "sites": meta.get("sites", {}),
        "glv_cfg": meta.get("cfg", None),
        "samples": {
            "names": sample_names,
            "site": stacked["site"].tolist(),
            "time": stacked["time"].tolist(),
        },
        "output_files": {
            "counts_tsv": counts_path,
            "clr_tsv": clr_path,
            "disc_tsv": disc_path,
        },
    }

    meta_path = f"glv_meta_alpha{alpha}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(glv_meta, f, indent=2)

    print(f"[OK] alpha={alpha} -> {counts_path}, {clr_path}, {disc_path}, {meta_path}")


def main() -> None:
    for alpha in ALPHAS:
        generate_one_alpha(alpha)


if __name__ == "__main__":
    main()
