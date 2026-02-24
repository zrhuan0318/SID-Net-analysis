from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple


def _set_seed(seed: Optional[int]) -> None:
    if seed is not None:
        np.random.seed(seed)


def _make_sparse_matrix(
    n: int,
    connectance: float = 0.08,
    neg_pos_ratio: float = 0.6,
    w_scale: float = 0.08,
) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    mask = (np.random.rand(n, n) < connectance)

    # Laplace gives occasional larger values; you can switch to normal if you prefer.
    vals = np.random.laplace(loc=0.0, scale=w_scale, size=(n, n))
    signs = np.where(np.random.rand(n, n) < neg_pos_ratio, -1.0, 1.0)
    vals *= signs

    A[mask] = vals[mask]
    np.fill_diagonal(A, 0.0)
    return A


def _make_block_modular(
    n: int,
    n_blocks: int = 3,
    p_in: float = 0.16,
    p_out: float = 0.03,
    neg_pos_ratio: float = 0.6,
    w_scale: float = 0.08,
) -> np.ndarray:
    sizes = np.full(n_blocks, n // n_blocks, dtype=int)
    sizes[: (n % n_blocks)] += 1
    idx = np.cumsum(np.r_[0, sizes])

    A = np.zeros((n, n), dtype=float)
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            i0, i1 = idx[bi], idx[bi + 1]
            j0, j1 = idx[bj], idx[bj + 1]
            p = p_in if bi == bj else p_out
            mask = (np.random.rand(i1 - i0, j1 - j0) < p)

            vals = np.random.laplace(loc=0.0, scale=w_scale, size=(i1 - i0, j1 - j0))
            signs = np.where(np.random.rand(i1 - i0, j1 - j0) < neg_pos_ratio, -1.0, 1.0)
            vals *= signs

            A[i0:i1, j0:j1] = mask * vals

    np.fill_diagonal(A, 0.0)
    return A


def _stabilize(
    A: np.ndarray,
    self_lim: float = -2.0,
    margin: float = 0.2,
    max_iter: int = 5,
) -> np.ndarray:
    A = A.copy()
    np.fill_diagonal(A, float(self_lim))

    for _ in range(max_iter):
        eig = np.linalg.eigvals(A)
        max_real = float(np.max(np.real(eig)))
        if np.isfinite(max_real) and max_real > -margin:
            scale = (-margin / max_real) * 0.9
            A *= scale
            np.fill_diagonal(A, float(self_lim))
        else:
            break
    return A


def _external_covariates(cfg: "GLVConfig") -> np.ndarray:
    T = cfg.n_time
    m = cfg.n_covariates
    if m <= 0:
        return np.zeros((T, 0), dtype=float)

    t = np.arange(T) * cfg.dt
    E = np.zeros((T, m), dtype=float)
    for j in range(m):
        if cfg.cov_pattern == "sinusoid":
            freq = 0.05 + 0.15 * np.random.rand()
            phase = 2 * np.pi * np.random.rand()
            E[:, j] = np.sin(2 * np.pi * freq * t + phase)
        elif cfg.cov_pattern == "step":
            t0 = int(0.5 * T)
            E[:t0, j] = 0.0
            E[t0:, j] = 1.0
        else:
            E[:, j] = np.random.normal(0, 1, size=T)
    return cfg.covariate_amplitude * E


@dataclass
class GLVConfig:
    n_species: int
    n_time: int
    dt: float = 0.01
    seed: Optional[int] = None

    # Growth
    growth_mean: float = 0.2
    growth_sd: float = 0.10

    # Interactions
    connectance: float = 0.08
    neg_pos_ratio: float = 0.60
    weight_scale: float = 0.08
    modular: bool = False
    n_blocks: int = 3

    # Stabilization
    self_lim: float = -2.0
    margin: float = 0.2

    # Saturation (important for stability)
    holling_h: float = 0.1

    # Noise & floors
    process_noise_sd: float = 0.02
    obs_logn_sd: float = 0.0
    immigration: float = 1e-6
    floor: float = 1e-12

    # Optional covariates
    n_covariates: int = 0
    covariate_amplitude: float = 0.5
    cov_sensitivity_sd: float = 0.2
    cov_pattern: str = "sinusoid"

    # Cohort
    n_sites: int = 1
    site_variability: float = 0.0

    # Counts
    return_counts: bool = False
    depth_mean: int = 20000
    depth_sd: int = 3000


def _generate_params(cfg: GLVConfig) -> Dict[str, Any]:
    n = cfg.n_species
    r = np.random.normal(cfg.growth_mean, cfg.growth_sd, size=n)

    if cfg.modular:
        A = _make_block_modular(
            n=n,
            n_blocks=cfg.n_blocks,
            p_in=min(0.8, cfg.connectance * 2.0),
            p_out=max(0.001, cfg.connectance / 2.0),
            neg_pos_ratio=cfg.neg_pos_ratio,
            w_scale=cfg.weight_scale,
        )
    else:
        A = _make_sparse_matrix(
            n=n,
            connectance=cfg.connectance,
            neg_pos_ratio=cfg.neg_pos_ratio,
            w_scale=cfg.weight_scale,
        )

    A = _stabilize(A, self_lim=cfg.self_lim, margin=cfg.margin)

    B = None
    if cfg.n_covariates > 0:
        B = np.random.normal(0.0, cfg.cov_sensitivity_sd, size=(n, cfg.n_covariates))
    return {"r": r, "A": A, "B": B}


def simulate_glv(
    cfg: GLVConfig,
    preset_A: Optional[np.ndarray] = None,
    preset_r: Optional[np.ndarray] = None,
    preset_B: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    _set_seed(cfg.seed)
    n, T, dt = cfg.n_species, cfg.n_time, cfg.dt

    if (preset_A is None) or (preset_r is None) or (cfg.n_covariates > 0 and preset_B is None):
        pars = _generate_params(cfg)
        r = pars["r"] if preset_r is None else preset_r
        A = pars["A"] if preset_A is None else preset_A
        B = pars["B"] if preset_B is None else preset_B
    else:
        r, A, B = preset_r, preset_A, preset_B

    A = _stabilize(A, self_lim=cfg.self_lim, margin=cfg.margin)

    E_t = _external_covariates(cfg)


    X = np.abs(np.random.lognormal(mean=-1.0, sigma=0.5, size=n)) + cfg.immigration
    X = np.clip(X, cfg.floor, 10.0)

    traj = np.zeros((T, n), dtype=float)
    MAX_X = 1e6  # hard cap to avoid inf

    for t in range(T):
        if cfg.holling_h > 0:
            X_eff = X / (1.0 + cfg.holling_h * X)
        else:
            X_eff = X

        inter = A @ X_eff
        drift = X * (r + inter)

        if B is not None and E_t.shape[1] > 0:
            drift += X * (B @ E_t[t])

        dX = drift * dt + cfg.process_noise_sd * np.sqrt(dt) * np.random.normal(size=n)
        X = X + dX + cfg.immigration

        X = np.nan_to_num(X, nan=cfg.floor, posinf=MAX_X, neginf=cfg.floor)
        X = np.clip(X, cfg.floor, MAX_X)

        traj[t] = X

    if cfg.obs_logn_sd > 0:
        traj = traj * np.exp(np.random.normal(0.0, cfg.obs_logn_sd, size=traj.shape))
        traj = np.clip(traj, cfg.floor, MAX_X)

    df = pd.DataFrame(traj, columns=[f"S{i+1}" for i in range(n)])
    df.insert(0, "time", np.arange(T) * dt)

    meta = {"r": r, "A": A, "B": B, "cfg": asdict(cfg)}
    return df, meta


def _safe_get(lst: Optional[List[Any]], i: int) -> Any:
    if lst is None:
        return None
    if i < 0 or i >= len(lst):
        return None
    return lst[i]


def simulate_glv_cohort(
    n_species: int,
    n_time: int,
    dt: float = 0.01,
    n_sites: int = 1,
    seed: Optional[int] = None,
    site_variability: float = 0.0,
    preset_As: Optional[List[np.ndarray]] = None,
    preset_rs: Optional[List[np.ndarray]] = None,
    preset_Bs: Optional[List[np.ndarray]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    _set_seed(seed)

    base = GLVConfig(
        n_species=n_species,
        n_time=n_time,
        dt=dt,
        seed=None,
        n_sites=n_sites,
        site_variability=site_variability,
        **kwargs,
    )

    tables: List[pd.DataFrame] = []
    meta: Dict[str, Any] = {"cfg": asdict(base), "sites": {}}

    for i in range(n_sites):
        cfg_i = GLVConfig(**asdict(base))
        cfg_i.seed = None if seed is None else (seed + i + 1)

        if site_variability > 0:
            cfg_i.growth_mean = cfg_i.growth_mean + np.random.normal(0, site_variability * 0.05)
            cfg_i.connectance = float(np.clip(cfg_i.connectance + np.random.normal(0, site_variability * 0.02), 0.01, 0.5))
            cfg_i.weight_scale = float(max(1e-4, cfg_i.weight_scale + np.random.normal(0, site_variability * 0.02)))

        A_i = _safe_get(preset_As, i)
        r_i = _safe_get(preset_rs, i)
        B_i = _safe_get(preset_Bs, i)

        df, pars = simulate_glv(cfg_i, preset_A=A_i, preset_r=r_i, preset_B=B_i)
        df["site"] = f"site{i+1}"
        tables.append(df)

        meta["sites"][f"site{i+1}"] = {
            "r": pars["r"].tolist(),
            "A": pars["A"].tolist(),
            "B": (pars["B"].tolist() if pars["B"] is not None else None),
            "cfg": pars["cfg"],
        }

    stacked = pd.concat(tables, ignore_index=True)


    species_cols = [c for c in stacked.columns if c.startswith("S")]
    stacked = stacked[["site", "time"] + species_cols]


    if kwargs.get("return_counts", False):
        rng = np.random.default_rng(seed)

        X = stacked[species_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X[X < 0] = 0.0
        X = X + 1e-12

        row_sums = X.sum(axis=1, keepdims=True)
        bad = (~np.isfinite(row_sums)) | (row_sums <= 0)


        good_mask = ~bad[:, 0]
        if np.any(good_mask):
            X[good_mask] = X[good_mask] / row_sums[good_mask]
            X[good_mask] = np.clip(X[good_mask], 0.0, 1.0)
            X[good_mask] = X[good_mask] / X[good_mask].sum(axis=1, keepdims=True)


        if np.any(bad):
            X[bad[:, 0]] = 1.0 / X.shape[1]

        depths = rng.normal(
            loc=kwargs.get("depth_mean", 20000),
            scale=kwargs.get("depth_sd", 3000),
            size=X.shape[0],
        ).astype(int)
        depths = np.maximum(depths, 100)

        counts = np.vstack([rng.multinomial(depths[i], X[i]) for i in range(X.shape[0])])
        stacked.loc[:, species_cols] = counts

    return stacked, meta
