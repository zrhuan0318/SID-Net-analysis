#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, uuid, pandas as pd
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context   # 更稳的启动方式
ROOT = "trim_results"
N_WORKERS = 64
CHUNK = 200_000
EPS = 0.0

def _init_threads_env():
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

def canonicalize_feature(feature_str: str, type_str: str):
    if str(type_str) == "Unique":
        return (str(feature_str), )
    names = re.findall(r"'([^']+)'|\"([^\"]+)\"", str(feature_str))
    flat = [a or b for a, b in names]
    if not flat:
        flat = re.split(r"[,\s\(\)]+", str(feature_str).strip())
        flat = [x for x in flat if x and x not in {"(",")",","}]
    return tuple(sorted(set(flat)))    

def iter_rows_robust(path: str):
    need = ["Feature","Contribution","Type"]
    try:
        for chunk in pd.read_csv(path, sep="\t", usecols=need,
                                 dtype={"Feature":"string","Contribution":"float64","Type":"string"},
                                 engine="c", on_bad_lines="skip",
                                 chunksize=CHUNK, low_memory=False):
            if EPS > 0: chunk = chunk[chunk["Contribution"] > EPS]
            chunk = chunk.dropna(subset=["Contribution"])
            if chunk.empty: continue
            for feat, cval, typ in zip(chunk["Feature"], chunk["Contribution"], chunk["Type"]):
                yield (canonicalize_feature(str(feat), str(typ)), float(cval))
        return
    except Exception:
        pass
    # 降级逐行解析
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not ("Feature" in header and "Contribution" in header and "Type" in header):
            f.seek(0)
        for line in f:
            if "\t" not in line: continue
            parts = line.rstrip("\n").split("\t", maxsplit=2)
            if len(parts) != 3: continue
            feat, contrib, typ = parts
            try:
                cval = float(contrib)
            except:
                continue
            if EPS > 0 and cval <= EPS: continue
            yield (canonicalize_feature(str(feat), str(typ)), float(cval))

def agg_one_file_to_temp(path: str, tmp_dir: str):
    acc = {}
    for cf, val in iter_rows_robust(path):
        if cf not in acc:
            acc[cf] = {"occ":1, "sum":val, "max":val, "min":val}
        else:
            a = acc[cf]
            a["occ"] += 1
            a["sum"] += val
            if val > a["max"]: a["max"] = val
            if val < a["min"]: a["min"] = val

    # 写临时文件（尽量小）
    if not acc:
        return None
    rows = []
    for cf, v in acc.items():
        feat_str = cf[0] if len(cf)==1 else "||".join(cf)
        rows.append([feat_str, v["occ"], v["sum"], v["max"], v["min"]])
    df = pd.DataFrame(rows, columns=["Feature_norm","occ","sum","max","min"])
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"part_{uuid.uuid4().hex}.tsv")
    df.to_csv(tmp_path, sep="\t", index=False)
    return tmp_path

def main():
    _init_threads_env()
    if len(sys.argv) != 2:
        print("Usage: python 2_merge_sid_results_per_env.py <ENV_NAME>")
        sys.exit(1)

    env = sys.argv[1].strip()
    env_path = os.path.join(ROOT, env)
    if not os.path.isdir(env_path):
        print(f"Environment folder not found: {env_path}")
        sys.exit(2)

    files = sorted(glob(os.path.join(env_path, "*_sid_results.tsv")))
    if not files:
        print(f"[ENV {env}] no *_sid_results.tsv")
        return

    tmp_dir = os.path.join(env_path, ".merge_tmp")
    ctx = get_context("forkserver")
    temp_files = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(agg_one_file_to_temp, f, tmp_dir): f for f in files}
        for fut in as_completed(futs):
            p = fut.result()
            if p: temp_files.append(p)

    if not temp_files:
        print(f"[ENV {env}] empty after filtering")
        return

    it = (pd.read_csv(p, sep="\t", dtype={"Feature_norm":"string",
                                          "occ":"int64","sum":"float64",
                                          "max":"float64","min":"float64"})
          for p in temp_files)
    big = pd.concat(it, ignore_index=True)

    summary = (big.groupby("Feature_norm", as_index=False)
                  .agg(occ=("occ","sum"),
                       sum=("sum","sum"),
                       contrib_max=("max","max"),
                       contrib_min=("min","min")))
    summary["contrib_mean"] = summary["sum"] / summary["occ"]
    summary["k"] = summary["Feature_norm"].str.count(r"\|\|") + 1
    summary = summary.drop(columns=["sum"]).sort_values(
        by=["k","occ","contrib_mean"], ascending=[True,False,False]
    )

    out_path = os.path.join(env_path, f"{env}_sid_feature_summary.tsv")
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"[ENV {env}] ✓ summary -> {out_path}")

    # 清理临时文件
    for p in temp_files:
        try: os.remove(p)
        except: pass
    try:
        os.rmdir(tmp_dir)
    except: pass

if __name__ == "__main__":
    main()
