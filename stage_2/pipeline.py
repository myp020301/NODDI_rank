#!/usr/bin/env python3
"""
一键运行：
  • 01-04 预处理 + tractography (可并行)
  • 05   生成 fdt 连通向量
  • 06   个体初始迭代
  • 07   群体迭代
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from utils import run, DATASET_DIR, WORKDIR

def list_subs():
    return sorted([d.name for d in DATASET_DIR.iterdir()
                   if d.is_dir() and d.name.startswith("HC")])

def pipe_one(sub):
    try:
        #run(f"python 01_dtifit.py       --sub {sub}")
        #run(f"python 02_bedpostx.py      --sub {sub}")
        #run(f"python 03_warp_atlas.py    --sub {sub}")
        #run(f"python 04_probtrackx.py    --sub {sub}")
        #run(f"python 05_conn_vectors.py  --sub {sub}")
        run(f"python 06_iter_subject.py  --sub {sub}")
        return None
    except Exception as e:
        return f"{sub}: {e}"

def main():
    subs = list_subs()
    if not subs:
        print("No subjects found.")
        return
    WORKDIR.mkdir(exist_ok=True, parents=True)

    with ProcessPoolExecutor(max_workers=2) as exe:
        futs = [exe.submit(pipe_one, s) for s in subs]
        for fut in as_completed(futs):
            err = fut.result()
            if err: print("[ERR]", err)

    # 群体迭代
    run("python 07_giaga_driver.py")

if __name__ == "__main__":
    main()
