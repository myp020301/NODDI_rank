import os
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_cluster(base_dir, roi, cl_num, mpm_thres, subjects, prob_dir):
    """
    处理每个簇，执行多数投票，并保存处理后的 MPM，同时打印各标签的体素数。
    """
    filename = f"{roi}_{cl_num}_MPM_thr{int(mpm_thres * 100)}_swapped.nii.gz"
    file_path = os.path.join(prob_dir, filename)

    # 加载图像数据
    img_nii = nib.load(file_path)
    img = img_nii.get_fdata()
    img[np.isnan(img)] = 0  # 替换 NaN 值为 0

    m, n, p = img.shape

    coords = [(i, j, k)
              for i in range(1, m-1)
              for j in range(1, n-1)
              for k in range(1, p-1)
              if img[i, j, k] != 0]

    # 多数投票
    for i, j, k in coords:
        label = np.zeros(int(cl_num) + 1, dtype=int)
        neighbors = [
            img[i-1, j, k], img[i+1, j, k],
            img[i, j-1, k], img[i, j+1, k],
            img[i, j, k-1], img[i, j, k+1],
        ]
        for nb in neighbors:
            idx = int(nb)
            if 0 <= idx <= cl_num:
                label[idx] += 1

        wjs = label.max()
        if wjs >= 3:
            jsh = np.where(label == wjs)[0]
            b = jsh[1] if len(jsh) >= 2 else jsh[0]
            img[i, j, k] = b

    # 保存处理后的图像
    out_img = nib.Nifti1Image(img, img_nii.affine, img_nii.header)
    output_filename = f"{roi}_{cl_num}_MPM_thr{int(mpm_thres * 100)}_group_smoothed.nii.gz"
    output_path = os.path.join(prob_dir, output_filename)
    nib.save(out_img, output_path)

    print(f"{roi}_{cl_num} 完成!")

def postprocess_mpm_group_xmm(base_dir, roi, subject_paths, max_clusters, mpm_thres, njobs):
    # 读取被试列表
    if os.path.isfile(subject_paths):
        with open(subject_paths) as f:
            subjects = [s.strip() for s in f if s.strip()]
    else:
        subjects = subject_paths.split(',')

    prob_dir = os.path.join(base_dir, f"MPM_{len(subjects)}")
    os.makedirs(prob_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = []
        for cl_num in range(2, max_clusters + 1):
            futures.append(executor.submit(
                process_cluster, base_dir, roi, cl_num, mpm_thres, subjects, prob_dir))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] 处理簇失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group-level MPM 后处理并打印标签统计"
    )
    parser.add_argument("--base_dir",     required=True)
    parser.add_argument("--roi_name",     required=True)
    parser.add_argument("--subject_data", required=True)
    parser.add_argument("--max_clusters", type=int, default=5)
    parser.add_argument("--mpm_thres",    type=float, default=0.25)
    parser.add_argument("--njobs",        type=int, default=3)
    args = parser.parse_args()

    postprocess_mpm_group_xmm(
        base_dir=args.base_dir,
        roi=args.roi_name,
        subject_paths=args.subject_data,
        max_clusters=args.max_clusters,
        mpm_thres=args.mpm_thres,
        njobs=args.njobs
    )
