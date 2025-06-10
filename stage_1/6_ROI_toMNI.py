#!/usr/bin/env python3
import os, glob, subprocess, argparse, sys

# Predefined list of brain regions (corresponding to the JHU50 template)


def run_command(cmd):
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def warp_roi(input_roi, output_roi, warp_file, ref_image, premat=None):
    cmd = f"applywarp --ref={ref_image} --in={input_roi} --warp={warp_file}"
    if premat:
        cmd += f" --premat={premat}"
    cmd += f" --out={output_roi} --interp=nn"
    run_command(cmd)
    print(f"[INFO] Warped {input_roi} -> {output_roi}")

def main():
    parser = argparse.ArgumentParser(description="将单个 ROI 的分割结果从个体空间投影到标准空间")
    parser.add_argument("--data_path",    required=True)
    parser.add_argument("--roi_dir", default="/data2/mayupeng/BNU/ROI")
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--max_clusters",   type=int,    default=6)
    parser.add_argument("--method",       default="sc", choices=["sc","kmeans","simlr"])
    parser.add_argument("--use_t1",       action="store_true")
    parser.add_argument("--mni_template", default=os.path.join(os.environ.get("FSLDIR","/usr/local/fsl"),"data/standard/MNI152_T1_1mm_brain.nii.gz"))
    parser.add_argument("--fa_template", default=os.path.join(os.environ.get("FSLDIR","/usr/local/fsl"),"data/standard/FMRIB58_FA_1mm.nii.gz"))
    args = parser.parse_args()

    os.chdir(args.data_path)
    region   = args.roi_name
    inp_dir  = os.path.join("data", "probtrack_old", f"parcellation_{args.method}")
    out_base = os.path.join("data", "probtrack_old", f"parcellation_{args.method}_MNI", region)
    os.makedirs(out_base, exist_ok=True)

    # 根据流程选择 warp & premat & ref_image
    if args.use_t1:
        warp_file = os.path.join("data", "T1_to_MNI_warpcoef.nii.gz")
        aff_mat   = os.path.join("data", "T1_to_DWI.mat")
        if not (os.path.exists(warp_file) and os.path.exists(aff_mat)):
            print(f"[ERROR] 找不到 {warp_file} 或 {aff_mat}", file=sys.stderr)
            sys.exit(1)
        # 反转仿射：DWI -> T1，仅在不存在时执行
        inv_aff = os.path.join("data", "DWI_to_T1.mat")
        if not os.path.exists(inv_aff):
            run_command(f"convert_xfm -omat {inv_aff} -inverse {aff_mat}")
        premat   = inv_aff
        ref_img  = args.mni_template
    else:
        warp_file = os.path.join("data", "FA_to_std_warpcoef.nii.gz")
        if not os.path.exists(warp_file):
            print(f"[ERROR] 找不到 {warp_file}", file=sys.stderr)
            sys.exit(1)
        premat   = None
        ref_img  = args.fa_template

    # 批量把每个 k 的 ROI 从个体空间推到标准空间
    for k in range(2, args.max_clusters + 1):
        in_roi  = os.path.join(inp_dir, f"seed_{args.roi_name}_{k}.nii.gz")
        if not os.path.isfile(in_roi):
            print(f"[WARN] 未找到 {in_roi}，跳过 k={k}")
            continue
        out_roi = os.path.join(out_base, f"seed_{k}.nii.gz")
        warp_roi(in_roi, out_roi, warp_file, ref_img, premat=premat)

    print(f"[INFO] ROI-to-MNI 完成: {region}")

if __name__ == "__main__":
    main()