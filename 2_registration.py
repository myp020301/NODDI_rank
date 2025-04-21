#!/usr/bin/env python3
import os, glob, subprocess, argparse, sys

def run_command(cmd):
    """Run a shell command and exit on failure."""
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="将单个标准空间 ROI 配准到被试 DWI 空间并输出 ROI 掩膜")
    parser.add_argument("--data_path", required=True, help="被试目录，包含 nodif_brain.nii.gz；若启用 T1 流程，还需包含 T1_brain.nii.gz 和 data/dtifit_FA.nii.gz")
    parser.add_argument("--roi_dir",default="/data2/mayupeng/BNU/ROI",help="标准空间 ROI 文件夹，文件名 *.nii 或 *.nii.gz")
    parser.add_argument("--roi_name",required=True,help="要处理的 ROI 名称（不含扩展名），如 MCP 或 FA_L")
    parser.add_argument("--mni_template", default=os.path.join(os.environ.get("FSLDIR","/usr/local/fsl"),"data/standard/MNI152_T1_1mm_brain.nii.gz"), help="MNI152 T1 1mm 脑模板")
    parser.add_argument("--fa_template", default=os.path.join(os.environ.get("FSLDIR","/usr/local/fsl"),"data/standard/FMRIB58_FA_1mm.nii.gz"), help="FMRIB58 FA 1mm 模板")
    parser.add_argument("--use_t1", action="store_true", help="启用基于 T1 的注册流程；否则使用 FA↔FA 流程")
    args = parser.parse_args()

    data_path = args.data_path
    os.chdir(data_path)
    out_dir = os.path.join(data_path, "data")
    os.makedirs(out_dir, exist_ok=True)

    roi_file = os.path.join(args.roi_dir, args.roi_name)
    roi_name = os.path.splitext(os.path.basename(roi_file))[0]

    # 1. 检查或生成 nodif_brain
    nodif_candidate = next(iter(glob.glob(os.path.join(data_path, "nodif_brain.nii*"))), None)
    nodif_path = os.path.join(data_path, "data", "nodif_brain.nii.gz")
    if not nodif_candidate:
        # 如果未找到 nodif_brain，则从 data/data.nii.gz 根据 bvals=0 生成
        bvals_file = os.path.join(data_path, "data", "bvals")
        vols_raw = subprocess.check_output(
            f"awk '{{for(i=1;i<=NF;i++) if($i==0) printf \"%d \" , i-1}}' {bvals_file}",
            shell=True
        ).decode().strip()
        vols = ",".join(vols_raw.split())
        run_command(f"fslselectvols -i data/data.nii.gz -o data/b0_vols.nii.gz --vols={vols}")
        run_command(f"fslmaths data/b0_vols.nii.gz -Tmean {nodif_path}")
    nodif = nodif_path

    if args.use_t1:
        # 2a. 检查 T1 相关变换是否已计算
        mat_t1_dwi = os.path.join(out_dir, "T1_to_DWI.mat")
        warp_std2t1 = os.path.join(out_dir, "std2T1_warp.nii.gz")
        if not (os.path.exists(mat_t1_dwi) and os.path.exists(warp_std2t1)):
            t1 = os.path.join(data_path, "data", "T1_brain.nii.gz")
            if not os.path.exists(t1):
                print("[ERROR] 未找到 T1_brain.nii.gz", file=sys.stderr)
                sys.exit(1)
            # T1 to DWI (b0)
            run_command(f"flirt -in {t1} -ref {nodif} -omat {mat_t1_dwi} -out {out_dir}/rT1_in_DWI.nii.gz")
            # T1 to MNI (affine)
            run_command(f"flirt -in {t1} -ref {args.mni_template} -omat {out_dir}/T1_to_MNI_aff.mat -out {out_dir}/rT1_in_MNI_aff.nii.gz")
            # Nonlinear warp T1->MNI
            run_command(f"fnirt --in={t1} --aff={out_dir}/T1_to_MNI_aff.mat --ref={args.mni_template} --cout={out_dir}/T1_to_MNI_warpcoef.nii.gz")
            # Invert warp (standard->T1)
            run_command(f"invwarp -w {out_dir}/T1_to_MNI_warpcoef.nii.gz -r {t1} -o {warp_std2t1}")

        # 3a. 应用配准，仅 do applywarp
        run_command(
            f"applywarp --ref={nodif} --in={roi_file} --warp={warp_std2t1} "
            f"--premat={mat_t1_dwi} --out={out_dir}/{roi_name}_mask.nii.gz --interp=nn"
        )
        print(f"[INFO] 基于 T1 流程完成, 输出: {out_dir}/{roi_name}_mask.nii.gz")

    else:
        # 2b. 检查 FA 相关变换是否已计算
        fa = os.path.join(data_path, "data", "dtifit_FA.nii.gz")
        aff_fa_std = os.path.join(out_dir, "FA_to_std_aff.mat")
        warp_std2fa = os.path.join(out_dir, "std2FA_warp.nii.gz")
        if not (os.path.exists(aff_fa_std) and os.path.exists(warp_std2fa)):
            # Run dtifit to generate FA map
            if not os.path.exists(fa):
                run_command(
                    f"dtifit -k data/data.nii.gz -o data/dtifit "
                    f"-m data/nodif_brain_mask.nii.gz -r data/bvecs -b data/bvals"
                )
            if not os.path.exists(fa):
                print("[ERROR] 未找到 dtifit 生成的 FA 图", file=sys.stderr)
                sys.exit(1)
            # FA to Standard affine
            run_command(f"flirt -in {fa} -ref {args.fa_template} -omat {aff_fa_std} -out {out_dir}/FA_in_std_aff.nii.gz")
            # Nonlinear warp FA->std
            run_command(f"fnirt --in={fa} --aff={aff_fa_std} --ref={args.fa_template} --cout={out_dir}/FA_to_std_warpcoef.nii.gz")
            # Invert warp (std->FA)
            run_command(f"invwarp -w {out_dir}/FA_to_std_warpcoef.nii.gz -r {fa} -o {warp_std2fa}")

        # 3b. 应用配准，仅 do applywarp
        run_command(
            f"applywarp --ref={fa} --in={roi_file} --warp={warp_std2fa} "
            f"--out={out_dir}/{roi_name}_mask.nii.gz --interp=nn"
        )
        print(f"[INFO] 基于 FA 流程完成, 输出: {out_dir}/{roi_name}_mask.nii.gz")

if __name__ == "__main__": main()
