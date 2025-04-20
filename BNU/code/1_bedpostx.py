#!/usr/bin/env python3
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to the data directory (contains the 'data' subfolder)")
    args = parser.parse_args()
    data_path = args.data_path
    os.chdir(data_path)

    if os.path.isdir("data.bedpostX"):
        print("[WARNING] data.bedpostX already exists, skipping bedpostx")
    else:
        subprocess.run("bedpostx_gpu data -n 3", shell=True, check=True)
        print("[INFO] bedpostx_gpu completed.")
        
if __name__ == "__main__":
    main()
