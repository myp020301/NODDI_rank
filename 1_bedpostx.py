#!/usr/bin/env python3
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    data_path = args.data_path
    os.chdir(data_path)

    if os.path.isdir("data.bedpostX"):
        print("[WARNING] data.bedpostX already exists, skipping bedpostx")
    else:
        subprocess.run("bedpostx data -n 3", shell=True, check=True)
        print("[INFO] bedpostx completed.")
        
if __name__ == "__main__":
    main()
