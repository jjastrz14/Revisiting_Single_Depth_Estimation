import subprocess
import os
import sys
import argparse

def run_command(command):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

def ensure_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Created directory: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            sys.exit(1)

def check_file_exists(path):
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts in sequence.")
    parser.add_argument("--input_image", default="data/demo/img_nyu2.png", help="Input image for infer.py")
    parser.add_argument("--infer_output", default="data/demo/", help="Output directory for infer.py")
    parser.add_argument("--semantic_image", default="data/semantic_tvmonitor.png", help="Semantic image for pointcloud.py")
    parser.add_argument("--pointcloud_file_name", default="tv_pointcloud.ply", help="Output file for pointcloud.py")
    parser.add_argument("--pointcloud_directory", default="pointclouds/complex", help="Output folder to save pointclouds")
    args = parser.parse_args()

    # Check and create necessary directories
    ensure_directory("data")
    ensure_directory("data/demo")
    ensure_directory(args.infer_output)
    ensure_directory(args.pointcloud_directory)

    # Check if input files exist
    check_file_exists(args.input_image)
    check_file_exists(args.semantic_image)

    # Run infer.py
    #print("Running infer.py...")
    #run_command(["python3", "infer.py", "--input", args.input_image, "--output_path", args.infer_output])

    # Run pointcloud.py
    print("\nRunning pointcloud.py...")
    run_command(["python3", "pointcloud.py", "--input", args.semantic_image, "--output_path", args.pointcloud_file_name])

    # Run main.py
    print("\nRunning main.py...")
    run_command(["python3", "main.py", "--dir", args.pointcloud_directory])
    
    print("\nFinished with success!")

if __name__ == "__main__":
    main()