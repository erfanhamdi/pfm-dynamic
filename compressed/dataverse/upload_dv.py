import argparse
import subprocess
import time

API_KEY = "f61c98d4-6808-476c-96d6-2db94be2e499"
SERVER = "https://dataverse.harvard.edu"
JAR = "DVUploader-v1.3.0-beta.jar"
LIMIT = 50
RUNS = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", type=str, default="doi:10.7910/DVN/HKXBMO")
    parser.add_argument("--files", type=str, default="801/tension/1c")
    parser.add_argument("--runs", type=int, default=RUNS, help="Number of batches")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Files per batch")
    parser.add_argument("--wait", type=int, default=300, help="Seconds to wait between batches")
    args = parser.parse_args()

    for i in range(args.runs):
        skip = i * args.limit
        print(f"\n=== Run {i+1} of {args.runs} (skip={skip}, limit={args.limit}) ===\n")
        cmd = [
            "java", "-jar", JAR,
            f"-key={API_KEY}",
            f"-did={args.doi}",
            f"-server={SERVER}",
            f"-limit={args.limit}",
            f"-skip={skip}",
            args.files,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: Run {i+1} exited with code {result.returncode}")
        if i < args.runs - 1:
            print(f"Waiting {args.wait} seconds before next run...")
            time.sleep(args.wait)

if __name__ == "__main__":
    main()
