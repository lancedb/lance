import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run code coverage analysis.")
parser.add_argument("-p", "--package", type=str, help="The Rust crate to analyze.")
parser.add_argument(
    "-f", "--file", type=str, help="The specific file to show coverage for."
)
args = parser.parse_args()

cmd = ["cargo", "+nightly", "llvm-cov", "-q"]
if args.package:
    cmd += ["-p", args.package]
cmd += ["--branch"]
if args.file:
    cmd += ["--text"]
    cmd += ["--color", "always"]

result = subprocess.run(cmd, capture_output=True)
if result.returncode != 0:
    print("Error running coverage analysis:")
    print(result.stderr)
elif args.file:
    # Look for the specific file's coverage details
    lines = result.stdout.splitlines()
    in_file_section = False
    file_bytes = args.file.encode()
    for line in lines:
        if file_bytes in line:
            in_file_section = True
        elif in_file_section and line.strip() == "":
            break
        if in_file_section:
            print(line.decode())
else:
    print(result.stdout)
