import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run code coverage analysis.")
parser.add_argument("-p", "--package", type=str, help="The Rust crate to analyze.")
parser.add_argument(
    "-f", "--file", type=str, help="The specific file to show coverage for."
)
args = parser.parse_args()

cmd = ["cargo", "+nightly", "llvm-cov", "-q", "--branch", "--text", "--color", "always"]
if args.package:
    cmd += ["-p", args.package]

result = subprocess.run(cmd, capture_output=True)
if result.returncode != 0:
    print("Error running coverage analysis:")
    print(result.stderr.decode())
elif args.file:
    # Look for the specific file's coverage details
    # Section headers look like: /path/to/file.rs:
    lines = result.stdout.splitlines()
    in_file_section = False
    file_bytes = args.file.encode()
    for line in lines:
        # Check if this is a section header (path ending with colon)
        stripped = line.rstrip()
        is_section_header = stripped.endswith(b":") and b"|" not in line
        if is_section_header:
            if file_bytes in line:
                in_file_section = True
            elif in_file_section:
                # Hit a new section, stop
                break
        if in_file_section:
            print(line.decode())
else:
    print(result.stdout.decode())
