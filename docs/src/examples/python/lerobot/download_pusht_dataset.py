#!/usr/bin/env python3
"""
Download the real PushT dataset from HuggingFace for Lance conversion example.
This script downloads the actual LeRobot-format robotics dataset used in research.
"""

import subprocess
import sys
from pathlib import Path
import json


def setup_git_lfs():
    """Ensure Git LFS is available."""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("✅ Git LFS already available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing Git LFS...")
        try:
            # Try to install via brew on macOS
            subprocess.run(["brew", "install", "git-lfs"], check=True, capture_output=True)
            subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
            print("✅ Git LFS installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Could not install Git LFS automatically")
            print("💡 Please install manually: https://git-lfs.com")
            return False


def download_pusht_dataset(output_dir="/tmp/pusht_dataset"):
    """Download the real PushT dataset from HuggingFace."""
    output_path = Path(output_dir)

    if output_path.exists():
        print(f"📁 Dataset already exists at {output_path}")
        return str(output_path)

    print("📚 Downloading PushT dataset from HuggingFace...")

    try:
        # Clone the repository
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/datasets/lerobot/pusht",
            str(output_path)
        ], check=True, cwd=output_path.parent)

        # Pull LFS files
        print("📦 Downloading actual data files (may take a moment)...")
        subprocess.run(["git", "lfs", "pull"], check=True, cwd=output_path)

        # Verify we have real data
        parquet_file = output_path / "data/chunk-000/file-000.parquet"
        if parquet_file.exists() and parquet_file.stat().st_size > 1000:
            print(f"✅ Successfully downloaded PushT dataset to {output_path}")

            # Show dataset info
            show_dataset_info(output_path)
            return str(output_path)
        else:
            print("❌ Downloaded files appear to be LFS pointers, not actual data")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return None


def show_dataset_info(dataset_path: Path):
    """Display information about the downloaded dataset."""
    try:
        # Read metadata
        info_path = dataset_path / "meta/info.json"
        with open(info_path) as f:
            info = json.load(f)

        print(f"\n📊 PushT Dataset Information:")
        print(f"   🤖 Robot Type: {info.get('robot_type', 'N/A')}")
        print(f"   📊 Episodes: {info.get('total_episodes', 'N/A')}")
        print(f"   📊 Frames: {info.get('total_frames', 'N/A'):,}")
        print(f"   🎯 FPS: {info.get('fps', 'N/A')}")
        print(f"   📱 Version: {info.get('codebase_version', 'N/A')}")

        # Show features
        features = info.get('features', {})
        print(f"   🔧 Features:")
        for feature_name, feature_info in features.items():
            dtype = feature_info.get('dtype', 'unknown')
            shape = feature_info.get('shape', [])
            print(f"      - {feature_name}: {dtype} {shape}")

        # Check data file size
        parquet_file = dataset_path / "data/chunk-000/file-000.parquet"
        if parquet_file.exists():
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            print(f"   💾 Data size: {size_mb:.1f} MB")

    except Exception as e:
        print(f"⚠️  Could not read dataset info: {e}")


def main():
    """Main function to download PushT dataset."""
    print("🚀 PushT Dataset Downloader for Lance OSS Demo")
    print("=" * 60)

    # Setup Git LFS
    if not setup_git_lfs():
        print("❌ Git LFS required but not available")
        return False

    # Download dataset
    dataset_path = download_pusht_dataset()

    if dataset_path:
        print(f"\n🎉 Success! Real PushT dataset ready at: {dataset_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Run conversion: python lerobot_to_lance.py --source-lerobot-dataset-path {dataset_path}")
        print(f"   2. Run tests: python test_real_pusht.py")
        print(f"   3. Explore data with Lance queries")
        return True
    else:
        print("❌ Failed to download dataset")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)