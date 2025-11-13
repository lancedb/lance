#!/usr/bin/env python3
"""
Test suite for LeRobot to Lance conversion using the real PushT dataset.
This validates the OSS conversion with actual robotics data from HuggingFace.
"""

import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import lance
import numpy as np
import subprocess
import sys

from lerobot_to_lance import LeRobotToLanceConfig, LeRobotWrapper
from lerobot_to_lance import main as convert_main


class TestRealPushT:
    """Test suite for real PushT dataset conversion."""

    def __init__(self):
        self.pusht_path = Path("/tmp/pusht_dataset")
        self.lance_path = None

    def setup_pusht_dataset(self):
        """Ensure PushT dataset is available."""
        if self.pusht_path.exists():
            # Verify it has real data, not just LFS pointers
            parquet_file = self.pusht_path / "data/chunk-000/file-000.parquet"
            if parquet_file.exists() and parquet_file.stat().st_size > 1000:
                print(f"✅ Real PushT dataset already available at {self.pusht_path}")
                return True

        print("📚 PushT dataset not found or incomplete. Download with:")
        print("   python download_pusht_dataset.py")
        return False

    def test_pusht_loading(self):
        """Test loading real PushT dataset."""
        print("\n🧪 Testing Real PushT Dataset Loading...")

        wrapper = LeRobotWrapper(str(self.pusht_path))

        # Test dataset properties
        assert wrapper.dataset_name == "pusht_dataset"
        assert wrapper.num_episodes == 206
        assert wrapper.total_frames == 25650

        # Test real PushT metadata
        info = wrapper.metadata_info
        assert info["codebase_version"] in ["v2.0", "v3.0"]
        assert info["robot_type"] == "unknown"  # PushT uses "unknown"
        assert info["fps"] == 10.0

        # Test features
        features = info["features"]
        assert "observation.image" in features
        assert "observation.state" in features
        assert "action" in features

        # Verify image is video type
        assert features["observation.image"]["dtype"] == "video"
        assert features["observation.image"]["shape"] == [96, 96, 3]

        # Verify state and action are 2D (PushT specific)
        assert features["observation.state"]["shape"] == [2]
        assert features["action"]["shape"] == [2]

        print("✅ Real PushT dataset loading tests passed!")
        return True

    def test_pusht_data_content(self):
        """Test the actual data content from PushT."""
        print("\n🧪 Testing Real PushT Data Content...")

        wrapper = LeRobotWrapper(str(self.pusht_path))

        # Load data (all episodes in one file for v3.0)
        df = wrapper.load_episode_parquet(0)

        # Check data structure
        assert len(df) == 25650  # Total frames
        expected_columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Test data types and ranges
        sample_obs = df["observation.state"].iloc[0]
        sample_action = df["action"].iloc[0]

        assert len(sample_obs) == 2, "PushT should have 2D observations"
        assert len(sample_action) == 2, "PushT should have 2D actions"

        # Check episode distribution
        episode_counts = df["episode_index"].value_counts().sort_index()
        assert len(episode_counts) == 206, "Should have 206 episodes"

        # Test task loading
        tasks = wrapper.metadata_tasks
        assert len(tasks) == 1
        assert "Push the T-shaped block onto the T-shaped target" in tasks[0]["task"]

        print("✅ Real PushT data content tests passed!")
        return True

    def test_pusht_conversion(self):
        """Test conversion of real PushT data to Lance."""
        print("\n🧪 Testing Real PushT Conversion...")

        # Setup temporary Lance output
        temp_dir = Path(tempfile.mkdtemp())
        self.lance_path = temp_dir / "pusht_lance"

        try:
            # Run conversion (limit to 10 episodes for testing)
            config = LeRobotToLanceConfig(
                source_lerobot_dataset_path=str(self.pusht_path),
                lance_output_path=str(self.lance_path),
                max_episodes=10
            )

            convert_main(config)

            # Verify Lance tables exist
            assert (self.lance_path / "datasets").exists()
            assert (self.lance_path / "episodes").exists()
            assert (self.lance_path / "frames").exists()

            print("✅ Real PushT conversion tests passed!")
            return True

        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_pusht_lance_quality(self):
        """Test the quality of converted Lance data."""
        print("\n🧪 Testing Real PushT Lance Data Quality...")

        # Use temporary Lance output
        temp_dir = Path(tempfile.mkdtemp())
        lance_path = temp_dir / "pusht_lance"

        try:
            # Convert limited dataset
            config = LeRobotToLanceConfig(
                source_lerobot_dataset_path=str(self.pusht_path),
                lance_output_path=str(lance_path),
                max_episodes=5
            )
            convert_main(config)

            # Test datasets table
            datasets = lance.dataset(str(lance_path / "datasets"))
            df = datasets.to_table().to_pandas()

            assert len(df) == 1
            dataset_row = df.iloc[0]
            assert dataset_row['name'] == 'pusht_dataset'
            assert dataset_row['robot_type'] == 'unknown'
            assert dataset_row['fps'] == 10.0

            # Test episodes table
            episodes = lance.dataset(str(lance_path / "episodes"))
            df = episodes.to_table().to_pandas()

            assert len(df) == 5  # 5 episodes converted
            for _, episode in df.iterrows():
                assert "Push the T-shaped block" in episode['task']

                # Check video paths
                video_paths = json.loads(episode['video_paths'])
                assert 'observation.image' in video_paths

            # Test frames table
            frames = lance.dataset(str(lance_path / "frames"))
            df = frames.to_table().to_pandas()

            # Should have many frames (5 episodes worth)
            assert len(df) > 1000

            # Test frame data quality
            sample_frame = df.iloc[0]
            assert len(sample_frame['observation_state']) == 2
            assert len(sample_frame['action']) == 2
            assert "Push the T-shaped block" in sample_frame['task']

            print("✅ Real PushT Lance data quality tests passed!")
            return True

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_pusht_queries(self):
        """Test Lance query performance with real PushT data."""
        print("\n🧪 Testing Real PushT Query Performance...")

        temp_dir = Path(tempfile.mkdtemp())
        lance_path = temp_dir / "pusht_lance"

        try:
            # Convert dataset
            config = LeRobotToLanceConfig(
                source_lerobot_dataset_path=str(self.pusht_path),
                lance_output_path=str(lance_path),
                max_episodes=3
            )
            convert_main(config)

            frames = lance.dataset(str(lance_path / "frames"))

            # Test episode filtering
            episode_frames = frames.to_table(
                filter="episode_frame_index < 10"
            ).to_pandas()
            assert len(episode_frames) <= 30  # Max 10 frames per episode × 3 episodes

            # Test task filtering
            task_frames = frames.to_table(
                filter="task = 'Push the T-shaped block onto the T-shaped target.'"
            ).to_pandas()
            assert len(task_frames) > 100  # Should have substantial data

            # Test column selection
            actions_only = frames.to_table(
                columns=["frame_index", "action"]
            ).to_pandas()
            assert list(actions_only.columns) == ["frame_index", "action"]

            print("✅ Real PushT query performance tests passed!")
            return True

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def show_pusht_demo(self):
        """Demonstrate the real PushT dataset conversion."""
        print("\n" + "="*80)
        print("🤖 REAL PUSHT DATASET DEMONSTRATION")
        print("="*80)

        # Show input dataset
        wrapper = LeRobotWrapper(str(self.pusht_path))
        info = wrapper.metadata_info

        print(f"\n📁 Input: Real PushT Dataset from HuggingFace")
        print(f"   📊 Episodes: {info['total_episodes']:,}")
        print(f"   📊 Frames: {info['total_frames']:,}")
        print(f"   🎯 Task: Push manipulation with T-shaped blocks")
        print(f"   🤖 Observations: 2D robot state + 96x96x3 camera")
        print(f"   🎮 Actions: 2D motor controls")
        print(f"   📱 Version: {info['codebase_version']}")

        # Show conversion benefits
        print(f"\n🚀 Lance Conversion Benefits:")
        print(f"   📊 Columnar Storage: Efficient compression and access")
        print(f"   🔍 Fast Queries: Filter by episode, time, task instantly")
        print(f"   💾 Selective Loading: Load only needed columns/frames")
        print(f"   🎥 Video References: Efficient MP4 path storage")
        print(f"   📈 ML Ready: Native Pandas/PyArrow integration")

        # Show sample conversion
        temp_dir = Path(tempfile.mkdtemp())
        lance_path = temp_dir / "demo_lance"

        try:
            print(f"\n⚡ Converting first 3 episodes for demo...")
            config = LeRobotToLanceConfig(
                source_lerobot_dataset_path=str(self.pusht_path),
                lance_output_path=str(lance_path),
                max_episodes=3
            )
            convert_main(config)

            # Show results
            frames = lance.dataset(str(lance_path / "frames"))
            total_frames = len(frames.to_table().to_pandas())

            print(f"   ✅ Converted {total_frames:,} frames to Lance format")
            print(f"   ⚡ Query demo: frames with episode_frame_index < 5")

            early_frames = frames.to_table(
                filter="episode_frame_index < 5"
            ).to_pandas()
            print(f"   📊 Result: {len(early_frames)} frames returned instantly")

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def run_all_tests(self):
        """Run complete test suite on real PushT data."""
        print("🚀 Real PushT Dataset Test Suite for Lance OSS")
        print("="*60)

        # Check dataset availability
        if not self.setup_pusht_dataset():
            print("❌ PushT dataset not available. Run download_pusht_dataset.py first.")
            return False

        # Run tests
        tests = [
            self.test_pusht_loading,
            self.test_pusht_data_content,
            self.test_pusht_conversion,
            self.test_pusht_lance_quality,
            self.test_pusht_queries
        ]

        passed = 0
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed: {e}")
                import traceback
                traceback.print_exc()

        # Show demo regardless
        if passed > 0:
            self.show_pusht_demo()

        print("\n" + "="*60)
        print(f"🎉 Test Results: {passed}/{len(tests)} tests passed")

        if passed == len(tests):
            print("🌟 All real PushT tests passed! OSS package ready for Lance contribution.")
            return True
        else:
            print("⚠️  Some tests failed.")
            return False


def main():
    """Main test runner."""
    tester = TestRealPushT()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)