#!/usr/bin/env python3
"""
Convert LeRobot format datasets to Lance format for efficient querying and storage.

This example converts robotics datasets from the LeRobot format (used by HuggingFace)
to Lance format, demonstrating fast columnar access for machine learning workflows.

Example usage:
    python lerobot_to_lance.py --source-lerobot-dataset-path /tmp/pusht_dataset \
                               --lance-output-path /tmp/pusht_lance \
                               --max-episodes 10
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from uuid import uuid4

import lance
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import tyro

# Lance table schemas for robotics data
DATASET_SCHEMA = pa.schema([
    pa.field("_id", pa.string()),
    pa.field("name", pa.string()),
    pa.field("modality", pa.string()),  # JSON string containing modality info
    pa.field("metadata", pa.string()),  # JSON string containing full metadata
    pa.field("stats", pa.string()),  # JSON string containing statistics
    pa.field("features", pa.string()),  # JSON string containing feature definitions
    pa.field("embodiment", pa.string()),  # JSON string containing embodiment info
    pa.field("fps", pa.float64()),
    pa.field("observation_state_shape", pa.int64()),
    pa.field("action_shape", pa.int64()),
    pa.field("robot_type", pa.string()),
    pa.field("total_episodes", pa.int64()),
    pa.field("total_frames", pa.int64()),
    pa.field("total_videos", pa.int64()),
    pa.field("source_dataset_path", pa.string()),
    pa.field("created_at", pa.timestamp('us')),
])

EPISODE_SCHEMA = pa.schema([
    pa.field("_id", pa.string()),
    pa.field("episode_index", pa.int64()),
    pa.field("task", pa.string()),
    pa.field("length", pa.int64()),
    pa.field("dataset_id", pa.string()),
    pa.field("video_paths", pa.string()),  # JSON string with camera->path mapping
    pa.field("start_frame_index", pa.int64()),
    pa.field("end_frame_index", pa.int64()),
])

FRAME_SCHEMA = pa.schema([
    pa.field("frame_index", pa.int64()),
    pa.field("episode_frame_index", pa.int64()),
    pa.field("observation_state", pa.list_(pa.float32())),
    pa.field("action", pa.list_(pa.float32())),
    pa.field("task", pa.string()),
    pa.field("annotation", pa.string()),  # JSON string containing annotations
    pa.field("timestamp", pa.float32()),
    pa.field("episode_id", pa.string()),
    pa.field("dataset_id", pa.string()),
])


@dataclass
class LeRobotToLanceConfig:
    """Configuration for LeRobot to Lance conversion."""
    source_lerobot_dataset_path: str = "/tmp/pusht_dataset"
    lance_output_path: str = "/tmp/pusht_lance"
    max_episodes: int = -1  # -1 means process all episodes, otherwise limit to this number


# Lance table names
DATASETS_TABLE_NAME = "datasets"
EPISODES_TABLE_NAME = "episodes"
FRAMES_TABLE_NAME = "frames"


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read a JSON file from the filesystem."""
    with open(file_path, 'r') as f:
        return json.load(f)


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file from the filesystem."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


class LeRobotWrapper:
    """Wrapper for reading LeRobot format datasets from the filesystem.

    This class provides a unified interface for accessing LeRobot datasets,
    handling both v2.x and v3.0 formats automatically.
    """

    def __init__(self, lerobot_dataset_path: str):
        self.lerobot_dataset_path = Path(lerobot_dataset_path)
        if not self.lerobot_dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {lerobot_dataset_path}")

        self.dataset_name = self.lerobot_dataset_path.name

        # Load core metadata
        meta_dir = self.lerobot_dataset_path / "meta"
        if not meta_dir.exists():
            raise ValueError(f"Meta directory not found: {meta_dir}")

        self.metadata_info = read_json_file(meta_dir / "info.json")

        # Cache for lazy-loaded metadata
        self._metadata_tasks = None
        self._metadata_stats = None
        self._metadata_modality = None
        self._metadata_embodiment = None

        # Extract key dataset properties
        self.num_episodes = self.metadata_info["total_episodes"]
        self.total_frames = self.metadata_info["total_frames"]
        self.video_path_template = self.metadata_info["video_path"]
        self.data_path_template = self.metadata_info["data_path"]
        self.chunk_size = self.metadata_info.get("chunks_size", self.metadata_info.get("chunk_size", 1000))
        self.video_keys = [
            k for k, v in self.metadata_info["features"].items() if v["dtype"] == "video"
        ]

    @property
    def metadata_tasks(self) -> List[Dict]:
        if self._metadata_tasks:
            return self._metadata_tasks

        # Try both v2.x (tasks.jsonl) and v3.0 (tasks.parquet) formats
        jsonl_path = self.lerobot_dataset_path / "meta" / "tasks.jsonl"
        parquet_path = self.lerobot_dataset_path / "meta" / "tasks.parquet"

        if jsonl_path.exists():
            self._metadata_tasks = read_jsonl_file(str(jsonl_path))
        elif parquet_path.exists():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            # Handle both formats: task as index or as column
            if 'task' in df.columns:
                self._metadata_tasks = df.to_dict('records')
            else:
                # Task names are the index, task_index values are the data
                tasks = []
                for task_name, row in df.iterrows():
                    tasks.append({
                        "task_index": row['task_index'],
                        "task": str(task_name)
                    })
                self._metadata_tasks = tasks
        else:
            print("No tasks file found, creating default task")
            self._metadata_tasks = [{"task_index": 0, "task": "default_task"}]

        return self._metadata_tasks

    @property
    def metadata_stats(self) -> Dict:
        if self._metadata_stats:
            return self._metadata_stats
        stats_path = self.lerobot_dataset_path / "meta" / "stats.json"
        try:
            self._metadata_stats = read_json_file(str(stats_path))
        except FileNotFoundError:
            print(f"No stats.json found for {stats_path}")
            self._metadata_stats = {}
        return self._metadata_stats

    @property
    def metadata_modality(self) -> Dict:
        if self._metadata_modality:
            return self._metadata_modality
        modality_path = self.lerobot_dataset_path / "meta" / "modality.json"
        try:
            self._metadata_modality = read_json_file(str(modality_path))
        except FileNotFoundError:
            print(f"No modality.json found for {modality_path}, using default")
            # Create default modality based on features
            self._metadata_modality = {
                "observation": {},
                "action": {"field_name": "action"}
            }
            # Add observation features dynamically
            for feature_name in self.metadata_info.get("features", {}):
                if feature_name.startswith("observation."):
                    key = feature_name.replace("observation.", "")
                    self._metadata_modality["observation"][key] = {"field_name": feature_name}
        return self._metadata_modality

    @property
    def metadata_embodiment(self) -> Dict:
        if self._metadata_embodiment:
            return self._metadata_embodiment
        embodiment_path = self.lerobot_dataset_path / "meta" / "embodiment.json"
        try:
            self._metadata_embodiment = read_json_file(str(embodiment_path))
        except FileNotFoundError:
            print(f"No embodiment.json found for {embodiment_path}")
            self._metadata_embodiment = {}
        return self._metadata_embodiment

    def load_episode_parquet(self, episode_index: int) -> pd.DataFrame:
        if episode_index < 0 or episode_index >= self.num_episodes:
            raise ValueError(
                f"Episode index {episode_index} out of bounds. Must be between 0 and {self.num_episodes - 1}"
            )

        # Handle both v2.x and v3.0 formats
        if "episode_index" in self.data_path_template:
            # v2.x format: episode-based files
            chunk_idx = episode_index // self.chunk_size
            data_path = self.lerobot_dataset_path / self.data_path_template.format(
                episode_chunk=chunk_idx, episode_index=episode_index
            )
        else:
            # v3.0 format: file-based chunks, need to find which file contains this episode
            chunk_idx = episode_index // self.chunk_size
            file_idx = 0  # For simplicity, assume single file per chunk for now
            data_path = self.lerobot_dataset_path / self.data_path_template.format(
                chunk_index=chunk_idx, file_index=file_idx
            )

        return pd.read_parquet(data_path)

    def get_task_str_from_index(self, task_index: int) -> str:
        """Returns the task string from the task index by looking up the metadata/tasks.jsonl"""
        if task_index < 0 or task_index >= len(self.metadata_tasks):
            raise ValueError(
                f"Task index {task_index} out of bounds. Must be between 0 and {len(self.metadata_tasks) - 1}"
            )

        for task in self.metadata_tasks:
            if task["task_index"] == task_index:
                return task["task"]
        raise ValueError(f"Task index {task_index} not found in metadata tasks")


def create_dataset_record(lerobot_wrapper: LeRobotWrapper) -> Dict[str, Any]:
    """Create a dataset record for the Lance datasets table."""
    _id = str(uuid4())
    print(f"Creating dataset record for {lerobot_wrapper.lerobot_dataset_path} with _id {_id}")

    return {
        "_id": _id,
        "name": lerobot_wrapper.dataset_name,
        "modality": json.dumps(lerobot_wrapper.metadata_modality),
        "metadata": json.dumps(lerobot_wrapper.metadata_info),
        "stats": json.dumps(lerobot_wrapper.metadata_stats),
        "features": json.dumps(lerobot_wrapper.metadata_info["features"]),
        "embodiment": json.dumps(lerobot_wrapper.metadata_embodiment),
        "fps": lerobot_wrapper.metadata_info["fps"],
        "observation_state_shape": lerobot_wrapper.metadata_info["features"]["observation.state"]["shape"][0],
        "action_shape": lerobot_wrapper.metadata_info["features"]["action"]["shape"][0],
        "robot_type": lerobot_wrapper.metadata_info["robot_type"],
        "total_episodes": lerobot_wrapper.metadata_info["total_episodes"],
        "total_frames": lerobot_wrapper.metadata_info["total_frames"],
        "total_videos": lerobot_wrapper.metadata_info.get("total_videos", 0),
        "source_dataset_path": str(lerobot_wrapper.lerobot_dataset_path),
        "created_at": pd.Timestamp.now(),
    }


def process_frames_and_episodes_data(
    episode_index: int,
    lerobot_wrapper: LeRobotWrapper,
    dataset_id: str,
    global_frame_index: int,
) -> tuple[Dict[str, Any], pd.DataFrame, int]:
    """Process episode and frames data for Lance ingestion.

    Args:
        episode_index: Index of the episode to process
        lerobot_wrapper: LeRobot dataset wrapper
        dataset_id: UUID of the parent dataset
        global_frame_index: Running count of frames across all episodes

    Returns:
        Tuple of (episode_record, frames_dataframe, updated_global_frame_index)
    """
    chunk_idx = episode_index // lerobot_wrapper.chunk_size

    # Build video paths for all cameras per episode
    video_paths = {}
    for video_key in lerobot_wrapper.video_keys:
        if "episode_index" in lerobot_wrapper.video_path_template:
            # v2.x format
            video_path = lerobot_wrapper.lerobot_dataset_path / lerobot_wrapper.video_path_template.format(
                episode_chunk=chunk_idx, episode_index=episode_index, video_key=video_key
            )
        else:
            # v3.0 format
            file_idx = 0  # Assume single file per chunk for simplicity
            video_path = lerobot_wrapper.lerobot_dataset_path / lerobot_wrapper.video_path_template.format(
                chunk_index=chunk_idx, file_index=file_idx, video_key=video_key
            )
        video_paths[video_key] = str(video_path)

    episode_uuid = str(uuid4())

    # Load frames data
    df = lerobot_wrapper.load_episode_parquet(episode_index)

    # Process annotations
    annotations = [col_name for col_name in list(df.columns) if col_name.startswith("annotation.")]

    def _combine_annotations(row):
        annotation_dict = {}
        for col in annotations:
            task_name = col.split("annotation.")[-1]
            if pd.notna(row[col]):
                annotation_dict[task_name] = lerobot_wrapper.get_task_str_from_index(row[col])
        return json.dumps(annotation_dict)

    # Process frame data
    df["annotation"] = df.apply(_combine_annotations, axis=1)
    df["task"] = df["task_index"].apply(lambda x: lerobot_wrapper.get_task_str_from_index(x))

    # Add frame indices
    df["episode_frame_index"] = range(len(df))
    df["frame_index"] = range(global_frame_index, global_frame_index + len(df))

    # Clean up columns
    cols_to_drop = annotations + [
        "task_index", "next.done", "next.reward", "episode_index", "index",
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.rename(columns={"observation.state": "observation_state"})
    df["episode_id"] = episode_uuid
    df["dataset_id"] = dataset_id

    # Create episode record with video paths
    current_episode = {
        "_id": episode_uuid,
        "episode_index": episode_index,
        "task": df["task"].iloc[0],
        "length": len(df),
        "dataset_id": dataset_id,
        "video_paths": json.dumps(video_paths),
        "start_frame_index": global_frame_index,
        "end_frame_index": global_frame_index + len(df) - 1,
    }

    return current_episode, df, global_frame_index + len(df)


def write_to_lance(
    lance_output_path: str,
    dataset_record: Dict[str, Any],
    episodes_data: List[Dict[str, Any]],
    frames_data: pd.DataFrame
):
    """Write processed robotics data to Lance format tables.

    Creates three Lance tables:
    - datasets: Metadata about the dataset
    - episodes: Per-episode information and video paths
    - frames: Individual frame data with observations and actions
    """
    output_path = Path(lance_output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create datasets table from list of records
    dataset_table = pa.Table.from_pylist([dataset_record], schema=DATASET_SCHEMA)
    dataset_path = output_path / DATASETS_TABLE_NAME
    print(f"Writing dataset to {dataset_path}")
    lance.write_dataset(dataset_table, str(dataset_path))

    # Create episodes table
    episodes_table = pa.Table.from_pylist(episodes_data, schema=EPISODE_SCHEMA)
    episodes_path = output_path / EPISODES_TABLE_NAME
    print(f"Writing {len(episodes_data)} episodes to {episodes_path}")
    lance.write_dataset(episodes_table, str(episodes_path))

    # Create frames table
    frames_table = pa.Table.from_pandas(frames_data, schema=FRAME_SCHEMA)
    frames_path = output_path / FRAMES_TABLE_NAME
    print(f"Writing {len(frames_data)} frames to {frames_path}")
    lance.write_dataset(
        frames_table,
        str(frames_path),
        max_rows_per_group=8192,  # Optimize for performance
        max_rows_per_file=1024*1024
    )


def main(config: LeRobotToLanceConfig):
    """Main conversion function."""
    print(f"Starting ingestion of {config.source_lerobot_dataset_path}")

    # Initialize wrapper
    lerobot_wrapper = LeRobotWrapper(config.source_lerobot_dataset_path)

    # Create dataset record
    dataset_record = create_dataset_record(lerobot_wrapper)
    dataset_id = dataset_record["_id"]

    # Determine how many episodes to process
    num_episodes = lerobot_wrapper.num_episodes
    if config.max_episodes > 0:
        num_episodes = min(config.max_episodes, lerobot_wrapper.num_episodes)
        print(f"Limiting to first {num_episodes} episodes out of {lerobot_wrapper.num_episodes}")

    # Process episodes and frames
    episodes_data = []
    frames_buffer = None
    global_frame_index = 0

    print(f"Processing {num_episodes} episodes")

    for ep_idx in tqdm(range(num_episodes)):
        try:
            current_episode, processed_df, global_frame_index = process_frames_and_episodes_data(
                ep_idx, lerobot_wrapper, dataset_id, global_frame_index
            )

            episodes_data.append(current_episode)

            if frames_buffer is None:
                frames_buffer = processed_df
            else:
                frames_buffer = pd.concat([frames_buffer, processed_df])

        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}. Skipping...")
            continue

    if frames_buffer is not None and len(episodes_data) > 0:
        print(f"Writing {len(episodes_data)} episodes and {len(frames_buffer)} frames to Lance")
        write_to_lance(config.lance_output_path, dataset_record, episodes_data, frames_buffer)
        print("Ingestion completed successfully")
    else:
        print("No data was processed successfully")


if __name__ == "__main__":
    config = tyro.cli(LeRobotToLanceConfig)
    main(config)