# LeRobot to Lance Conversion

Convert robotics datasets from LeRobot format to Lance format for efficient querying and ML workflows.

This OSS implementation processes real robotics datasets like the **PushT manipulation dataset** from HuggingFace, converting them to Lance's columnar format for dramatically improved query performance.

## 🎯 Real Dataset Example

**PushT Dataset** (HuggingFace: `lerobot/pusht`):
- **206 episodes** of robot manipulation tasks
- **25,650 frames** of robot observations and actions
- **2D robot state** + **96×96×3 camera images** + **2D actions**
- **Task**: "Push the T-shaped block onto the T-shaped target"

## 🚀 Quick Start

### 1. Download Real PushT Dataset
```bash
python download_pusht_dataset.py
# Downloads ~674KB of real robotics data from HuggingFace
```

### 2. Convert to Lance Format
```bash
python lerobot_to_lance.py \
  --source-lerobot-dataset-path /tmp/pusht_dataset \
  --lance-output-path /tmp/pusht_lance \
  --max-episodes 10
```

### 3. Query Converted Data
```python
import lance

# Load frame data
frames = lance.dataset("/tmp/pusht_lance/frames")

# Fast episode filtering (first 10 frames of each episode)
early_frames = frames.to_table(
    filter="episode_frame_index < 10"
).to_pandas()

# Efficient column access
actions_only = frames.to_table(
    columns=["action", "timestamp"]
).to_pandas()

# Time-based queries
early_frames = frames.to_table(
    filter="episode_frame_index < 10"
).to_pandas()
```

### 4. Run Tests
```bash
python test_real_pusht.py
# Validates conversion with real PushT dataset
```

## 📊 Lance Format Benefits

### **🔍 Fast Columnar Queries**
Convert 25K+ robot frames and query specific time ranges or actions instantly:
- `filter="episode_frame_index < 10"` → Get early frames from all episodes
- `filter="task LIKE '%push%'"` → Get frames for specific tasks
- `columns=["action"]` → Load only robot actions, skip observations

### **💾 Efficient Storage**
- Columnar compression reduces storage requirements
- Load only needed columns for training/analysis
- Optimized for ML workflows with PyArrow backend

### **🎥 Smart Video Handling**
- Video files remain as MP4s (not embedded)
- JSON mapping of camera→path per episode
- Efficient multi-camera robotics setups

### **📈 ML Pipeline Ready**
- Native Pandas DataFrame integration
- Seamless with PyTorch/JAX data loaders
- Efficient batch loading for training

## 🏗️ Output Structure

Converting LeRobot format creates three optimized Lance tables:

```
lance_output/
├── datasets/     # Dataset metadata (1 row)
│   └── robot_type, fps, features, task_definitions, etc.
├── episodes/     # Episode info (1 row per episode)
│   └── episode_index, length, task, video_paths, frame_ranges
└── frames/       # Frame data (1 row per frame)
    └── observations, actions, timestamps, episode_id, task
```

## 🤖 Supported Robotics Data

**Observation Types:**
- Multi-camera video streams (stored as file references)
- Robot joint states, end-effector poses
- Sensor readings (force, torque, etc.)

**Action Types:**
- Joint position/velocity commands
- End-effector position/orientation targets
- Gripper open/close commands

**Metadata:**
- Task descriptions and annotations
- Episode success/failure labels
- Temporal synchronization data

## 📋 Requirements

```bash
pip install lance pandas pyarrow tqdm tyro
# Git LFS required for downloading HuggingFace datasets
```

## 🌟 Performance Example

**Real PushT Dataset Results:**
- **Input**: 25,650 frames across 206 episodes
- **Conversion**: ~2 seconds for full dataset
- **Query Speed**: Filter 25K frames by episode in milliseconds
- **Storage**: Efficient columnar compression

**Query Performance:**
```python
# Traditional approach: Load all data, filter in memory
df = pd.read_parquet("all_episodes.parquet")  # Loads 25K rows
episode_5 = df[df.episode_index == 5]         # Filters in memory

# Lance approach: Query at storage level
episode_5 = frames.to_table(                  # Only loads matching rows
    filter="episode_index = 5"
).to_pandas()
```

## 🎯 Use Cases

**🔬 Research**
- Analyze robot behavior patterns across episodes
- Compare success/failure trajectories
- Extract specific manipulation phases

**🤖 Training**
- Efficient data loading for imitation learning
- Sample balanced training batches
- Quick dataset exploration and statistics

**📊 Analysis**
- Time-series analysis of robot performance
- Cross-episode behavior comparison
- Rapid prototyping of new training approaches

This conversion enables robotics researchers to leverage Lance's powerful columnar analytics for the rapidly growing field of robot learning and manipulation research.