# 🚗 nuScenes to Lance Converter

This repository provides a full pipeline to convert the [nuScenes](https://www.nuscenes.org/) autonomous driving dataset into the [Lance](https://lancedb.github.io/lance/) columnar format for high-performance querying and machine learning.

It extracts all relevant metadata and binary content from each `sample`, organizing them into one fragment per scene.

---

## 📦 Features

- Converts entire nuScenes dataset into Lance format.
- Stores per-sample info: metadata, sensor data, poses, calibrations, annotations, and files.
- Uses Arrow schema with optional ZSTD compression.
- Each scene becomes a Lance fragment → enabling fast queries like:
  - Random X consecutive frames from a scene
  - Iterate through a full scene

---

## 📁 Output Structure

The converter writes one Lance fragment per scene, where each row represents a frame (sample), including:

- Sample metadata
- All sensor data (`CAM_FRONT`, `LIDAR_TOP`, etc.)
- Ego pose & calibration data
- Binary file content (images, point clouds)
- Annotations (bounding boxes, category, attributes)

Example structure:

```
/path/to/lance_output/
├── data-00000.lance   # Scene fragment
└── data-00001.lance
```

---

## 1. 📥 Installation

Install required dependencies:

```bash
pip install nuscenes-devkit pyarrow lance
```

---

## 2. 🚀 Run the Converter

### Syntax

```bash
python convert_nuscenes_to_lance.py \
    <nuscenes_root> \
    <version> \
    <lance_output_path> \
    --compression_algo <algo> \
    --compression_level <level>
```

### Parameters

```markdown
- nuscenes_root: The root directory of the nuScenes dataset
- version: The dataset version (e.g. 'v1.0-mini', 'v1.0-trainval')
- lance_output_path: Output directory to store the Lance dataset
- --compression_algo: (Optional) Compression algorithm, e.g., 'zstd'
- --compression_level: (Optional) Compression level for Lance (default: 22)
```

### Example

```bash
python convert_nuscenes_to_lance.py --all --per_scene  --nusc_root ./v1.0-mini \
  --version v1.0-mini   \
  --lance_root ./lance_output/
```

---

## 💡 Example Query Use Cases

- Sample 10 random consecutive frames from a random scene
- Iterate over all frames in one scene
- Perform nearest-neighbor search on sensor embeddings
- Visualize frames with high object count or specific categories

---

## 📚 References

- [nuScenes Dataset](https://www.nuscenes.org/)
- [nuScenes DevKit](https://github.com/nutonomy/nuscenes-devkit)
- [Lance Format](https://lancedb.github.io/lance/)
- [Apache Arrow](https://arrow.apache.org/)

---

## 👤 Maintainer

| Name  | GitHub                             |
| ----- | ---------------------------------- |
| slyrx | [@slyrx](https://github.com/slyrx) |

---

## 📝 License

This project is licensed under the **MIT License**.
