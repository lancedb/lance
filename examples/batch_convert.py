from nuscenes.nuscenes import NuScenes
from extractor import extract_sample_all
from dataset_utils import get_samples_in_scene
from schema_builder import build_arrow_schema
from lance_writer import write_batch_to_lance
import pyarrow as pa
import os
from tqdm import tqdm
from pathlib import Path


def convert_all_scenes(
    nusc: NuScenes,
    data_root: str,
    lance_root: str,
    compression_algo="zstd",
    compression_level="22",
    per_scene=False,
    batch_size=1000,
):
    Path(lance_root).mkdir(parents=True, exist_ok=True)

    if per_scene:
        # 每个 scene 写一个 lance 文件
        for scene in tqdm(nusc.scene, desc="Converting scenes individually"):
            scene_name = scene["name"].replace("/", "_")
            sample_tokens = get_samples_in_scene(nusc, scene)
            records = [
                extract_sample_all(nusc, token, data_root) for token in sample_tokens
            ]

            schema = build_arrow_schema(records, compression_algo, compression_level)
            table = pa.Table.from_pylist(records, schema=schema)
            out_path = os.path.join(lance_root, f"scene_{scene_name}.lance")
            write_batch_to_lance(table, out_path)
            print(f"✅ Saved scene {scene_name} to {out_path}")

    else:
        # 全部写入一个 lance 文件，分批提交
        all_records = []
        schema = None
        sample_count = 0

        for scene in tqdm(nusc.scene, desc="Converting scenes in batch"):
            sample_tokens = get_samples_in_scene(nusc, scene)
            for token in sample_tokens:
                record = extract_sample_all(nusc, token, data_root)
                all_records.append(record)
                sample_count += 1

                if len(all_records) >= batch_size:
                    if schema is None:
                        schema = build_arrow_schema(
                            all_records, compression_algo, compression_level
                        )
                    table = pa.Table.from_pylist(all_records, schema=schema)
                    write_batch_to_lance(table, lance_root, mode="append")
                    print(f"✅ Appended {len(all_records)} samples")
                    all_records = []

        if all_records:
            if schema is None:
                schema = build_arrow_schema(
                    all_records, compression_algo, compression_level
                )
            table = pa.Table.from_pylist(all_records, schema=schema)
            write_batch_to_lance(table, lance_root, mode="append")
            print(f"✅ Final append: {len(all_records)} samples")

        print(f"✅ Total samples processed: {sample_count}")
