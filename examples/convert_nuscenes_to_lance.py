import argparse
from nuscenes.nuscenes import NuScenes
from extractor import extract_sample_all
from dataset_utils import select_random_scene_samples
from schema_builder import build_arrow_schema
from lance_writer import write_batch_to_lance
import pyarrow as pa
from batch_convert import convert_all_scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_root", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--lance_root", required=True)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--compression_algo", default="zstd")
    parser.add_argument("--compression_level", default="22")
    parser.add_argument("--all", action="store_true", help="If set, convert all scenes")
    parser.add_argument(
        "--per_scene",
        action="store_true",
        help="Save each scene to individual lance file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for memory-safe writes"
    )

    args = parser.parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)

    if args.all:
        convert_all_scenes(
            nusc=nusc,
            data_root=args.nusc_root,
            lance_root=args.lance_root,
            compression_algo=args.compression_algo,
            compression_level=args.compression_level,
            per_scene=args.per_scene,
            batch_size=args.batch_size,
        )
        return

    # 仅转一个随机 scene（默认）
    scene, sample_tokens = select_random_scene_samples(nusc, args.max_samples)
    print(f"🎬 Selected scene {scene['name']} with {len(sample_tokens)} samples")

    records = [
        extract_sample_all(nusc, token, args.nusc_root) for token in sample_tokens
    ]
    schema = build_arrow_schema(records, args.compression_algo, args.compression_level)
    table = pa.Table.from_pylist(records, schema=schema)
    write_batch_to_lance(table, args.lance_root)
    print(f"✅ Saved {len(records)} records to {args.lance_root}")


if __name__ == "__main__":
    main()
