#!/usr/bin/env python3
#
# Generate HD-Vila-100M dataset
#
# https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m


import argparse
import datetime
import logging
from collections import defaultdict
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Any, Generator, List, Tuple

import pyarrow as pa
import ray
import yt_dlp as youtube_dl
from lance.ray.sink import _register_hooks

FORMAT_IDS = {
    "720p": "22",
}

_register_hooks()


ZERO_DATETIME = datetime.datetime.strptime("00:00:00.000", "%H:%M:%S.%f")


SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("clip_id", pa.string()),
        pa.field("start", pa.float32()),
        pa.field("duration", pa.float32()),
        pa.field("video", pa.large_binary()),
    ]
)


def parse_span(span_str: List[str]) -> Tuple[float, float]:
    """Return (start, duration)"""
    start, end = [datetime.datetime.strptime(s, "%H:%M:%S.%f") for s in span_str]
    duration = (end - start).total_seconds()
    start = (start - ZERO_DATETIME).total_seconds()
    return (start, duration)


def download_youtube(
    video_id: str,
    output_dir: Path,
    format_id: int = FORMAT_IDS["720p"],
    ext: str = "mp4",
) -> Path | None:
    """Return the path of downloaded video."""
    dl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "merge_output_format": ext,
        "format": format_id,
        "skip_download": False,
        "ignoreerrors": True,
        "quiet": True,
        "max_sleep_interval": 15,
    }

    with youtube_dl.YoutubeDL(dl_opts) as ydl:
        rst = ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if rst != 0:
            logging.error("Failed to download video %s", video_id)
            return None
        return output_dir / f"{video_id}.{ext}"


def cut_clip(video_path: Path, start: float, duration: float, output_path: Path):
    """Cut video clip.

    Parameters
    ----------
    video_path : Path
        Path to video file
    start : float
        Start time in seconds
    duration : float
        Duration in seconds
    output_path : Path
        Output path
    """
    check_call(
        [
            "ffmpeg",
            "-y",  # Overwrite
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-loglevel",
            "quiet",
            str(output_path),
        ]
    )


def transform_clip(
    batch: pa.Table, cut_clips: bool = False
) -> Generator[None, dict[str, Any], None]:
    """Transform a row of the dataset."""
    for idx, video_id in enumerate(batch["video_id"]):
        results = defaultdict(list)
        with TemporaryDirectory() as dl_dir:
            video_path = download_youtube(video_id, Path(dl_dir))
            if video_path is None:
                continue
            if cut_clips:
                for clip in batch["clip"][idx]:
                    clip_id = clip["clip_id"]
                    start, duration = parse_span(clip["span"])
                    logging.debug(
                        "Cutting %s: start=%s duration=%s", video_id, start, duration
                    )
                    clip_file: Path = Path(dl_dir) / f"{clip_id}.mp4"

                    cut_clip(video_path, start, duration, clip_file)
                    with clip_file.open("rb") as f:
                        clip_video = f.read()
                    results["video_id"].append(video_id)
                    results["clip_id"].append(clip_id)
                    results["start"].append(start)
                    results["duration"].append(duration)
                    results["video"].append(clip_video)
            else:
                # DO not cut video to get large enough blobs
                results["video_id"].append(video_id)
                results["clip_id"].append(video_id)
                results["start"].append(0)
                results["duration"].append(0)
                with video_path.open("rb") as f:
                    results["video"].append(f.read())
        yield results


def dataset_generator(args):
    obj_store_memory = args.ray_object_store_memory_gb * 1024**3
    ray.init(object_store_memory=obj_store_memory)
    context = ray.data.DataContext.get_current()
    context.execution_options.resource_limits.object_store_memory = obj_store_memory

    ds = ray.data.read_json(args.input, override_num_blocks=8)
    if args.limit:
        ds = ds.limit(args.limit)
    ds.repartition(8).materialize().write_lance(
        args.output,
        transform=transform_clip,
        max_rows_per_file=args.max_rows_per_file,
        max_bytes_per_file=args.max_gb_per_file * 1024**3,
        schema=SCHEMA,
        storage_options={"timeout": "10m"},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HD-Vila-100M dataset in Lance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory",
        default="hd-vila.lance",
        metavar="DIR",
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="limit number of video to proceed",
        default=None,
        metavar="NUM",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        help="How many Ray workers to use",
        default=8,
        type=int,
        metavar="NUM",
    )
    parser.add_argument(
        "-b", "--batch", help="batch size", default=10, type=int, metavar="NUM"
    )
    parser.add_argument(
        "--max-rows-per-file",
        help="max rows per file",
        default=4 * 1024,
        type=int,
        metavar="NUM",
    )
    parser.add_argument(
        "--max-gb-per-file",
        help="max GB per file",
        default=1024,
        type=int,
        metavar="NUM",
    )
    parser.add_argument(
        "--ray-object-store-memory-gb",
        type=int,
        default=8,
        metavar="NUM",
        help="Ray object store memory in GB",
    )
    parser.add_argument(
        "input",
        help="Input files (HD-Vila-100M dataset)",
        nargs="+",
        metavar="PART.JSONL",
    )

    args = parser.parse_args()

    dataset_generator(args)


if __name__ == "__main__":
    main()
