from nuscenes.nuscenes import NuScenes
import os
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np


def load_lidarseg_bin(file_path: str) -> np.ndarray:
    """
    加载 lidar segmentation 标签文件（.bin），返回 numpy 数组。
    """
    try:
        return np.fromfile(file_path, dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ 读取 lidarseg 文件失败: {file_path} — {e}")
        return np.array([], dtype=np.uint8)


def extract_sample(
    nusc: NuScenes, sample_token: str, data_root: str
) -> dict:  # 输入是 nusc 的数据接口，场景的 ID，加载路径
    sample_dict = {}
    sample = nusc.get(
        "sample", sample_token
    )  # 下面是基础信息，从基础信息中获取场景 id，和发生的时间
    sample_dict["sample_token"] = sample["token"]
    sample_dict["timestamp"] = sample["timestamp"]

    for sensor, token in sample[
        "data"
    ].items():  # 遍历场景收集到的信息，遍历顺序是访问每个传感器的名字，获取传感器的数据
        sensor_data = nusc.get("sample_data", token)

        for (
            k,
            v,
        ) in (
            sensor_data.items()
        ):  # 如果不是 id 的内容，按照传感器和属性名的方式加入到字典中来，这个相当于处理了一个表 sample_data
            if "token" not in k and k not in {
                "ego_pose_token",
                "calibrated_sensor_token",
                "filename",
                "prev",
                "next",
            }:
                sample_dict[f"{sensor}-{k}"] = v

        ego_pose = nusc.get(
            "ego_pose", sensor_data["ego_pose_token"]
        )  # 读取 ego_pose 表
        for k, v in ego_pose.items():
            if "token" not in k:
                sample_dict[f"{sensor}-ego_pose-{k}"] = v

        calib = nusc.get(
            "calibrated_sensor", sensor_data["calibrated_sensor_token"]
        )  # 读取 calibrated_sensor 表
        for k, v in calib.items():
            if "token" not in k:
                sample_dict[f"{sensor}-calibrated_sensor-{k}"] = v

        file_path = os.path.join(data_root, sensor_data["filename"])  # 读取图像
        with open(file_path, "rb") as f:
            binary = f.read()
            sample_dict[f"{sensor}-file"] = binary

    anns = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)  # 又一个表
        ann_dict = {
            f"ann-{k}": v
            for k, v in ann.items()
            if "token" not in k and k not in {"prev", "next"}
        }

        inst = nusc.get("instance", ann["instance_token"])  # 一个表
        for k, v in inst.items():
            if "token" not in k and k != "category_token":
                ann_dict[f"instance-{k}"] = v

        category = nusc.get("category", inst["category_token"])  # 一个表
        ann_dict["category"] = category

        attributes = []
        for attr_token in ann["attribute_tokens"]:
            attr = nusc.get("attribute", attr_token)
            attr_filtered = {
                k: v for k, v in attr.items() if "token" not in k
            }  # 一个表
            attributes.append(attr_filtered)
        ann_dict["attributes"] = attributes

        anns.append(ann_dict)

    sample_dict["sample_annotations"] = anns  # 一个表
    return sample_dict


def extract_sample_all(nusc: NuScenes, sample_token: str, data_root: str) -> dict:
    sample = nusc.get("sample", sample_token)
    sample_dict = {
        "sample_token": sample["token"],
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
        "prev": sample["prev"],
        "next": sample["next"],
    }

    # 提取 sensor 数据（图像/点云） + 标定 + ego pose + 文件内容
    for sensor_name, sd_token in sample["data"].items():
        sensor_data = nusc.get("sample_data", sd_token)  # 表 1
        sensor_prefix = f"{sensor_name}"

        # 基本字段
        for k, v in sensor_data.items():
            if k not in {
                "token",
                "filename",
                "ego_pose_token",
                "calibrated_sensor_token",
                "next",
                "prev",
            }:
                sample_dict[f"{sensor_prefix}-sample_data-{k}"] = v

        # 文件内容（图像、点云）
        file_path = os.path.join(data_root, sensor_data["filename"])  # 表 2
        with open(file_path, "rb") as f:
            sample_dict[f"{sensor_prefix}-file"] = f.read()

        # ego pose
        ego_pose = nusc.get("ego_pose", sensor_data["ego_pose_token"])  # 表 3
        for k, v in ego_pose.items():
            if k != "token":
                sample_dict[f"{sensor_prefix}-ego_pose-{k}"] = v

        # 标定
        calib = nusc.get(
            "calibrated_sensor", sensor_data["calibrated_sensor_token"]
        )  # 表 4
        for k, v in calib.items():
            if k not in {"token", "sensor_token"}:
                sample_dict[f"{sensor_prefix}-calib-{k}"] = v

        # 原始 sensor 类型
        sensor_info = nusc.get("sensor", calib["sensor_token"])  # 表 5
        for k, v in sensor_info.items():
            if k != "token":
                sample_dict[f"{sensor_prefix}-sensor-{k}"] = v

        # lidarseg（如果是 lidar 点云）
        if "LIDAR" in sensor_name.upper():
            try:
                seg = nusc.get("lidarseg", sd_token)  # 表 6
                seg_path = os.path.join(data_root, seg["filename"])
                sample_dict[f"{sensor_prefix}-lidarseg"] = load_lidarseg_bin(seg_path)
            except Exception:
                sample_dict[f"{sensor_prefix}-lidarseg"] = None

    # 提取 annotations（含 instance、attribute、category、visibility）
    annotations = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)  # 表 7
        ann_info = {
            "translation": ann["translation"],
            "size": ann["size"],
            "rotation": ann["rotation"],
            "num_lidar_pts": ann["num_lidar_pts"],
            "num_radar_pts": ann["num_radar_pts"],
        }

        # 可见性
        if ann["visibility_token"]:
            visibility = nusc.get("visibility", ann["visibility_token"])  # 表 8
            ann_info["visibility_level"] = visibility["level"]
            ann_info["visibility_description"] = visibility["description"]

        # 属性列表
        attributes = []
        for attr_token in ann.get("attribute_tokens", []):
            attr = nusc.get("attribute", attr_token)  # 表 9
            attributes.append(attr["name"])
        ann_info["attributes"] = attributes

        # instance 信息
        instance = nusc.get("instance", ann["instance_token"])
        ann_info["instance_nbr_annotations"] = instance["nbr_annotations"]  # 表 10

        # 类别
        category = nusc.get("category", instance["category_token"])  # 表 11
        ann_info["category_name"] = category["name"]

        annotations.append(ann_info)

    sample_dict["sample_annotations"] = annotations

    # log + map（场景级别）
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])  # 表 12
    sample_dict["log_location"] = log["location"]
    sample_dict["log_vehicle"] = log["vehicle"]
    sample_dict["log_date"] = log["date_captured"]

    # map 加载文件路径
    maps = (
        nusc.map
        if hasattr(nusc, "map")  # 表 13
        else [NuScenesMap(dataroot=data_root, map_name=f"maps/{log['location']}.json")]
    )
    if maps:
        sample_dict["map_available"] = True
        sample_dict["map_location"] = log["location"]
    else:
        sample_dict["map_available"] = False

    return sample_dict
