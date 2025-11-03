import random


def get_samples_in_scene(nusc, scene):
    sample_token = scene["first_sample_token"]
    sample_tokens = []
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sample_tokens.append(sample_token)
        sample_token = sample["next"]
    return sample_tokens


def select_random_scene_samples(nusc, max_samples=10):
    scene = random.choice(nusc.scene)
    all_tokens = get_samples_in_scene(nusc, scene)
    if len(all_tokens) <= max_samples:
        return scene, all_tokens
    start = random.randint(0, len(all_tokens) - max_samples)
    return scene, all_tokens[start : start + max_samples]
