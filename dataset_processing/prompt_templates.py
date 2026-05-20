"""Prompt generator for LoRA training, driven by dashboard prompt_config.

Loaded as a custom_metadata_module by stable-audio-tools.
The dataset loader calls set_config(dataset_config) before training starts,
passing the full dataset JSON which includes prompt_config from the dashboard.

If no prompt_config is present, falls back to the same behavior as songs_simple.py.
"""

import os
import random

_prompt_config = None

_TAG_DISPLAY = {
    "title": "Title", "artist": "Artist", "album": "Album",
    "genre": "Genre", "label": "Label", "date": "Year",
    "composer": "Composer", "bpm": "BPM", "prompt": "Prompt",
}
_ALL_TAG_KEYS = list(_TAG_DISPLAY.keys())


def set_config(dataset_config):
    """Called by dataset loader with the full dataset config JSON."""
    global _prompt_config
    _prompt_config = dataset_config.get("prompt_config", None)


def _get(metadata, key):
    val = metadata.get(key, "")
    if isinstance(val, (list, tuple)):
        val = val[0] if val else ""
    return str(val).strip()


# ── Tag-based prompts ────────────────────────────────────────────────

def _build_tag_prompt(metadata, pc):
    tag_keys = pc.get("tag_keys", _ALL_TAG_KEYS)
    hide_names = pc.get("hide_tag_names", False)
    hide_commas = pc.get("hide_commas", False)
    split_commas = pc.get("split_commas", False)

    parts = []
    for key in tag_keys:
        val = _get(metadata, key)
        if not val:
            continue
        label = _TAG_DISPLAY.get(key, key)
        if split_commas and "," in val:
            for sub in val.split(","):
                sub = sub.strip()
                if sub:
                    parts.append(sub if hide_names else f"{label}: {sub}")
        else:
            parts.append(val if hide_names else f"{label}: {val}")

    if not parts:
        return ""

    if pc.get("shuffle", True):
        # 50% shuffle all, 50% random subset
        if random.random() < 0.5:
            random.shuffle(parts)
        else:
            parts = random.sample(parts, random.randint(1, len(parts)))

    return (" " if hide_commas else ", ").join(parts)


# ── Path-based prompts ───────────────────────────────────────────────

def _build_path_prompt(metadata, pc):
    relpath = _get(metadata, "relpath")
    if not relpath:
        return ""

    path_opts = pc.get("path_opts", {})
    # Frontend sends camelCase keys
    hide_ext = path_opts.get("hideExt", path_opts.get("hide_ext", False))
    hide_dirs = path_opts.get("hideDirs", path_opts.get("hide_dirs", False))
    hide_topmost = path_opts.get("hideTopmostDir", path_opts.get("hide_topmost_dir", False))

    parts = relpath.replace("\\", "/").split("/")
    filename = parts[-1]
    dirs = parts[:-1]

    if hide_ext:
        dot = filename.rfind(".")
        if dot > 0:
            filename = filename[:dot]

    if hide_dirs:
        if hide_topmost or not dirs:
            return filename
        return dirs[0] + "/" + filename

    if hide_topmost and dirs:
        return os.path.join(*dirs[1:], filename) if len(dirs) > 1 else filename

    return os.path.join(*dirs, filename) if dirs else filename


# ── Legacy fallback (no prompt_config) ───────────────────────────────

def _legacy_prompt(metadata):
    properties = []
    for key in _ALL_TAG_KEYS:
        val = _get(metadata, key)
        if val:
            properties.append(f"{_TAG_DISPLAY.get(key, key)}: {val}")
    if not properties:
        return metadata.get("text", "")
    if random.random() < 0.5:
        random.shuffle(properties)
    else:
        properties = random.sample(properties, random.randint(1, len(properties)))
    return ", ".join(properties)


# ── Main entry point ─────────────────────────────────────────────────

def get_custom_metadata(metadata, audio):
    pc = _prompt_config

    if pc is None:
        return {"prompt": _legacy_prompt(metadata), "lyrics": ""}

    use_tags = pc.get("use_tags", True)
    use_paths = pc.get("use_paths", False)
    use_fixed = pc.get("use_fixed", False)
    fixed_text = pc.get("fixed_text", "")
    balance = pc.get("balance", {})
    trigger = pc.get("trigger", "")
    trigger_pct = pc.get("trigger_pct", 80)

    # Build candidate prompts for each enabled method
    candidates = []  # (method_name, prompt_text)
    weights = []

    if use_tags:
        candidates.append(("tags", _build_tag_prompt(metadata, pc)))
        weights.append(balance.get("tags", 50))

    if use_paths:
        candidates.append(("paths", _build_path_prompt(metadata, pc)))
        weights.append(balance.get("paths", 50))

    if use_fixed:
        candidates.append(("fixed", fixed_text))
        weights.append(balance.get("fixed", 0))

    # Pick one method based on balance weights
    chosen_method = None
    prompt = ""
    if not candidates:
        prompt = metadata.get("text", "")
    else:
        total = sum(weights)
        if total <= 0:
            chosen_method, prompt = random.choice(candidates)
        else:
            chosen_method, prompt = random.choices(candidates, weights=weights, k=1)[0]

    # Prepend trigger token based on trigger_pct
    if trigger and trigger_pct > 0 and random.random() * 100 < trigger_pct:
        if prompt:
            use_comma = chosen_method == "tags" and not pc.get("hide_commas", False)
            prompt = trigger + (", " if use_comma else " ") + prompt
        else:
            prompt = trigger

    return {"prompt": prompt, "lyrics": ""}
