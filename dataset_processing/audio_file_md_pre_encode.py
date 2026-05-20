import json
import os

import audio_metadata
import warnings

warnings.filterwarnings("ignore", module="audio_metadata")

TAG_KEYS = ["title", "artist", "album", "genre", "label", "date", "composer", "bpm"]


def get_custom_metadata(metadata, audio):
    """Extract tag fields from an audio file for pre-encoding.

    Checks JSON sidecars first ({stem}.json same dir or json/ subdir),
    then falls back to embedded ID3/Vorbis tags.
    Saves raw tag values as individual keys in the metadata dict.
    Prompt building from these fields is deferred to training time
    (e.g. via prompt_templates.py).
    """
    filepath = metadata["path"]
    properties = {}

    # Check for JSON sidecar first
    stem = os.path.splitext(os.path.basename(filepath))[0]
    parent = os.path.dirname(filepath)
    sidecar_candidates = [
        os.path.join(parent, stem + ".json"),
        os.path.join(os.path.dirname(parent), "json", stem + ".json"),
    ]
    for sc_path in sidecar_candidates:
        if os.path.isfile(sc_path):
            try:
                with open(sc_path) as f:
                    sc = json.load(f)
                for k, v in sc.items():
                    if v and isinstance(v, (str, int, float)):
                        properties[k] = str(v)
                if properties:
                    return properties
            except Exception:
                pass

    # Fall back to embedded audio tags
    try:
        track_metadata = audio_metadata.load(filepath)
    except Exception as e:
        print(f"Couldn't load metadata for {filepath}: {e}")
        return properties

    if "tags" in track_metadata:
        tags = track_metadata["tags"]
        for key in TAG_KEYS:
            if key in tags:
                val = tags[key]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = str(val[0])
                else:
                    val = str(val)
                if val:
                    properties[key] = val

    return properties
