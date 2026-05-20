import random


def get_custom_metadata(metadata, audio):
    """Build a randomized prompt from pre-encoded tag fields.

    Expects tag values as plain strings (from our pre_encode.py),
    not lists (like the original songs_simple.py expected from WebDataset).
    """
    properties = []

    def _get(key):
        val = metadata.get(key, "")
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ""
        return str(val).strip()

    artist = _get("artist")
    if artist:
        properties.append(f"Artist: {artist}")

    title = _get("title")
    if title:
        properties.append(f"Title: {title}")

    date = _get("date")
    if date:
        properties.append(f"Year: {date}")

    bpm = _get("bpm")
    if bpm:
        properties.append(f"BPM: {bpm}")

    genre = _get("genre")
    if genre:
        properties.append(f"Genre: {genre}")

    label = _get("label")
    if label:
        properties.append(f"Label: {label}")

    album = _get("album")
    if album:
        properties.append(f"Album: {album}")

    composer = _get("composer")
    if composer:
        properties.append(f"Composer: {composer}")

    if len(properties) == 0:
        prompt = metadata.get("text", "")
    else:
        # 50% shuffle all, 50% random subset
        if random.random() < 0.5:
            random.shuffle(properties)
        else:
            properties = random.sample(properties, random.randint(1, len(properties)))
        prompt = ", ".join(properties)

    return {"prompt": prompt, "lyrics": ""}
