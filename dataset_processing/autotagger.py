#!/usr/bin/env python3
"""Autotagger: generate JSON metadata sidecars from folder/filename structure.

Scans a directory of audio files and creates {stem}.json sidecars containing
artist, title, album, year, and genre fields — parsed from the folder names
and filenames. These sidecars are read by pre_encode.py during latent encoding
and become the training prompts.

Designed for artist collections organized as:
    root/
      Artist - Album Name (Year)/
        01 Title.wav
        Artist - Album - 02 Title.wav

Handles many naming conventions:
  - "Artist - Album" folder names (split on " - ")
  - Track numbers: "01.", "01 ", "1. ", leading digits
  - Filenames containing "Artist - Album - NN Title"
  - Year extraction from "(2025)" in folder names
  - Strips " - Masters", "Vinyl Masters ...", "EP Masters" suffixes
  - Falls back to filename stem as title when nothing else works

Usage:
    # Dry run (preview what would be written)
    python autotagger.py /path/to/audio-dir

    # Actually write sidecars
    python autotagger.py /path/to/audio-dir --write

    # Force overwrite existing sidecars
    python autotagger.py /path/to/audio-dir --write --force

    # Custom genre
    python autotagger.py /path/to/audio-dir --write --genre "Electronic"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".m4a", ".aac"}
MIN_FILE_SIZE = 4096


def find_audio_files(root):
    """Recursively find audio files, sorted by path."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("._"):
                continue
            fp = Path(dirpath) / fn
            if fp.suffix.lower() in AUDIO_EXTS:
                try:
                    if fp.stat().st_size < MIN_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(fp)
    files.sort()
    return files


def clean_album_name(raw):
    """Clean up album folder name: strip 'Masters', bitrate/format suffixes, etc."""
    name = raw
    # Strip common suffixes
    name = re.sub(r'\s*-?\s*Masters?\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+Vinyl\s+Masters?\s+\d+k\s+\d+bit\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+\d+k\s+\d+bit\s*$', '', name, flags=re.IGNORECASE)
    # Strip trailing " - " left over after removing suffix
    name = name.rstrip(' -')
    return name.strip()


def extract_year(text):
    """Extract a 4-digit year (1950-2039) from text like '(2025)' or just '2025'."""
    m = re.search(r'\((\d{4})\)', text)
    if m:
        y = int(m.group(1))
        if 1950 <= y <= 2039:
            return str(y)
    return ""


def strip_track_number(filename_stem):
    """Remove leading track number from a filename stem.

    Handles: "01 Title", "01. Title", "1. Title", "01 - Title"
    """
    # "01. Title" or "1. Title"
    s = re.sub(r'^\d+\.\s*', '', filename_stem)
    if s != filename_stem:
        return s.strip()
    # "01 - Title"
    s = re.sub(r'^\d+\s*-\s+', '', filename_stem)
    if s != filename_stem:
        return s.strip()
    # "01 Title" (only if starts with 1-3 digits followed by space and uppercase or non-digit)
    s = re.sub(r'^\d{1,3}\s+(?=[A-Z\(\[])', '', filename_stem)
    if s != filename_stem:
        return s.strip()
    return filename_stem.strip()


def clean_title(title):
    """Clean up a title: strip 'Master' suffix, trailing whitespace, etc."""
    title = re.sub(r'\s+Master\s*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+streaming\s*$', '', title, flags=re.IGNORECASE)
    return title.strip()


def parse_folder_name(folder_name):
    """Parse 'Artist - Album (Year)' from a folder name.

    Returns (artist, album, year).
    """
    year = extract_year(folder_name)

    # Remove year from the string for cleaner parsing
    clean = re.sub(r'\s*\(\d{4}\)\s*', ' ', folder_name).strip()
    clean = clean_album_name(clean)

    # Try splitting on " - " for "Artist - Album"
    parts = clean.split(" - ", 1)
    if len(parts) == 2 and len(parts[0].strip()) > 1 and len(parts[1].strip()) > 1:
        return parts[0].strip(), parts[1].strip(), year

    # No clear split — treat whole thing as album, no artist from folder
    return "", clean, year


def parse_filename(stem, folder_artist, folder_album):
    """Parse artist and title from a filename stem.

    Tries several patterns:
      1. "Artist - Album - NN Title"
      2. "Artist - Album - NN. Title"
      3. "NN. Artist - Title (..."
      4. "NN. Title"
      5. Bare "Title"

    Returns (artist, title) — artist may be empty if not found in filename.
    """
    # Pattern: "Artist - Album - NN Title" or "Artist - Album - NN. Title"
    # Also handles "Artist - Album - NN - Title" (e.g. Inner G tracks)
    # e.g. "Zebbler Encanti Experience - Freakquency - 03 Neuron Dialect"
    # e.g. "Zebbler Encanti Experience - Inner G - 05 - Vimana"
    m = re.match(r'^(.+?)\s+-\s+(.+?)\s+-\s+(\d{1,3})\.?\s*-?\s+(.+)$', stem)
    if m:
        artist = m.group(1).strip()
        # album = m.group(2).strip()  # we already have it from folder
        title = m.group(4).strip()
        # Strip leading " - " from title if present
        title = re.sub(r'^-\s*', '', title)
        return artist, clean_title(title)

    # Pattern: "NN. Artist - Title"
    # e.g. "1. Zebbler Encanti Experience - End Trance (FLY Remix) Master"
    m = re.match(r'^\d+\.?\s+(.+?)\s+-\s+(.+)$', stem)
    if m:
        artist = m.group(1).strip()
        title = m.group(2).strip()
        return artist, clean_title(title)

    # Pattern: "Artist - Title" (no track number)
    # e.g. "Zebbler Encanti Experience - IMADODIS"
    m = re.match(r'^(.+?)\s+-\s+(.+)$', stem)
    if m:
        candidate_artist = m.group(1).strip()
        title = m.group(2).strip()
        # Only use this if the candidate looks like an artist name (not a track number)
        if not re.match(r'^\d+$', candidate_artist):
            return candidate_artist, clean_title(title)

    # Just strip track number, use rest as title
    title = strip_track_number(stem)
    return "", clean_title(title)


def autotag_file(fpath, root):
    """Generate metadata tags for one audio file based on its path.

    Returns a dict with: title, artist, album, date (year).
    """
    rel = fpath.relative_to(root)
    parts = rel.parts  # e.g. ('Album Folder', 'subdir', 'file.wav')

    # The first directory component under root is the album folder
    if len(parts) >= 2:
        album_folder = parts[0]
    else:
        album_folder = ""

    folder_artist, folder_album, year = parse_folder_name(album_folder) if album_folder else ("", "", "")

    # Handle subfolder names (e.g. "Encanti 2005 to 2009" under "Demos and Unreleased")
    # or "bandcamp new" / "streaming new" under an album
    sub_artist = ""
    if len(parts) >= 3:
        subfolder = parts[1]
        # Check if subfolder looks like an artist/era grouping
        m = re.match(r'^(\w[\w\s]+?)\s+(\d{4})\s+to\s+(\d{4})$', subfolder)
        if m:
            sub_artist = m.group(1).strip()
        # Otherwise ignore subfolders like "bandcamp new", "streaming new"

    stem = fpath.stem
    file_artist, title = parse_filename(stem, folder_artist, folder_album)

    # Resolve artist: filename > subfolder > folder
    artist = file_artist or sub_artist or folder_artist

    # If album is "Demos and Unreleased" or "Singles", it's not a real album
    album = folder_album
    if album.lower() in ("demos and unreleased", "singles"):
        album = ""

    tags = {}
    if artist:
        tags["artist"] = artist
    if title:
        tags["title"] = title
    else:
        tags["title"] = stem  # absolute fallback
    if album:
        tags["album"] = album
    if year:
        tags["date"] = year

    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON metadata sidecars from folder/filename structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=str, help="Root directory of audio files")
    parser.add_argument("--write", action="store_true",
                        help="Actually write .json sidecars (default: dry run)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .json sidecars")
    parser.add_argument("--genre", type=str, default="",
                        help="Set genre for all tracks (e.g. 'Electronic')")
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}")
        sys.exit(1)

    audio_files = find_audio_files(root)
    if not audio_files:
        print(f"No audio files found in {root}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files in {root}\n")

    written = 0
    skipped = 0
    would_write = 0

    for fpath in audio_files:
        tags = autotag_file(fpath, root)
        if args.genre:
            tags["genre"] = args.genre

        json_path = fpath.with_suffix(".json")
        rel = fpath.relative_to(root)

        # Build preview prompt (same format as training)
        parts = []
        if tags.get("artist"):
            parts.append(f"Artist: {tags['artist']}")
        parts.append(f"Title: {tags.get('title', fpath.stem)}")
        if tags.get("date"):
            parts.append(f"Year: {tags['date']}")
        if tags.get("genre"):
            parts.append(f"Genre: {tags['genre']}")
        if tags.get("album"):
            parts.append(f"Album: {tags['album']}")
        prompt = ", ".join(parts)

        if args.write:
            if json_path.exists() and not args.force:
                skipped += 1
                continue
            with open(json_path, "w") as f:
                json.dump(tags, f, indent=2)
            written += 1
            print(f"  {rel}")
            print(f"    -> {prompt}")
        else:
            would_write += 1
            exists = " [EXISTS]" if json_path.exists() else ""
            print(f"  {rel}{exists}")
            print(f"    -> {prompt}")

    print()
    if args.write:
        print(f"Written: {written}, Skipped (existing): {skipped}")
    else:
        existing = sum(1 for f in audio_files if f.with_suffix(".json").exists())
        print(f"DRY RUN: would write {would_write} sidecars ({existing} already exist)")
        print(f"Run with --write to create them (--force to overwrite existing)")


if __name__ == "__main__":
    main()
