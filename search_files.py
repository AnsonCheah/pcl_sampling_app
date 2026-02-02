#!/usr/bin/env python3
"""search_files.py

Search a folder recursively for files whose names contain one or more
query strings from a queries.txt file.

Edit the configuration section below to customize behavior.
"""
from __future__ import annotations
import os
import shutil
import sys
from typing import Iterable, List

# ========== CONFIGURATION ==========
SEARCH_ROOT = "C:/FTP_SERVER/3D_FILES"  # Root folder to search (e.g., ".", "./mesh_raw", "C:\\path\\to\\folder")
QUERIES_FILE = "CE_part_list.txt"  # Path to file containing queries (one per line)
MATCH_ALL = False  # Set to True to require all queries, False for any match
IGNORE_CASE = False  # Set to True for case-insensitive matching
ALLOWED_EXTENSIONS = [".cgr", ".stl", ".stp", ".step", ".iges", ".igs"]  # File extensions to search
COPY_TO_FOLDER = "CGR"  # Set to folder path to copy matched files, None to skip copying
PRESERVE_STRUCTURE = False  # When copying, preserve directory structure relative to root
OVERWRITE_FILES = True  # When copying, overwrite existing files
# ====================================

def parse_queries_from_arg(queries_arg: Iterable[str]) -> List[str]:
    # Accept either multiple args or a single comma-separated string
    out: List[str] = []
    for q in queries_arg:
        q = q.strip()
        if not q:
            continue
        if "," in q:
            parts = [p.strip() for p in q.split(",") if p.strip()]
            out.extend(parts)
        else:
            out.append(q)
    return out


def read_queries_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    # ignore blank lines and comments starting with #
    return [l for l in lines if l and not l.startswith("#")]


def file_matches(filename: str, queries: List[str], match_all: bool, case_sensitive: bool) -> bool:
    if not case_sensitive:
        name = filename.lower()
        queries = [q.lower() for q in queries]
    else:
        name = filename

    if match_all:
        return all(q in name for q in queries)
    return any(q in name for q in queries)


def find_files(root: str, queries: List[str], match_all: bool = False, case_sensitive: bool = False, extensions: List[str] | None = None) -> List[str]:
    matches: List[str] = []
    # Normalize provided extensions (ensure leading dot, lower-case)
    if extensions:
        norm_exts = { (e.lower() if e.startswith('.') else '.' + e.lower()) for e in extensions }
    else:
        norm_exts = None

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            # Filter by extension if requested
            ext = os.path.splitext(fn)[1].lower()
            if norm_exts is not None and ext not in norm_exts:
                continue
            if file_matches(fn, queries, match_all, case_sensitive):
                matches.append(os.path.join(dirpath, fn))
    return matches


def main() -> None:
    # Read queries from file
    try:
        queries = read_queries_file(QUERIES_FILE)
    except FileNotFoundError:
        print(f"error: queries file not found: {QUERIES_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error reading queries file: {e}", file=sys.stderr)
        sys.exit(1)

    if not queries:
        print("error: no queries found in file", file=sys.stderr)
        sys.exit(1)

    # Derive output filename from queries file
    base_name = os.path.splitext(QUERIES_FILE)[0]
    output_file = f"{base_name}_results.txt"

    # Search for files
    results = find_files(SEARCH_ROOT, queries, match_all=MATCH_ALL, case_sensitive=not IGNORE_CASE, extensions=ALLOWED_EXTENSIONS)

    # Output results
    with open(output_file, "w", encoding="utf-8") as out:
        for p in results:
            out.write(p + "\n")
    print(f"results written to {output_file}")

    # Copy files if requested
    if COPY_TO_FOLDER:
        dest_root = COPY_TO_FOLDER
        try:
            os.makedirs(dest_root, exist_ok=True)
        except OSError as e:
            print(f"error: failed to create destination folder '{dest_root}': {e}", file=sys.stderr)
            sys.exit(1)

        abs_root = os.path.abspath(SEARCH_ROOT)
        for src in results:
            src_abs = os.path.abspath(src)
            if PRESERVE_STRUCTURE:
                try:
                    rel = os.path.relpath(src_abs, start=abs_root)
                except Exception:
                    rel = os.path.basename(src_abs)
                dest_path = os.path.join(dest_root, rel)
            else:
                dest_path = os.path.join(dest_root, os.path.basename(src_abs))

            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            if os.path.exists(dest_path) and not OVERWRITE_FILES:
                print(f"skipping existing file: {dest_path}", file=sys.stderr)
                continue

            try:
                shutil.copy2(src_abs, dest_path)
                print(f"copied: {dest_path}")
            except OSError as e:
                print(f"failed to copy '{src_abs}' -> '{dest_path}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
