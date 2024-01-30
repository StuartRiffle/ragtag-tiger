# RAG/TAG Tiger - files.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

import os, pathspec, shutil, time
from logging import raglog, log_verbose, log_error
from timer import time_since

def split_root_from_spec(spec):
    """Split off the root path from the wildcard part if possible"""
    for pos in range(len(spec)):
        if spec[pos] == "*" or spec[pos] == "?":
            sep_pos = spec.rfind(os.path.sep, 0, pos)
            if sep_pos >= 0:
                return spec[:sep_pos + 1], spec[sep_pos + 1:]
    return "", spec

def match_files_to_index(search_spec):
    """Find files matching a wildcard spec"""
    all_matches = []
    try:
        if os.path.isfile(search_spec):
            all_matches.append(search_spec)
        else:
            file_spec_root, file_spec = split_root_from_spec(search_spec)
            file_spec_pattern = file_spec.replace(os.path.sep, '/') # required by pathspec
            relative_pathspec = pathspec.PathSpec.from_lines('gitwildmatch', [file_spec_pattern])

            matches = relative_pathspec.match_tree(file_spec_root)
            matches = [os.path.join(file_spec_root, match) for match in matches]

            log_verbose(f"\t{len(matches)} files match \"{os.path.normpath(file_spec)}\" from \"{os.path.normpath(file_spec_root)}\"")
            all_matches.extend(matches)

    except Exception as e: log_error(e)
    return all_matches

def separate_files_by_extension(file_list, extensions):
    """Separate out files with certain extensions"""
    matching_files = set([f for f in file_list if os.path.splitext(f)[1].lower() in extensions])
    non_matching_files = set([f for f in file_list if not f in matching_files])
    return matching_files, non_matching_files

def clean_up_temporary_files(tempdir, verbose=False):
    if tempdir and os.file.exists(tempdir) and os.path.isdir(tempdir):
        info = f" \"{tempdir}\"" if verbose else ""
        raglog(f"Removing temporary folder{info}...")
        time_before = time.time()
        for reattempt_after_delay in [0, 2, 5, 10]:
            if reattempt_after_delay:
                log_verbose(f"\t...retrying in {reattempt_after_delay} seconds")
                time.sleep(reattempt_after_delay)
            try:
                shutil.rmtree(tempdir)
                tempdir = None
                break
            except Exception as e: 
                # Errors are expected sometimes if the OS has files open for virus scanning, etc.
                log_verbose(f"\tignoring error: {e}")
        if tempdir:
            log_error(f"couldn't remove temporary folder \"{tempdir}\"")
        else:
            log_verbose(f"\t...success ({time_since(time_before)})")

def strip_and_remove_comments(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not line.startswith('#')]
    return "\n".join(lines)

def resolve_stock_data_prefix(path):
    # If path starts with +/ replace that with the data folder at the repo root
    if path.replace(os.path.dirsep, "/").startswith("+/"):
        data_path = os.path.join(os.path.dirname(__file__), "data")
        path = os.path.join(data_path, path[1:])
    return path

def load_stock_data(path):
    fixedpath = cleanpath(resolve_stock_data_prefix(os.path.join('+', path)))
    try: 
        with open(fixedpath, "r", encoding="utf-8") as f: 
            return f.read()
    except Exception as e: 
        log_error(f"failed loading \"{fixedpath}\": {e}")
    return None

def load_stock_text(path):
    return str(load_stock_data(path))

def cleanpath(path):
    path = resolve_stock_data_prefix(path)
    path = os.path.normpath(path)
    path = os.path.abspath(path)
    path = os.path.realpath(path)
    path = os.path.normcase(path)
    return path        



