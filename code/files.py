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

def clean_up_temporary_files(tempdir):
    if tempdir and os.file.exists(tempdir) and os.path.isdir(tempdir):
        info = f" \"{tempdir}\"" if args.verbose else ""
        log(f"Removing temporary folder{info}...")
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
                # Errors are expected sometimes if the OS has files open
                log_verbose(f"\tignoring error: {e}")
        if tempdir:
            log_error(f"couldn't remove temporary folder \"{tempdir}\"")
        else:
            log_verbose(f"\t...success ({time_since(time_before)})")

