import os
import pathspec

def find_files_using_gitignore(folder, use_gitignore=True, parent_gitignore=None):
    """Recursively find all files in a folder, filtering them using nested .gitignore files."""
    file_list = []
    
    if use_gitignore:
        gitignore = parent_gitignore
        gitignore_path = os.path.join(folder, '.gitignore')
        if os.path.isfile(gitignore_path):
            with open(gitignore_path, 'r') as file:
                gitignore = pathspec.PathSpec.from_lines('gitwildmatch', file)
                if parent_gitignore:
                    all_patterns = set(gitignore.patterns + parent_gitignore.patterns)
                    gitignore = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns)
                        
    for item in os.listdir(folder):
        full_path = os.path.join(folder, item)
        if os.path.isdir(full_path):
            file_list += find_files_recursively(full_path, use_gitignore, parent_gitignore=gitignore)
        elif os.path.isfile(full_path):
            rel_path = os.path.relpath(full_path, start=folder)
            if gitignore and gitignore.match_file(rel_path):
                continue
            file_list.append(full_path)

    return file_list
