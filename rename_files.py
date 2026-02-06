from pathlib import Path

def rename_files_first_10_chars_uppercase(folder_path):
    """
    Rename all files in a folder so that the first 10 characters are uppercase.
    
    Args:
        folder_path (str): Path to the folder containing files to rename
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    if not folder.is_dir():
        print(f"Path is not a directory: {folder_path}")
        return
    
    files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in {folder_path}")
        return
    
    print(f"Found {len(files)} file(s) to process\n")
    
    for file in files:
        original_name = file.name
        # Split filename and extension
        stem = file.stem  # filename without extension
        extension = file.suffix  # file extension (including the dot)
        
        # Get first 10 characters of stem and make them uppercase
        if len(stem) <= 10:
            new_stem = stem.upper()
        else:
            # First 10 chars uppercase, rest stays the same
            new_stem = stem[:10].upper()
        
        # Combine new stem with original extension
        new_name = new_stem + extension
        
        if original_name != new_name:
            new_path = file.parent / new_name
            try:
                file.rename(new_path)
                print(f"✓ Renamed: {original_name} -> {new_name}")
            except Exception as e:
                print(f"✗ Error renaming {original_name}: {e}")
        else:
            print(f"- No change needed: {original_name}")

if __name__ == "__main__":
    folder_path = "./CE-FL STL"
    rename_files_first_10_chars_uppercase(folder_path)
