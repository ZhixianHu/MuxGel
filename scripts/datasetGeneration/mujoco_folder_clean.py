'''
Clean up generated local visual-tactile patches by removing directories with insufficient data
'''

from pathlib import Path

def count_files_recursive(dir_path: Path) -> int:
    return sum(1 for p in dir_path.rglob("*") if p.is_file())

def prune_and_report(root_dir: str, target_count: int = 850) -> None:
    """
    Cleans up the visual-tactile patch dataset by removing incomplete directories.
    
    Args:
        root_dir: Path to the root dataset.
        target_count: Exact number of files expected in a valid patch folder.
        dry_run: If True, only prints what would be deleted without actual removal.
    """
    root = Path(root_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root directory: {root}")

    all_dirs = sorted([p for p in root.rglob("*") if p.is_dir()],
                      key=lambda p: len(p.parts),
                      reverse=True)

    for d in all_dirs:
        is_empty = not any(d.iterdir())
        if is_empty:
            print(f"[EMPTY -> DELETE] {d}")
            try:
                d.rmdir()  
            except OSError as e:
                print(f"  [WARN] Failed to delete {d}: {e}")
            continue

        n_files = count_files_recursive(d)
        if n_files != target_count:
            print(f"[COUNT != {target_count}] {d}  files={n_files}")

if __name__ == "__main__":
    prune_and_report("data/mujoco_patch_output", target_count=850)
