import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2] 
SRC_DIR = PROJECT_ROOT / "data" / "mujoco_patch_output"
DST_DIR = PROJECT_ROOT / "data" / "mujoco_patch_output_320_240"


TARGET_SIZE = (320, 240)

def resize_and_save():
    if not SRC_DIR.exists():
        print(f"Error: Source directory {SRC_DIR} does not exist.")
        return

    obj_dirs = [d for d in SRC_DIR.iterdir() if d.is_dir()]
    
    print(f"Found {len(obj_dirs)} object directories.")
    print(f"Processing... Target Size: {TARGET_SIZE}")
    for obj_dir in tqdm(obj_dirs, desc="Processing Objects"):
        rel_path = obj_dir.relative_to(SRC_DIR)
        target_obj_dir = DST_DIR / rel_path
        target_obj_dir.mkdir(parents=True, exist_ok=True)

        for file_path in obj_dir.iterdir():
            target_file_path = target_obj_dir / file_path.name
            
            
            if file_path.suffix.lower() in ['.jpg', '.png']:
                img = cv2.imread(str(file_path))
                if img is not None:
                    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(target_file_path), img_resized)
            

            elif file_path.suffix.lower() == '.npz':
                try:
                    data = np.load(file_path) 
                    new_data = {}
                    for key in data.files:
                        arr = data[key]
                        if arr.ndim >= 2:
                            resized_arr = cv2.resize(arr, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            new_data[key] = resized_arr
                        else:
                            new_data[key] = arr 
                    
                    np.savez_compressed(target_file_path, **new_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            else:
                shutil.copy2(file_path, target_file_path)

    print(f"\nDone! Processed data saved to: {DST_DIR}")

if __name__ == "__main__":
    resize_and_save()
