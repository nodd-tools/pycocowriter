import os
import json
import shutil
import random
import argparse
from pathlib import Path

from .coco2yolo import discover_coco_files

def coco_subset_percent(source_dir: str, dest_dir: str, percentage: float, seed: int = 42) -> None:
    """
    Create a percentage-based subset of a COCO dataset.

    Utilizes `discover_coco_files` to locate train, val, and test JSON files 
    in the source directory. It samples a percentage of images from each 
    discovered split, filters the corresponding annotations, and copies the 
    referenced image files to a mirrored directory structure in the destination.

    Images are expected to be located in `{source_dir}/{json_basename}/images/` 
    matching the structural expectations of the wider pycocowriter library.

    Parameters
    ----------
    source_dir : str
        The directory containing the source COCO annotation JSON files and 
        their corresponding image directories.
    dest_dir : str
        The destination directory where the subset dataset will be saved.
    percentage : float
        Percentage of the dataset to keep (must be between 0.0 and 100.0).
    seed : int, optional
        Random seed for reproducibility when sampling images, by default 42.

    Returns
    -------
    None

    Examples
    --------
    >>> # Extract 10.5% of the dataset into a new directory
    >>> coco_subset_percent('./data/raw_coco', './data/subset_coco', 10.5)
    """
    if not (0 < percentage <= 100):
        raise ValueError(f"Percentage must be strictly greater than 0 and less than or equal to 100. Got {percentage}")

    random.seed(seed)
    os.makedirs(dest_dir, exist_ok=True)
    
    split_files = discover_coco_files(source_dir)
    
    processed_any = False

    for stage, json_paths in split_files.items():
        for json_path in json_paths:
            basename = os.path.splitext(os.path.basename(json_path))[0]
            print(f"Processing '{stage}' split from file: {json_path}")
            
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
                
            all_images = coco_data.get('images', [])
            if not all_images:
                print(f"  -> No images found in {basename}. Skipping.")
                continue
                
            num_total = len(all_images)
            num_keep = max(1, int(num_total * (percentage / 100.0)))
            print(f"  -> Total images: {num_total}. Subsetting to {num_keep} images ({percentage}%).")
            
            sampled_images = random.sample(all_images, num_keep)
            sampled_image_ids = {img['id'] for img in sampled_images}
            
            all_annotations = coco_data.get('annotations', [])
            sampled_annotations = [
                ann for ann in all_annotations
                if ann.get('image_id') in sampled_image_ids
            ]
            
            subset_data = {
                "info": coco_data.get("info", {}),
                "licenses": coco_data.get("licenses", []),
                "images": sampled_images,
                "annotations": sampled_annotations,
                "categories": coco_data.get("categories", [])
            }
            
            # Save new JSON
            dest_json_path = os.path.join(dest_dir, os.path.basename(json_path))
            with open(dest_json_path, 'w') as f:
                json.dump(subset_data, f)
                
            source_img_dir = os.path.join(source_dir, basename, 'images')
            dest_img_dir = os.path.join(dest_dir, basename, 'images')
            
            os.makedirs(dest_img_dir, exist_ok=True)
            
            copied_count = 0
            missing_count = 0
            
            for img_info in sampled_images:
                img_filename = img_info.get('file_name')
                if not img_filename:
                    continue
                    
                src_img_path = os.path.join(source_img_dir, img_filename)
                dst_img_path = os.path.join(dest_img_dir, img_filename)
                
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    copied_count += 1
                else:
                    print(f"  -> Warning: Source image not found: {src_img_path}")
                    missing_count += 1
                    
            print(f"  -> Finished '{basename}'. Copied {copied_count} images. Missing: {missing_count}.\n")
            processed_any = True

    if not processed_any:
        print(f"Warning: No valid dataset splits were found or processed in {source_dir}.")
    else:
        print(f"Subset creation complete. Data saved to: {Path(dest_dir).absolute()}")

def _cli():
    """Parses command-line arguments and executes the subsetter."""
    parser = argparse.ArgumentParser(description="Create a percentage-based subset of a COCO dataset.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True,
        help="Path to the source dataset directory containing COCO JSONs and images."
    )
    parser.add_argument(
        "--dest_dir", 
        type=str, 
        required=True,
        help="Path where the subset dataset will be saved."
    )
    parser.add_argument(
        "--percentage", 
        type=float, 
        required=True,
        help="Percentage of the dataset to keep (e.g., 10 for 10%, 50.5 for 50.5%)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist or is not a directory.")
        return

    print("Starting subset process...")
    print(f"Source: {args.source_dir}")
    print(f"Destination: {args.dest_dir}")
    print(f"Percentage: {args.percentage}%")
    
    coco_subset_percent(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        percentage=args.percentage,
        seed=args.seed
    )

if __name__ == "__main__":
    _cli()
