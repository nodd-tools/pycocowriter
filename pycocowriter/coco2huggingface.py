import os
import json
import argparse
import urllib.request
import urllib.error
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

# Import the existing discovery logic from your library
from .coco2yolo import discover_coco_files


def build_manifest(args: argparse.Namespace) -> dict:
    """
    Normalizes the CLI arguments into a standard manifest mapping.
    
    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments containing auto-discovery paths 
        or explicit file pairs.

    Returns
    -------
    dict
        A dictionary mapping split names to lists of tuples containing the 
        resolved JSON path and target image directory:
        { 'train': [('/path/to/train.json', '/path/to/images')], ... }

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     coco_dir=None, download_dir=None,
    ...     train_pair=[('train.json', 'images/train')],
    ...     val_pair=None, test_pair=None
    ... )
    >>> manifest = build_manifest(args)
    >>> len(manifest['train'])
    1
    """
    manifest = {'train': [], 'val': [], 'test': []}
    
    # Path A: Convention / Auto-Discovery
    if args.coco_dir:
        splits = discover_coco_files(args.coco_dir)
        for split_name, json_paths in splits.items():
            for json_path in json_paths:
                # Use the same basename logic as coco2yolo
                basename = os.path.splitext(os.path.basename(json_path))[0]
                expected_img_dir = Path(args.download_dir).resolve() / basename / "images"
                manifest[split_name].append((str(Path(json_path).resolve()), str(expected_img_dir)))
                
    # Path B: Explicit Override Pairs (Escape Hatch)
    else:
        if args.train_pair:
            manifest['train'].extend([(str(Path(j).resolve()), str(Path(d).resolve())) for j, d in args.train_pair])
        if args.val_pair:
            manifest['val'].extend([(str(Path(j).resolve()), str(Path(d).resolve())) for j, d in args.val_pair])
        if args.test_pair:
            manifest['test'].extend([(str(Path(j).resolve()), str(Path(d).resolve())) for j, d in args.test_pair])
            
    return manifest


def sync_images(manifest: dict) -> None:
    """
    Parses the JSONs and ensures all images exist locally. 
    
    Downloads them via coco_url if they are missing. Fails fast on errors.

    Parameters
    ----------
    manifest : dict
        A dictionary mapping splits to lists of (json_path, image_dir) tuples.

    Raises
    ------
    ValueError
        If a local image is missing and no `coco_url` was found in the JSON.
    RuntimeError
        If an image download fails.
    """
    print("\n--- Synchronizing Images ---")
    for split, pairs in manifest.items():
        for json_path, img_dir in pairs:
            os.makedirs(img_dir, exist_ok=True)
            
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            images_to_check = coco_data.get('images', [])
            print(f"[{split}] Checking {len(images_to_check)} images for {os.path.basename(json_path)}...")
            
            for img in images_to_check:
                file_name = img['file_name']
                coco_url = img.get('coco_url')
                local_path = Path(img_dir) / file_name
                
                if not local_path.exists():
                    if not coco_url:
                        raise ValueError(f"Failing Fast: Local image {file_name} is missing and no coco_url was found in {json_path}.")
                    
                    # Ensure subdirectories inside the images folder exist if file_name is nested
                    os.makedirs(local_path.parent, exist_ok=True)
                    
                    try:
                        print(f"  Downloading missing image: {file_name}")
                        urllib.request.urlretrieve(coco_url, str(local_path))
                    except urllib.error.URLError as e:
                        raise RuntimeError(f"Failing Fast: Failed to download {coco_url}. Error: {e}")


def build_staging_area(manifest: dict, staging_root: Path) -> None:
    """
    Populates a temporary directory with symlinks and generates HF metadata.
    
    Creates file-level symlinks to the images, symlinks to the original 
    JSON files, and generates the Hugging Face metadata.jsonl.

    Parameters
    ----------
    manifest : dict
        A dictionary mapping splits to lists of (json_path, image_dir) tuples.
    staging_root : Path
        The root directory of the temporary staging area.
    """
    print("\n--- Building Temporary Staging Area ---")
    
    for split, pairs in manifest.items():
        if not pairs:
            continue
            
        split_dir = staging_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / "metadata.jsonl"
        
        with open(metadata_path, 'w') as meta_f:
            for json_path, img_dir in pairs:
                basename = os.path.splitext(os.path.basename(json_path))[0]
                original_json_name = os.path.basename(json_path)
                
                # 1. Symlink the original COCO JSON for users who want raw downloads
                os.symlink(json_path, split_dir / original_json_name)
                
                # Namespaced image directory to prevent collisions between multiple JSONs
                staging_img_dir = split_dir / "images" / basename
                staging_img_dir.mkdir(parents=True, exist_ok=True)
                
                with open(json_path, 'r') as f:
                    coco_data = json.load(f)
                    
                # Group annotations by image_id for O(1) lookup
                anns_by_img = {}
                for ann in coco_data.get('annotations', []):
                    anns_by_img.setdefault(ann['image_id'], []).append(ann)
                    
                # 2. Symlink the images and build the Rosetta Stone
                for img in coco_data.get('images', []):
                    img_id = img['id']
                    file_name = img['file_name']
                    
                    src_path = Path(img_dir) / file_name
                    dst_path = staging_img_dir / file_name
                    
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(src_path, dst_path)
                    
                    # Hugging Face JSONL record
                    anns = anns_by_img.get(img_id, [])
                    hf_row = {
                        "file_name": f"images/{basename}/{file_name}",
                        "objects": {
                            "bbox": [a["bbox"] for a in anns],
                            "category_id": [a["category_id"] for a in anns]
                        }
                    }
                    meta_f.write(json.dumps(hf_row) + "\n")
                    
        print(f"[{split}] Staging complete. Indexed {len(pairs)} COCO files.")


def publish_to_huggingface(staging_dir: Path, repo_id: str, token: str, is_private: bool) -> None:
    """
    Authenticates and uploads the symlink structure to the Hugging Face Hub.

    Parameters
    ----------
    staging_dir : Path
        The path to the populated temporary staging directory.
    repo_id : str
        The Hugging Face Repository ID (e.g., 'org-name/dataset-name').
    token : str
        The Hugging Face write token.
    is_private : bool
        Whether the dataset should be private on the Hub.
    """
    print(f"\n--- Publishing to Hugging Face Hub: {repo_id} ---")
    api = HfApi(token=token)
    
    # Ensure the repository exists
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=is_private, exist_ok=True)
    
    # Hugging Face resolves file-level symlinks automatically and uploads the binary data
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("\nSuccess! Dataset uploaded.")


def stage_and_publish(manifest: dict, repo_id: str, token: str, is_private: bool) -> None:
    """
    Orchestrates the creation of a temporary staging directory, populating it
    with symlinks and metadata, and publishing it to Hugging Face.

    Parameters
    ----------
    manifest : dict
        A dictionary mapping splits to lists of (json_path, image_dir) tuples.
    repo_id : str
        The Hugging Face Repository ID (e.g., 'org-name/dataset-name').
    token : str
        The Hugging Face write token.
    is_private : bool
        Whether the dataset should be private on the Hub.
    """
    with tempfile.TemporaryDirectory() as staging_root:
        staging_path = Path(staging_root)
        
        build_staging_area(manifest, staging_path)
        publish_to_huggingface(staging_path, repo_id, token, is_private)
        
    print("\nTemporary staging directory cleaned up successfully.")


def main():
    """
    Main entrypoint for the pycocowriter.coco2huggingface module.
    """
    parser = argparse.ArgumentParser(description="Publish COCO datasets to Hugging Face with intelligent caching/symlinking.")
    
    # Hub Options
    parser.add_argument("--repo_id", required=True, help="Hugging Face Repository ID (e.g., 'org-name/dataset-name')")
    parser.add_argument("--token", help="Hugging Face write token (can also be set via HUGGING_FACE_HUB_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Set this flag to make the dataset private on the Hub")
    
    # Path A: Auto-Discovery
    parser.add_argument("--coco_dir", help="Directory containing COCO JSON files (triggers auto-discovery)")
    parser.add_argument("--download_dir", help="Root directory where images will be pooled/checked")
    
    # Path B: Explicit Mapping
    parser.add_argument("--train_pair", nargs=2, action="append", metavar=('JSON_PATH', 'IMG_DIR'), help="Explicit train pair")
    parser.add_argument("--val_pair", nargs=2, action="append", metavar=('JSON_PATH', 'IMG_DIR'), help="Explicit val pair")
    parser.add_argument("--test_pair", nargs=2, action="append", metavar=('JSON_PATH', 'IMG_DIR'), help="Explicit test pair")

    args = parser.parse_args()

    # XOR Validation
    has_discovery = bool(args.coco_dir and args.download_dir)
    has_explicit = bool(args.train_pair or args.val_pair or args.test_pair)
    
    if has_discovery and has_explicit:
        parser.error("You must use either auto-discovery (--coco_dir & --download_dir) OR explicit pairs (--train_pair, etc.), not both.")
    elif not has_discovery and not has_explicit:
        parser.error("You must provide inputs using auto-discovery (--coco_dir & --download_dir) or explicit pairs (--train_pair, etc.).")
    elif args.coco_dir and not args.download_dir:
        parser.error("--download_dir is required when using --coco_dir.")

    # 1. Normalize Inputs
    manifest = build_manifest(args)
    
    # 2. Check & Fetch Images
    sync_images(manifest)
    
    # 3. Create Temporary Staging and Publish
    stage_and_publish(manifest, args.repo_id, args.token, args.private)

if __name__ == "__main__":
    main()
