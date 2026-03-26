from pycocotools.coco import COCO
import os
import yaml
from pathlib import Path
import shutil

''' 
Constants that are expected/used by yolo or the ultralytics
conversion tools
''' 
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'
ULTRALYTICS_COCO_CONVERSION_DIR = 'coco_converted'

def discover_coco_files(coco_file_dir: str) -> dict[str, list[str]]:
    """
    Scans a directory for COCO JSON files and categorizes them into splits.

    Uses a simple naming convention: files containing 'val' are validation,
    files containing 'test' are testing, and all other JSON files are 
    considered training data.

    Parameters
    ----------
    coco_file_dir : str
        The directory containing the COCO annotation files.

    Returns
    -------
    dict[str, list[str]]
        A dictionary with keys 'train', 'val', and 'test', where each value
        is a list of absolute paths to the discovered JSON files.

    Examples
    --------
    >>> files = discover_coco_files('./data')
    >>> print(files['train'])
    ['/path/to/data/train_annotations.json']
    """
    results = {'train': [], 'val': [], 'test': []}
    
    # Ensure directory exists before globbing
    if not os.path.isdir(coco_file_dir):
        return results

    for coco_file in sorted(Path(coco_file_dir).resolve().glob("*.json")):
        basename = coco_file.name.lower()
        
        if 'val' in basename:
            results['val'].append(str(coco_file))
        elif 'test' in basename:
            results['test'].append(str(coco_file))
        else:
            results['train'].append(str(coco_file))
            
    return results


def coco2yoloyaml(coco_file_dir: str, destination: str) -> None:
    '''
    Construct a YOLO-format yaml metadata file from a 
    collection of COCO-format annotation json files.

    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ../datasets/coco128 # dataset root dir
    train: images/train2017 # train images (relative to 'path') 128 images
    val: images/train2017 # val images (relative to 'path') 128 images
    test: # test images (optional)

    # Classes (80 COCO classes)
    names:
        0: person
        1: bicycle
        2: car
        # ...
        77: teddy bear
        78: hair drier
        79: toothbrush

    Parameters
    ----------
    coco_file_dir: str
        The location of the COCO json files.
    destination: str
        Where to write the resulting yaml file
    '''
    # Use the shared discovery logic
    split_files = discover_coco_files(coco_file_dir)
    
    result = {
        'path': destination,
        'names': {}
    }
    
    os.makedirs(destination, exist_ok=True)
    
    for stage in split_files:
        for json_path in split_files[stage]:
            coco = COCO(json_path)
            basename = os.path.splitext(os.path.basename(json_path))[0]
            
            # Populate the YAML 'names' dictionary
            if not result['names']:
                for cat_id in coco.cats:
                    # YOLO classes are 0-indexed; COCO are typically 1-indexed
                    result['names'][cat_id - 1] = coco.cats[cat_id]['name']
            else:
                for cat_id in coco.cats:
                    # All coco files must have matching categories
                    assert result['names'][cat_id - 1] == coco.cats[cat_id]['name']
                
            # Determine the relative image directory path for this split
            # We assume YOLO images will be organized in folders named after the JSON basename
            img_rel_path = os.path.join(basename, IMAGE_DIR)
            if stage not in result:
                result[stage] = []
            result[stage].append(img_rel_path)

    # Mandatory fallback: YOLO training requires a 'val' path
    if 'train' in result and 'val' not in result:
        result['val'] = result['train']

    # Write the YAML file
    yaml_path = os.path.join(destination, 'train.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(result, f, sort_keys=False)
        
def download_coco_images(coco_file_dir: str, destination: str) -> None:
    '''
    download the images referenced in the "coco_url" of the coco files
    in a directory.

    Parameters
    ----------
    coco_file_dir: str
        The location of the COCO json files.
    destination: str
        The root directory to which the images should be writ.
    '''
    os.makedirs(destination, exist_ok=True)
    for coco_file in sorted(Path(coco_file_dir).resolve().glob("*.json")):
        coco = COCO(coco_file)
        basename = os.path.splitext(os.path.basename(coco_file))[0]
        img_dir = os.path.join(destination, basename, IMAGE_DIR)
        coco.download(img_dir)

def rename_label_paths(coco_file_dir: str, destination: str) -> None:
    '''
    depending on the `destination` to which we download the images,
    our labels need to be updated to point to this location.

    Parameters
    ----------
    coco_file_dir: str
        The location of the COCO json files.
    destination: str
        The root directory to which the images should be writ.
    '''
    os.makedirs(destination, exist_ok=True)
    for coco_file in sorted(Path(coco_file_dir).resolve().glob("*.json")):
        print(f'transferring {coco_file}')
        basename = os.path.splitext(os.path.basename(coco_file))[0]
        # where the ultralytics script stores the labels
        label_path = os.path.join(
            ULTRALYTICS_COCO_CONVERSION_DIR, LABEL_DIR, basename)
        # where they had ought to be given our image destination
        dest_label_path = os.path.join(
            destination, basename, LABEL_DIR)
        
        # Gracefully degrade if no labels were generated (e.g. test set)
        if os.path.exists(label_path):
            # Ensure the destination's parent directory exists
            os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)
            try:
                shutil.rmtree(dest_label_path)
            except FileNotFoundError:
                pass    # Fail silently if the directory doesn't exist
            shutil.move(label_path, dest_label_path)
    
    # Clean up the conversion directory if it exists
    if os.path.exists(ULTRALYTICS_COCO_CONVERSION_DIR):
        shutil.rmtree(ULTRALYTICS_COCO_CONVERSION_DIR)

def coco2yolo(coco_file_dir: str, destination: str, 
              use_segments: bool = False, use_keypoints: bool = False):
    '''
    Convert a COCO-format annotation dataset and metadata 
    into YOLO-format annotations and metadata.  Downloads the
    `coco_url` files into an appropriate location.

    Defaults to bounding-box style annotations.  If the dataset
    uses segmentation or keypoints, these details are not automatically
    inferred, and this needs to be toggled.  At most one of
    `use_segments` or `use_keypoints` should be True

    Parameters
    ----------
    coco_file_dir: str
        The location of the COCO json files.
    destination: str
        The root directory to which the images should be writ.
    use_segments: bool
        Whether to attempt to read segmentation data from the COCO
        files, and create a YOLO segmentation dataset
    use_keypoints: bool        
        Whether to attempt to read keypoint data from the COCO
        files, and create a YOLO keypoint dataset
    '''
    from ultralytics.data.converter import convert_coco
    convert_coco(
        coco_file_dir, 
        use_segments=use_segments, 
        use_keypoints=use_keypoints, 
        cls91to80=False
    )
    rename_label_paths(coco_file_dir, destination)
    coco2yoloyaml(coco_file_dir, destination)
    download_coco_images(coco_file_dir, destination)

#if __name__ == '__main__':
#    coco2yolo('../data/raw/oceaneyes_example', '../data/raw/oceaneyes_example/yolo_example_data', use_segments=False, use_keypoints=False)
