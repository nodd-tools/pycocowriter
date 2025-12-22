import json
from .coco import *
import copy

def coco_merge(*coco_dicts: dict, info: COCOInfo = COCOInfo()) -> dict:
    """
    Merge COCO annotations.

    adapted from pyodi https://github.com/Gradiant/pyodi under MPL 2.0

    Parameters
    ----------
    *coco_files: dicts
        dicts from coco files as e.g. read by json.load
    
    Returns
    -------
    coco_data: dict
        coco data for merged output as dict
    """

    categories = []
    images = []
    annotations = []
    licenses = []

    license_map = {}
    category_map = {}
    image_map = {}

    for data in coco_dicts:
        
        cat_id_map = {}
        for new_cat in data["categories"]:
            label = new_cat["name"]
            if label in category_map:
                cat_id_map[new_cat["id"]] = category_map[label]
            else:
                new_id = len(categories) + 1
                cat_id_map[new_cat["id"]] = new_id
                new_cat["id"] = new_id
                categories.append(new_cat)
                category_map[label] = new_id

        license_id_map = {}
        for new_license in data["licenses"]:
            license_name = new_license["name"]
            if license_name in license_map:
                license_id_map[new_license["id"]] = license_map[license_name]
            else:
                new_id = len(licenses) + 1
                license_id_map[new_license["id"]] = new_id
                new_license["id"] = new_id
                licenses.append(new_license)
                license_map[license_name] = new_id

        image_id_map = {}
        for new_image in data["images"]:
            file_name = new_image["file_name"]
            if file_name in image_map:
                image_id_map[new_image["id"]] = image_map[file_name]
            else:
                new_id = len(images) + 1
                image_id_map[new_image["id"]] = new_id
                new_image["id"] = new_id
                if "license_id" in new_image:
                    new_image["license_id"] = license_id_map[new_image["license_id"]]
                images.append(new_image)
                image_map[file_name] = new_id

        for new_annotation in data["annotations"]:
            new_id = len(annotations) + 1
            new_annotation["id"] = new_id
            new_annotation["category_id"] = cat_id_map[new_annotation["category_id"]]
            new_annotation["image_id"] = image_id_map[new_annotation["image_id"]]
            annotations.append(new_annotation)

    return {
        "annotations": annotations,
        "images": images,
        "info": info.to_dict(),
        "categories": categories,
        "licenses": licenses
    }

def coco_remap_categories(coco_dict: dict, name_mapping: dict) -> dict:
    """
    Rename categories based on a mapping. Does NOT merge or re-index.
    Resulting dataset may contain multiple categories with the same name.
    
    Parameters
    ----------
    coco_dict: dict
        Source dataset
    name_mapping: dict
        Old Name -> New Name. Unlisted names are kept as-is.

    Returns
    -------
    dict
        The original dataset, with category names remapped

    Examples
    --------
    >>> data = {
    ...     'categories': [
    ...         {'id': 1, 'name': 'taxi'},
    ...         {'id': 2, 'name': 'bus'},
    ...         {'id': 3, 'name': 'dog'}
    ...     ]
    ... }
    >>> mapping = {'taxi': 'vehicle', 'bus': 'vehicle'}
    >>> _ = coco_remap_categories(data, mapping)
    >>> import pprint; pprint.pprint(data['categories'])
    [{'id': 1, 'name': 'vehicle'},
     {'id': 2, 'name': 'vehicle'},
     {'id': 3, 'name': 'dog'}]
    """
    for cat in coco_dict.get("categories", []):
        if cat["name"] in name_mapping:
            cat["name"] = name_mapping[cat["name"]]

    return coco_dict

def coco_collapse_categories(coco_dict: dict) -> dict:
    """
    Merge categories that share the same name into a single ID.
    Updates annotations to point to the unified ID.

    Parameters
    ----------
    coco_dict: dict
        Source dataset

    Returns
    -------
    dict
        The original dataset, with duplicate categories collapsed

    Examples
    --------
    >>> data = {
    ...     'categories': [
    ...         {'id': 10, 'name': 'vehicle'},
    ...         {'id': 20, 'name': 'vehicle'}
    ...     ],
    ...     'annotations': [
    ...         {'id': 1, 'category_id': 10},
    ...         {'id': 2, 'category_id': 20}
    ...     ]
    ... }
    >>> _ = coco_collapse_categories(data)
    >>> # Notice ID 20 remains because it was last in the list
    >>> data['categories']
    [{'id': 20, 'name': 'vehicle'}]
    >>> # Both annotations now point to 20
    >>> [ann['category_id'] for ann in data['annotations']]
    [20, 20]
    """

    name_to_target_id = {cat["name"]: cat["id"] for cat in coco_dict["categories"]}
    all_target_ids = set(name_to_target_id.values())
    id_remap = {cat["id"]: name_to_target_id[cat["name"]] for cat in coco_dict["categories"]}
    coco_dict["categories"] = [cat for cat in coco_dict["categories"] if cat["id"] in all_target_ids]
    for ann in coco_dict["annotations"]:
        ann["category_id"] = id_remap[ann["category_id"]]

    return coco_dict

def coco_reindex_categories(coco_dict: dict) -> dict:
    """
    Re-index category IDs to be contiguous integers starting from 1.
    
    Parameters
    ----------
    coco_dict: dict
        Source dataset

    Returns
    -------
    dict
        The original dataset, with categories reindexed

    Examples
    --------
    >>> data = {
    ...     'categories': [{'id': 20, 'name': 'vehicle'}, {'id': 30, 'name': 'pickle'}],
    ...     'annotations': [{'id': 1, 'category_id': 20}]
    ... }
    >>> _ = coco_reindex_categories(data)
    >>> data['categories']
    [{'id': 2, 'name': 'vehicle'}, {'id': 1, 'name': 'pickle'}]
    >>> data['annotations']
    [{'id': 1, 'category_id': 2}]
    """
    category_name_map = {cat['name']: cat for cat in coco_dict['categories']}
    id_remap = {}
    for i, (name, cat) in enumerate(sorted(category_name_map.items())):
        new_id = i+1
        id_remap[cat['id']] = new_id
        cat['id'] = new_id
    for ann in coco_dict["annotations"]:
        ann["category_id"] = id_remap[ann["category_id"]]

    return coco_dict
