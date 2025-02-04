import json
from .coco import *

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