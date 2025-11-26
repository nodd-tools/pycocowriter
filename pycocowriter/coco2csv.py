import pycocotools.coco
import json
from collections.abc import Iterable
import csv

COCO2CSV_ANNOTATION_HEADER = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd', 'keypoints', 'num_keypoints', 'caption', 'file_name', 'segments_info']
COCO2CSV_IMAGE_HEADER = ['width', 'height', 'file_name', 'flickr_url', 'coco_url', 'date_captured']
COCO2CSV_CATEGORY_HEADER = ['name', 'supercategory', 'keypoints', 'skeleton', 'isthing', 'color']

def csv_safe(val):
    """
    Convert a value to a CSV-safe representation.

    Lists and dictionaries are converted to JSON strings. 
    All other values are returned unchanged.

    Parameters
    ----------
    val : Any
        The value to be converted. Can be of any type.

    Returns
    -------
    Any
        If `val` is a list or dictionary, returns a JSON-encoded string.
        Otherwise, returns the original value.

    Examples
    --------
    >>> csv_safe([1, 2, 3])
    '[1, 2, 3]'
    >>> csv_safe({'x': 1})
    '{"x": 1}'
    >>> csv_safe(42)
    42
    """
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return val

def read_header(ann: dict, img: dict, cat: dict) -> list:
    """
    Determine which columns exist in annotation, image, and category dictionaries.

    Filters the predefined COCO CSV headers to those keys actually present 
    in the provided dictionaries.

    Parameters
    ----------
    ann : dict
        A COCO annotation dictionary.
    img : dict
        A COCO image dictionary corresponding to `ann['image_id']`.
    cat : dict
        A COCO category dictionary corresponding to `ann['category_id']`.

    Returns
    -------
    tuple of list
        Three lists of column names:
        - Annotation columns present in `ann`
        - Image columns present in `img`
        - Category columns present in `cat`

    Examples
    --------
    >>> ann = {'id': 1, 'image_id': 5, 'bbox': [0,0,10,10]}
    >>> img = {'width': 640, 'height': 480, 'file_name': 'img.jpg'}
    >>> cat = {'name': 'person', 'supercategory': 'human'}
    >>> read_header(ann, img, cat)
    (['id', 'image_id', 'bbox'], ['width', 'height', 'file_name'], ['name', 'supercategory'])
    """
    return (
        [col for col in COCO2CSV_ANNOTATION_HEADER if col in ann],
        [col for col in COCO2CSV_IMAGE_HEADER      if col in img],
        [col for col in COCO2CSV_CATEGORY_HEADER   if col in cat]
    )

def flatten_coco(coco: pycocotools.coco.COCO) -> Iterable[list]:
    """
    Generate CSV rows from a COCO-format JSON annotation file.

    Each row combines fields from the annotation, the associated image, 
    and the associated category. Lists and dictionaries are JSON-encoded 
    for CSV compatibility. The first row yielded is a header row.

    Parameters
    ----------
    filename : pycocotools.coco.COCO
        The loaded COCO annotations.

    Yields
    ------
    list
        A row of CSV data as a list of values. The first row is the header.

    Examples
    --------
    >>> coco = pycocotools.coco.COCO('tests/example_coco.json')
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    >>> flat_coco = flatten_coco(coco)
    >>> import itertools
    >>> for row in itertools.islice(flat_coco, 4):
    ...     print(row)
    ['id', 'image_id', 'category_id', 'area', 'bbox', 'iscrowd', 'width', 'height', 'file_name', 'name']
    [1, 1, 60, 102225.5616, '[117.84031999999999, 235.14, 425.93984, 240.0]', 0, 640, 480, '000000000612.jpg', 'bed']
    [2, 1, 78, 6234.393046425599, '[320.4, 271.38984000000005, 71.07968, 87.70992]', 0, 640, 480, '000000000612.jpg', 'teddy bear']
    [3, 1, 78, 5028.19008, '[260.31968, 281.72016, 62.0, 81.09984]', 0, 640, 480, '000000000612.jpg', 'teddy bear']
    """
    headers = None
    for ann in coco.anns.values():
        img = coco.imgs[ann['image_id']]
        cat = coco.cats[ann['category_id']]
        if headers is None:
            headers = read_header(ann, img, cat)
            yield [*headers[0], *headers[1], *headers[2]]
        yield [
            csv_safe(obj[col]) 
            for obj, head in zip([ann, img, cat], headers) 
            for col in head
        ]

def coco2csv(coco_path: str, csv_path: str, **csv_writer_kwargs) -> None:
    """
    Convenience wrapper to write a coco json file directly to a csv file.
    Recommend that users use `flatten_coco` instead, and then manually adjust 
    the csv fields to align with their expected format before writing.

    Parameters
    ----------
    coco_path: str
    The path to the coco json file
  
    csv_path: str
    The path where the result csv should be written

    Returns
    -------
    returns None, but writes a file to disk at csv_path
    """
    coco = pycocotools.coco.COCO(coco_path)
    flat_coco = flatten_coco(coco)
    with open(csv_path, 'w') as f:
        writer = csv.writer(csv_path, **csv_writer_kwargs)
        writer.writerows(flat_coco)
