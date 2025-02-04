import json
import unittest

from pycocowriter.cocomerge import coco_merge
import utils

class TestCocoMerge(unittest.TestCase):

    example_coco_1 = {
        'info': {},
        'images': [
            {
                'id': 1, 
                'file_name': 'tests/static/example1.png', 
                'width': 1, 
                'height': 1
            },
            {
                'id': 2, 
                'file_name': 'tests/static/example2.png', 
                'width': 1, 
                'height': 1,
                'license_id': 1
            }
        ],
        'annotations': [
            {
                'image_id': 1,
                'id': 0,
                'category_id': 1,
                'bbox': (10, 11, 10, 10),
                'area': 100,
                'iscrowd': 0,
                'keypoints': [1, 2, 2, 3, 4, 2],
                'num_keypoints': 2
            },
            {
                'image_id': 2,
                'id': 1,
                'category_id': 1,
                'bbox': (100, 110, 100, 100),
                'area': 10000,
                'iscrowd': 0,
                'keypoints': [10, 20, 2, 70, 80, 2],
                'num_keypoints': 2},
            {
                'image_id': 1,
                'id': 2,
                'category_id': 2,
                'bbox': (1000, 1100, 1000, 1000),
                'area': 1000000,
                'iscrowd': 0,
                'keypoints': [100, 200, 2, 300, 400, 2],
                'num_keypoints': 2
            }
        ],
        'licenses': [
            {
                'name': 'license1',
                'url': 'example.com',
                'id': 1
            }
        ],
        'categories': [
            {
                'name': 'label1',
                'id': 1,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            },
            {
                'name': 'label2',
                'id': 2,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            }
        ]
    }

    example_coco_2 = {
        'info': {},
        'images': [
            {
                'id': 1, 
                'file_name': 'tests/static/example3.png', 
                'width': 1, 
                'height': 1
            },
            {
                'id': 2, 
                'file_name': 'tests/static/example1.png', 
                'width': 1, 
                'height': 1,
                'license_id': 1
            }
        ],
        'annotations': [
            {
                'image_id': 1,
                'id': 0,
                'category_id': 1,
                'bbox': (11, 12, 11, 11),
                'area': 100,
                'iscrowd': 0,
                'keypoints': [1, 2, 2, 3, 4, 2],
                'num_keypoints': 2
            },
            {
                'image_id': 2,
                'id': 1,
                'category_id': 1,
                'bbox': (101, 111, 101, 101),
                'area': 10000,
                'iscrowd': 0,
                'keypoints': [10, 20, 2, 70, 80, 2],
                'num_keypoints': 2},
            {
                'image_id': 1,
                'id': 2,
                'category_id': 2,
                'bbox': (1001, 1101, 1001, 1001),
                'area': 1000000,
                'iscrowd': 0,
                'keypoints': [100, 200, 2, 300, 400, 2],
                'num_keypoints': 2
            }
        ],
        'licenses': [
            {
                'name': 'license2',
                'url': 'example.com',
                'id': 1
            }
        ],
        'categories': [
            {
                'name': 'label4',
                'id': 1,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            },
            {
                'name': 'label1',
                'id': 2,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            }
        ]
    }

    examples_merged = {
        "annotations": [
            {
                "image_id": 1,
                "id": 1,
                "category_id": 1,
                "bbox": (
                    10,
                    11,
                    10,
                    10
                ),
                "area": 100,
                "iscrowd": 0,
                "keypoints": [
                    1,
                    2,
                    2,
                    3,
                    4,
                    2
                ],
                "num_keypoints": 2
            },
            {
                "image_id": 2,
                "id": 2,
                "category_id": 1,
                "bbox": (
                    100,
                    110,
                    100,
                    100
                ),
                "area": 10000,
                "iscrowd": 0,
                "keypoints": [
                    10,
                    20,
                    2,
                    70,
                    80,
                    2
                ],
                "num_keypoints": 2
            },
            {
                "image_id": 1,
                "id": 3,
                "category_id": 2,
                "bbox": (
                    1000,
                    1100,
                    1000,
                    1000
                ),
                "area": 1000000,
                "iscrowd": 0,
                "keypoints": [
                    100,
                    200,
                    2,
                    300,
                    400,
                    2
                ],
                "num_keypoints": 2
            },
            {
                "image_id": 3,
                "id": 4,
                "category_id": 3,
                "bbox": (
                    11,
                    12,
                    11,
                    11
                ),
                "area": 100,
                "iscrowd": 0,
                "keypoints": [
                    1,
                    2,
                    2,
                    3,
                    4,
                    2
                ],
                "num_keypoints": 2
            },
            {
                "image_id": 1,
                "id": 5,
                "category_id": 3,
                "bbox": (
                    101,
                    111,
                    101,
                    101
                ),
                "area": 10000,
                "iscrowd": 0,
                "keypoints": [
                    10,
                    20,
                    2,
                    70,
                    80,
                    2
                ],
                "num_keypoints": 2
            },
            {
                "image_id": 3,
                "id": 6,
                "category_id": 1,
                "bbox": (
                    1001,
                    1101,
                    1001,
                    1001
                ),
                "area": 1000000,
                "iscrowd": 0,
                "keypoints": [
                    100,
                    200,
                    2,
                    300,
                    400,
                    2
                ],
                "num_keypoints": 2
            }
        ],
        "images": [
            {
                "id": 1,
                "file_name": "tests/static/example1.png",
                "width": 1,
                "height": 1
            },
            {
                "id": 2,
                "file_name": "tests/static/example2.png",
                "width": 1,
                "height": 1,
                "license_id": 1
            },
            {
                "id": 3,
                "file_name": "tests/static/example3.png",
                "width": 1,
                "height": 1
            }
        ],
        "info": {},
        "categories": [
            {
                "name": "label1",
                "id": 1,
                "keypoints": [
                    "head",
                    "tail"
                ],
                "skeleton": [
                    1,
                    2
                ]
            },
            {
                "name": "label2",
                "id": 2,
                "keypoints": [
                    "head",
                    "tail"
                ],
                "skeleton": [
                    1,
                    2
                ]
            },
            {
                "name": "label4",
                "id": 3,
                "keypoints": [
                    "head",
                    "tail"
                ],
                "skeleton": [
                    1,
                    2
                ]
            }
        ],
        "licenses": [
            {
                "name": "license1",
                "url": "example.com",
                "id": 1
            },
            {
                "name": "license2",
                "url": "example.com",
                "id": 2
            }
        ]
    }

    def test_coco_merge(self):
        merged = coco_merge(TestCocoMerge.example_coco_1, TestCocoMerge.example_coco_2)
        self.assertDictEqual(merged, TestCocoMerge.examples_merged)