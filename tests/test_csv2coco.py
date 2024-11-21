import unittest
from pycocowriter.coco import COCOData, COCOInfo
import utils
import json
import pycocowriter.csv2coco


class TestCSV2COCO(unittest.TestCase):

    example_config = {
        'meta': {
            'skiprows': 2
        },
        'filename': 3,
        'label': 10,
        'keypoints': [
            {
                'name': 'head',
                'x': 13,
                'y': 14
            },
            {
                'name': 'tail',
                'x': 15,
                'y': 16
            }
        ],
        'bbox_tlbr': {
            'tlx': 4,
            'tly': 5,
            'brx': 6,
            'bry': 7
        },
        'keypoint_skeleton': [1, 2]
    }

    example_csv = [
        ['err'] * 17,
        ['err'] * 17,
        ['err', 'err', 'err', 'filename1', 10, 11, 20, 21, 'err', 'err', 'label1', 'err', 'err', 1, 2, 3, 4, 'err'],
        ['err', 'err', 'err', 'filename2', 100, 110, 200, 210, 'err', 'err', 'label1', 'err', 'err', 10, 20, 70, 80, 'err'],
        ['err', 'err', 'err', 'filename1', 1000, 1100, 2000, 2100, 'err', 'err', 'label2', 'err', 'err', 100, 200, 300, 400, 'err'],
    ]

    example_csv_as_coco = {
        'info': {},
        'images': [
            {'id': 0, 'file_name': 'filename1'},
            {'id': 1, 'file_name': 'filename2'}
        ],
        'annotations': [
            {
                'image_id': 0,
                'id': 0,
                'category_id': 0,
                'bbox': (10, 11, 10, 10),
                'area': 100,
                'iscrowd': 0,
                'keypoints': [1, 2, 2, 3, 4, 2],
                'num_keypoints': 2
            },
            {
                'image_id': 1,
                'id': 1,
                'category_id': 0,
                'bbox': (100, 110, 100, 100),
                'area': 10000,
                'iscrowd': 0,
                'keypoints': [10, 20, 2, 70, 80, 2],
                'num_keypoints': 2},
            {
                'image_id': 0,
                'id': 2,
                'category_id': 1,
                'bbox': (1000, 1100, 1000, 1000),
                'area': 1000000,
                'iscrowd': 0,
                'keypoints': [100, 200, 2, 300, 400, 2],
                'num_keypoints': 2
            }
        ],
        'licenses': [],
        'categories': [
            {
                'name': 'label1',
                'id': 0,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            },
            {
                'name': 'label2',
                'id': 1,
                'keypoints': ['head', 'tail'],
                'skeleton': [1, 2]
            }
        ]
    }

    def test_csv_parser(self):
        csv2coco_parser = pycocowriter.csv2coco.Iterable2COCO(
            pycocowriter.csv2coco.Iterable2COCOConfig(TestCSV2COCO.example_config)
        )
        images, annotations, categories = csv2coco_parser.parse(iter(TestCSV2COCO.example_csv))
        coco_data = COCOData(
            COCOInfo(),
            images,
            annotations,
            [],
            categories
        )
        coco_dict = coco_data.to_dict()
        self.assertDictEqual(coco_dict, TestCSV2COCO.example_csv_as_coco)


def load_tests(loader, tests, ignore):
    return utils.doctests(pycocowriter.csv2coco, tests)


if __name__ == '__main__':
    unittest.main()