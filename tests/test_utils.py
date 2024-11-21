import unittest
import utils
import json
import pycocowriter.utils

class TestAttrDict(unittest.TestCase):

    example_dict = {
        'meta': {
            'skiprows': 0
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
        'keypoint_skeleton': [1,2]
    }

    def test_attrdict(self):
        attr_dict = pycocowriter.utils.AttrDict(TestAttrDict.example_dict)
        self.assertEqual(attr_dict.meta.skiprows, 0)
        self.assertEqual(attr_dict.keypoints[1].name, 'tail')
        self.assertEqual(attr_dict.keypoint_skeleton[1], 2)

def load_tests(loader, tests, ignore):
    return utils.doctests(pycocowriter.utils, tests)

if __name__ == '__main__':
    unittest.main()