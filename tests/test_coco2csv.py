import unittest
import utils
import pycocowriter.coco2csv

def load_tests(loader, tests, ignore):
    return utils.doctests(pycocowriter.coco2csv, tests)


if __name__ == '__main__':
    unittest.main()
