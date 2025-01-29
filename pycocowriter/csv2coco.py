import json
import jsonschema
from . import coco, utils
import csv
from collections.abc import Sequence, Iterable


def parse_csv(config: dict, filename: str) -> tuple[
    list[coco.COCOImage], list[coco.COCOAnnotation], list[coco.COCOCategory]]:
    """Helper method to open a csv file and pass it row-by-row into the COCO builder

    Parameters
    ----------
    config : dict
        a dictionary conforming to the Iterable2COCOConfig.SCHEMA
    filename : str
        a csv file to be read and converted to COCO

    Returns
    -------
    images : list[COCOImage]
        a list of COCOImage reflecting the images from the csv file
    annotations : list[COCOAnnotation]
        a list of COCOAnnotation reflecting the annotations from the csv file
    categories : list[COCOCategory]
        a list of COCOCategory reflecting the categories of the annotations
    """
    csv2coco = Iterable2COCO(Iterable2COCOConfig(config))
    with open(filename) as f:
        reader = csv.reader(f)
        images, annotations, categories = csv2coco.parse(reader)
    return images, annotations, categories


def bbox_tlbr2xywh(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    '''
    Convert a bounding box in "top left, bottom right" format 
    to a bounding box in "top left, width height" format

    Parameters
    ----------
    bbox: tuple[int,int,int,int]
        a four-tuple of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Returns
    -------
    bbox: tuple[int,int,int,int]    
        a four-tuple of [top_left_x, top_left_y, width, height]
    '''
    return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


class Iterable2COCOConfig(utils.AttrDict):
    '''
    This class validates a configuration to convert a "flat" iterable type into COCO.
    Because COCO has complex nested and optional types, it is not possible to have a 
    "one-size-fits-all" flat iterable to COCO conversion.  This configuration tells
    the converter which fields are present, and in which columns they are located.
    This class exists only to validate a dict as a valid configuration.

    Parameters
    ----------
    config: dict
        a configuration dictionary adhering to SCHEMA.  Set up this way to be read from .json

    Attributes
    ----------
    SCHEMA: dict
        a jsonschema representing a valid configuration
    '''

    # TODO: This is the 'anything' schema.  Update to reflect the actual rules
    SCHEMA = {}

    def __init__(self, config: dict):
        self._validate_config(config)
        super().__init__(config)

    def _validate_config(self, config: dict):
        jsonschema.validate(config, Iterable2COCOConfig.SCHEMA)


class IterableBBoxParser(object):
    '''
    This class is to help parse bounding boxes from "row" data.  Sometimes these data
    are in different formats, so this class is intended to assist in dealing with these nuances

    Parameters
    ----------
    config: Iterable2COCOConfig
        this configuration should define how (and if) bounding boxes are present in each row
    '''

    def __init__(self, config: Iterable2COCOConfig):
        self.config = config
        self._init_bbox_method()

    def _init_bbox_tlbr(self):
        '''
        configures the `get_bbox` method to get bounding boxes in "top left, width/height" 
        format, given bounding boxes in "top left, bottom right" format
        '''
        self.get_bbox = self._get_bbox_tlbr
        self.bbox_cols = [
            self.config.bbox_tlbr.tlx,
            self.config.bbox_tlbr.tly,
            self.config.bbox_tlbr.brx,
            self.config.bbox_tlbr.bry,
        ]

    def _init_bbox_xywh(self):
        '''
        configures the `get_bbox` method to get bounding boxes in "top left, width/height" 
        format, given bounding boxes in "top left, width/height" format
        '''
        self.get_bbox = self._get_bbox_xywh
        self.bbox_cols = [
            self.config.bbox_xywh.x,
            self.config.bbox_xywh.y,
            self.config.bbox_xywh.w,
            self.config.bbox_xywh.h,
        ]

    def _init_bbox_method(self):
        '''
        dispatches configuration of the `get_bbox` method depending on the 
        contents of the configuration file.
        '''
        if 'bbox_tlbr' in self.config:
            self._init_bbox_tlbr()
        elif 'bbox_xywh' in self.config:
            self._init_bbox_xywh()

    def get_bbox(self, row: Sequence) -> list[int, int, int, int] | None:
        '''
        this method gets overwritten in __init__ if the config has a bbox option

        Parameters
        ----------
        row: Sequence
            a row, e.g. from a csv.  The bounding box should be in some columns of this row
            as defined in the configuration

        Returns
        -------
        bbox: list[int,int,int,int]
            the bounding box as [top_left_x, top_left_y, width, height]
        '''
        return None

    def _get_bbox_tlbr(self, row: Sequence) -> list[int, int, int, int]:
        '''
        gets a bounding box in "top left, width/height" format given bounding box subsetted from a  
        row from, e.g. a csv.  The subset of columns in the input row should be defined in the config, 
        and should contain a bounding box in "top left, bottom right" format.

        Parameters
        ----------
        row: Sequence
            a row, e.g. from a csv.  The bounding box should be in some columns of this row
            as defined in the configuration

        Returns
        -------
        bbox: list[int,int,int,int]
            the bounding box as [top_left_x, top_left_y, width, height]
        '''
        return bbox_tlbr2xywh([int(float(row[i])) for i in self.bbox_cols])

    def _get_bbox_xywh(self, row: Sequence) -> list[int, int, int, int]:
        '''
        gets a bounding box in "top left, width/height" format given bounding box subsetted from a  
        row from, e.g. a csv.  The subset of columns in the input row should be defined in the config, 
        and should contain a bounding box in "top left, width/height" format.

        Parameters
        ----------
        row: Sequence
            a row, e.g. from a csv.  The bounding box should be in some columns of this row
            as defined in the configuration

        Returns
        -------
        bbox: list[int,int,int,int]
            the bounding box as [top_left_x, top_left_y, width, height]
        '''
        return [int(float(row[i])) for i in self.bbox_cols]


class IterableKeypointParser(object):
    '''
    This class is to help parse keypoints from "row" data.  Sometimes these data
    are in different formats, so this class is intended to assist in dealing with these nuances

    Parameters
    ----------
    config: Iterable2COCOConfig
        this configuration should define how (and if) bounding boxes are present in each row
    '''
    FULLY_VISIBLE_COCO_KEYPOINT = 2

    def __init__(self, config: Iterable2COCOConfig):
        self.config = config

    def keypoint_config(self) -> tuple[list[str], list[list[int]]]:
        '''
        get the keypoint layout from the configuration file.  the coco keypoint layout is
        ['kpname1', 'kpname2', ...]
        and also a skeleton
        [edge1, edge2, ...]
        where edges are two-tuples of 1-indexed indexes of keypoints.  For example, if keypoints are:
        ['hip', 'knee', 'ankle'],
        the skeleton would be:
        [[1,2],[2,3]] because the hip has an edge with the knee, and the knee has an edge to the ankle.

        both of these items should be defined in the configuration

        TODO: we only support ONE keypoint structure per configuration right now....
        if you have multiple possible keypoint structures
        e.g. hands and also human poses, then we need to rework this to be more general

        Returns
        -------
        keypoints: list[str]
            the list of keypoint names
        skeleton: list[list[int]]
            the skeleton corresponding to the keypoint names
        '''
        if 'keypoints' not in self.config:
            return None, None
        return (
            [
                keypoint.name for keypoint in self.config.keypoints
            ],
            self.config.keypoint_skeleton
        )

    def get_keypoints(self, row: Sequence) -> list[int]:
        '''
        get keypoints from a "flat" row using expected indices in the row of keypoints 
        defined in self.config

        Parameters
        ----------
        row: Sequence
            a row, e.g. from a csv.  The keypoints should be in some columns of this row
            as defined in the configuration

        Returns
        -------
        keypoints: list[int]
            keypoint locations in form [x1,y1,v1,x2,y2,v2,....] where x,y are the location and v is the
            "visibility" according to the COCO docs
        '''
        if 'keypoints' not in self.config:
            return None
        return sum(
            [
                [
                    int(float(row[keypoint.x])),
                    int(float(row[int(keypoint.y)])),
                    IterableKeypointParser.FULLY_VISIBLE_COCO_KEYPOINT if 'visibility' not in keypoint else int(float(row[keypoint.visibility]))
                ]
                for keypoint in self.config.keypoints
            ],
            []
        )


class Iterable2COCO(object):
    '''
    class providing methods for parsing a "flat" iterable (e.g. a csv) of annotations into COCO
    format.  Each "row" should contain things such as the image filename, 
    and the annotation information (e.g a bounding box or keypoints and a category label)

    Parameters
    ----------
    config: Iterable2COCOConfig
        a configuration dictionary detailing which columns in each row correspond to various COCO
        features such as filename and bounding box coordinates.

    Attributes
    ----------
    bbox_parser: IterableBBoxParser
        a helper for parsing bounding boxes and the concomitant configuration
    keypoint_parser: IterableKeypointParser
        a helper for parsing keypoints and the concomitant configuration
    '''

    def __init__(self, config: Iterable2COCOConfig):
        self.config = config
        self.bbox_parser = IterableBBoxParser(config)
        self.keypoint_parser = IterableKeypointParser(config)

    def _get_scalar(self, field: str, row: Sequence):
        '''
        get a single value from a row given the field name in the configuration

        Parameters
        ----------
        field: str
            the field name as expected in the configuration, and should point to a column index.
            viz. self.config[field] should provide an index into row
        row: Sequence
            an indexable "row" e.g. from a csv file

        Returns
        -------
        scalar: any
           a single value expected for that field. 
        '''
        if field not in self.config:
            return None
        return row[self.config[field]]

    def parse(self, row_iterable: Iterable[Sequence]) -> tuple[
        list[coco.COCOImage], list[coco.COCOAnnotation], list[coco.COCOCategory]
    ]:
        '''
        parse an iterable of rows (e.g. from a csv file) containing image annotation information into
        COCO format.

        Parameters
        ----------
        row_iterable: Iterable[Sequence]
            an iterable of rows containing annotation information

        Returns
        -------
        images: list[COCOImage]
            a list of all unique images listed in the iterable, in COCO format
        annotations: list[COCOAnnotation]
            a list of all annotations listed in the iterable, correctly indexed
            against the images and categories lists
        categories: list[COCOCategory]
            a list of all unique categories listed in the iterable, in COCO format        
        '''
        categories = coco.COCOCategories()
        images = coco.COCOImages()
        annotations = []
        if 'meta' in self.config and 'skiprows' in self.config.meta:
            utils.skiprows(row_iterable, self.config.meta.skiprows)
        keypoint_names, keypoint_skeleton = self.keypoint_parser.keypoint_config()
        for row in row_iterable:
            bbox = self.bbox_parser.get_bbox(row)
            keypoints = self.keypoint_parser.get_keypoints(row)
            filename = self._get_scalar('filename', row)
            width = self._get_scalar('width', row)
            height = self._get_scalar('height', row)
            label = self._get_scalar('label', row)
            images.add(filename, width, height)
            categories.add(label, keypoint_names, keypoint_skeleton)
            annotations.append(
                coco.COCOAnnotation(
                    images.image_map[filename],
                    len(annotations),
                    categories.category_map[label],
                    bbox=bbox,
                    keypoints=keypoints
                )
            )
        return images.images, annotations, categories.categories
