import datetime
import json
from collections.abc import Iterable
import numpy as np
from PIL import Image
from . import utils

class COCOBase(object):
    '''
    base class to facilitate conversion of COCO stuff to a dictionary.

    TODO: refactor COCO classes to extend pycocotools' COCO

    TODO: refactor COCO classes to use AttrDict
    '''
    def _to_dict_fields(self, fields:list[str]) -> dict:
        return {field: self.__dict__[field] for field in fields if self.__dict__[field] is not None}
    def to_dict(self):
        '''convert COCO object to dictionary'''
        raise NotImplementedError('must implement a to_dict method!')


class COCOAnnotation(COCOBase):
    '''
        see https://cocodataset.org/#format-data

        see https://github.com/cocodataset/cocoapi/issues/184

            annotation {
                "id": int,
                "image_id": int,
                "category_id": int,
                "segmentation": RLE or [polygon],
                "area": float,
                "bbox": [x,y,width,height],
                "iscrowd": 0 or 1,
                "keypoints": [x1,y1,v1...],
                "num_keypoints": int
            }

        keypoints should be 3*len(keypoints_category), where keypoints_category is the keypoints in the corresponding category
    '''
    def __init__(self, image_id:int, eye_d:int, category_id:int, bbox:tuple[int]=None, area:float=None, segmentation=None, iscrowd:int=None, keypoints:list[int]=None):
        self.image_id = image_id
        self.id = eye_d
        self.category_id = category_id
        self.bbox = bbox
        area = area or self._compute_area()
        self.area = area
        self.segmentation = segmentation
        self.iscrowd = iscrowd or 0
        self.keypoints = keypoints
        self.num_keypoints = self._compute_num_keypoints()
            
    def _compute_num_keypoints(self):
        if self.keypoints is not None:
            # the number of keypoints is the number of "visible" keypoints.
            return sum([v > 0 for v in self.keypoints[::3]])
        return None

    def _compute_area(self):
        if self.bbox is not None:
            return self.bbox[-1] * self.bbox[-2]
        return None

    def to_dict(self):
        return self._to_dict_fields(
            ['image_id', 'id', 'category_id', 'bbox', 
             'area', 'segmentation', 'iscrowd', 
             'keypoints', 'num_keypoints'])
        

class COCOCategory(COCOBase):
    '''
        see https://cocodataset.org/#format-data
        
        see https://github.com/facebookresearch/Detectron/issues/640
        
            category {
                "id": int,
                "name": str,
                "supercategory": str,
                "keypoints": [str],
                "skeleton": [edge]
            }
        
        An edge is a tuple [a,b] where a,b are 1-indexed indices in the keypoints list.  So if keypoints is ["a", "b"], then [1,2] is an edge between "a" and "b"
    '''
    def __init__(self, name:str, eye_d:int, supercategory:str=None, keypoints:list[str]=None, skeleton:list[list[int]]=None):
        self.name = name
        self.id = eye_d
        self.supercategory = supercategory
        self.keypoints = keypoints
        self.skeleton = skeleton

    def to_dict(self):
        return self._to_dict_fields(
            ['name', 'id', 'supercategory', 'keypoints', 'skeleton']
        )


class COCOLicense(COCOBase):
    '''
        see https://cocodataset.org/#format-data
        
            license {
                "id": int,
                "name": str,
                "url": str,
            }
    '''
    def __init__(self, name:str, eye_d:int, url:str=None):
        self.name = name
        self.id = eye_d
        self.url = url

    def to_dict(self):
        return self._to_dict_fields(
            ['name', 'id', 'url']
        )


class COCOImage(COCOBase):
    '''
        see https://cocodataset.org/#format-data
        
            image {
                "id": int,
                "width": int,
                "height": int,
                "file_name": str,
                "license": int,
                "flickr_url": str,
                "coco_url": str,
                "date_captured": datetime,
            }
    '''
    def __init__(self, 
                 eye_d:int, file_name:str, 
                 width:int=None, height:int=None, 
                 license:int=None, coco_url:str=None, 
                 date_captured:datetime.datetime=None):
        self.id = eye_d
        self.file_name = file_name
        self.width = width
        self.height = height
        if (self.width is None) or (self.height is None):
            self.compute_width_and_height()
        self.license = license
        self.coco_url = coco_url
        self.date_captured = date_captured

    def compute_width_and_height(self):
        with Image.open(self.file_name) as im:
            self.width, self.height = im.size

    def to_dict(self) -> dict:
        the_dict = self._to_dict_fields(
            ['id', 'file_name', 'width', 'height', 'license', 'coco_url']
        )
        if self.date_captured:
            the_dict['date_captured'] = self.date_captured.isoformat()
        return the_dict

class COCOInfo(COCOBase):
    '''
        see https://cocodataset.org/#format-data
            
            info {
                "year": int,
                "version": str,
                "description": str,
                "contributor": str,
                "url": str,
                "date_created": datetime,
            }
    '''
    def __init__(self, year:int=None, version:str=None, description:str=None, contributor:str=None, url:str=None, date_created:datetime.datetime=None):
        self.year = year
        self.version = version
        self.description = description
        self.contributor = contributor
        self.url = url
        self.date_created = date_created

    def to_dict(self) -> dict:
        the_dict = self._to_dict_fields(
            ['year', 'version', 'description', 'contributor', 'url']
        )
        if self.date_created:
            the_dict['date_created'] = self.date_created.isoformat()
        return the_dict



class COCOCategories(object):
    '''
    helper class to hold the index on categories so that we can find categories by name or by index

    Parameters
    ----------
    categories: list[COCOCategory]
        existing list of COCOCategories if available

    Attributes
    ----------
    categories: list[COCOCategory]
        a list of unique categories
    category_map: dict[str, COCOCategory]
        maps category names to categories
    '''
    def __init__(self, categories: list[COCOCategory] | None = None):
        categories = categories or []
        # category ids MUST match their index+1 in the category list!
        self.categories = categories
        for i, category in enumerate(self.categories):
            assert category.id == i+1
        self.category_map = {category.name: category.id for category in self.categories}

    def add(self, label:str, keypoints:list[str]=None, skeleton:list[list[int]]=None) -> COCOCategory:
        '''
        Add a new category to the list.  Updates the map as well

        Parameters
        ----------
        label: str
            the string name of this category
        keypoints: list[str]
            the list of keypoint names for this category, if applicable
        skeleton: list[list[int]]
            the skeleton for this category, if applicable

        Returns
        -------
        category: COCOCategory
            returns the built COCOCategory
        '''
        if label not in self.category_map:
            category = COCOCategory(label, len(self.categories)+1, keypoints=keypoints, skeleton=skeleton)
            self.categories.append(category)
            self.category_map[self.categories[-1].name] = self.categories[-1].id
        return self.category_map[label]

    def __len__(self):
        return len(self.categories)

class COCOImages(object):
    '''
    helper class to hold the index on images so that we can find images by name or by index

    Parameters
    ----------
    images: list[COCOImage]
        existing list of COCOImages if available

    Attributes
    ----------
    images: list[COCOImage]
        a list of unique images
    image_map: dict[str, COCOImage]
        maps image names to images
    '''
    def __init__(self, images: list[COCOImage] | None = None):
        images = images or []
        # image ids MUST match their index+1 in the image list!
        self.images = images
        for i, image in enumerate(self.images):
            assert image.id == i+1
        self.image_map = {image.filename: image.id for image in self.images}

    def add(self, filename:str, width:int=None, height:int=None, 
            url:str=None, license:int=None, date_captured:datetime.datetime=None):
        '''
        Add a new image to the list.  Updates the map as well

        Parameters
        ----------
        filename: str
            the filename of this image
        width: int
            the width of this image, if known
        height: int
            the height of this image, if known
        url: str
            the url at which this image can be downloaded
        license: int
            the index in the list of COCOLicense with the applicable license
        date_captured: datetime.datetime
            the date and time when the image was captured

        Returns
        -------
        image: COCOImage
            returns the built COCOImage
        '''
        if filename not in self.image_map:
            image = COCOImage(len(self.images)+1, filename, 
                              width=width, height=height, coco_url=url, 
                              license=license, date_captured=date_captured)
            self.images.append(image)
            self.image_map[self.images[-1].file_name] = self.images[-1].id
        return self.image_map[filename]

    def __len__(self):
        return len(self.images)


class COCOData(COCOBase):
    '''
        see https://cocodataset.org/#format-data
        
            coco {
                "info": info,
                "images": [image],
                "annotations": [annotation],
                "licenses": [license],
                "categories": [category]
            }
    '''
    
    def __init__(self, info:COCOInfo, images:list[COCOImage], annotations:list[COCOAnnotation], licenses:list[COCOLicense], categories:list[COCOCategory]):
        self.info = info
        self.images = images
        self.annotations = annotations
        self.licenses = licenses
        self.categories = categories

    def to_dict(self) -> dict:
        return {
            'info': self.info.to_dict(),
            'images': [image.to_dict() for image in self.images],
            'annotations': [annotation.to_dict() for annotation in self.annotations],
            'licenses': [license.to_dict() for license in self.licenses],
            'categories': [category.to_dict() for category in self.categories]
        }

    def to_json(self, filename:str=None):
        '''
        dumps this COCO to json.  If a filename is provided, writes to disk, else, returns
        the string JSON

        Parameters
        ----------
        filename: str
            the optional location to which we should write the json
    
        Returns
        -------
        json: str | None
            if no filename is provided, returns the JSON COCO data as a string
        '''
        if filename is None:
            return json.dumps(self.to_dict(), cls=utils.NPEncoder)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, cls=utils.NPEncoder)