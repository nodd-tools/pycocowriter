from collections.abc import Iterable
import json
import numpy as np

class AttrDict(dict):
    '''
    This class allows javascript-like attribute access for dictionaries.
    Pass in a nested dictionary and recursively access members as attributes.

    Examples
    --------
    >>> attr_dict = AttrDict({'foo': [{'bar': [1,2]}]})
    >>> attr_dict.foo[0].bar[1] == 2
    True
    '''
    def __init__(self, some_dict:dict):
        super().__init__(some_dict)
        self.__dict__ = {
            k: _to_attrdict(v) for k,v in some_dict.items()
        }

def _to_attrdict(el) -> AttrDict:
    '''
    This method facilitates recursive application of AttrDict.
    Probably you don't want to call this method on its own -
    instead, instantiate an `AttrDict`

    Parameters
    ----------
    el: any
        element to recursively convert to AttrDict.
    
    Returns
    -------
    any
        the element, possibly converted to an AttrDict depending on type 
    '''
    # cast dictionaries at AttrDict
    if isinstance(el, dict):
        return AttrDict(el)
    # return strings
    if isinstance(el, str):
        return el
    # try to recurse into other iterable types
    if isinstance(el, Iterable):
        return type(el)((
            _to_attrdict(v) for v in el
        ))
    # return simple types unchanged
    return el

class NPEncoder(json.JSONEncoder):
    '''
    json encoder class used to convert numpy types into plain python types during json.dumps.
    
    Examples
    --------
    >>> json.dumps({'array': np.array([1,2,3])}, cls=NPEncoder)
    '{"array": [1, 2, 3]}'

    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)
    
def skiprows(iterable: Iterable, n: int) -> Iterable:
    '''
    skips the first n rows of iterable.  returns iterable as a convenience

    Parameters
    ----------
    iterable: Iterable
        the iterable to skip rows of
    n: int
        the number of rows to skip

    Returns
    -------
    iterable: Iterable
        the original iterable, but now with n rows having been skipped
    '''
    for i in range(n):
        next(iterable)
    return iterable