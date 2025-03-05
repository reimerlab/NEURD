"""
utility and wrapper functions to help output separate documentation derived from the documentation in the code

"""
from typing import List,Union
from functools import wraps

def tag(
    tags:Union[str,List[str]]="default",
    ):
    """
    Adds an attribute called "tag" to function that is a list
    of the string or list of strings sent in the argument

    Parameters
    ----------
    tags : Union[str,List[str]], optional
        _description_, by default "default"
    """
    
    def inner_tag(func):
        if not hasattr(func,"tags"):
            func.tags = []
            
        if "str" in str(type(tags)):
            my_tags = [tags]
        else:
            my_tags = tags
        for t in my_tags:
            if t not in func.tags:
                func.tags.append(t)
        return func
    return inner_tag

def name(
    name:str,
    ):
    
    def inner_func(func):
        func.alternative_name = name
        return func
    return inner_func

