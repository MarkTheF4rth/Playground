"""
Custom exceptions for easier error handling

General exceptions can be used for convenient error catching
example:

class GeneralException(Exception):
    ...

class SpecificException(GeneralException):
    ...

try:
    raise SpecificException
except GeneralException:
    ...
"""


class CustomException(Exception):
    """ User defined exception"""

    def __init__(self, *args):
        self.text = "An undefined custom exception has been raised"

    def __str__(self):
        """ raising an exception invokes the __str__ override """
        return self.text


class MissingAttributeException(CustomException):
    """ Exceptions for reading files """

    def __init__(self, attr):
        self.text = f"Attribute {attr} has been requested but not provided"
