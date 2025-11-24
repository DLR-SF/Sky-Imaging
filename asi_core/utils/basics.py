# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides basic helper functions.
"""

def ifnone_do(var, notnone_function, none_function):
    """
    Apply different functions to a variable depending on if it is None or not

    :param var: variable checked if None or not
    :param notnone_function: function applied to var if var is not None
    :param none_function: function applied to var if var is None
    :return: return of the respective function
    """
    return none_function(var) if var is None else notnone_function(var)
