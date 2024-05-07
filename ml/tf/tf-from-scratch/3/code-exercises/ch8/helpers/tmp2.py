import sys
sys.path.append('../..')

from common.settings import *
from common.classes import *

'''
def get_variable_name(obj, namespace):
    return [name for name, value in namespace.items() if value is obj][0]
 '''
# Example usage:
custom_variable = [1, 2, 3]
custom_variable_name = get_variable_name(custom_variable, locals())
print(f"Variable name using custom function: {custom_variable_name}")
