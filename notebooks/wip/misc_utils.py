import sys
import types
import pandas as pd

from typing import List, Union
from IPython import get_ipython

# Memory Cleaner
# Usage: clear_memory([list of variable(s) to keep], globals_dict=globals()]
def clear_memory(keep_vars: List[Union[pd.DataFrame, dict, list, tuple]] = [], globals_dict=None) -> None:
    if globals_dict is None:
        globals_dict = globals()
    shell = get_ipython()
    keep_ids = {id(k_var) for k_var in keep_vars}
    
    variables = [
        (name, obj, type(obj), obj.memory_usage(index=True).sum() if isinstance(obj, pd.DataFrame) else sys.getsizeof(obj))
        for name, obj in shell.user_ns.items()
        if not name.startswith("__") and not isinstance(obj, types.ModuleType) and isinstance(obj, (pd.DataFrame, dict, list, tuple))
    ]
    
    variables_df = pd.DataFrame(variables, columns=['name', 'object', 'type', 'size'])
    variables_df = variables_df[variables_df['size'] > 5000000]
    remove_vars = [var for var in variables_df['object'] if id(var) not in keep_ids]
    
    remove_names = variables_df.loc[variables_df['object'].apply(lambda x: id(x) not in keep_ids), 'name']
    print(f"Removing variables from memory: {', '.join(remove_names)}")
    
    for name in remove_names:
        del globals_dict[name]