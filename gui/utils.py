from warnings_errors import *

def process_str_value(str_value, default=None, target=int, min_value=None, max_value=None):
    if isinstance(str_value, str) and str_value != '':
        try:
            value = target(str_value)
        except Exception:
            return default
        
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
    else:
        return default
    return value

def process_tensors(tensor1, tensor2):
    if len(tensor1) != len(tensor2):
        incompatible_tensors()
        new_length = min(len(tensor1), len(tensor2))
        return tensor1[:new_length], tensor2[:new_length]
    return tensor1, tensor2

def process_tensor_lists(tensor_list1, tensor_list2):
    lengths = list(map(len, tensor_list1 + tensor_list2))
    new_length = min(lengths)
    if not all(el == lengths[0] for el in lengths):
        incompatible_tensors()
        def cut(tensor):
            return tensor[:new_length]
        return list(map(cut, tensor_list1)), list(map(cut, tensor_list2))
    else:
        return tensor_list1, tensor_list2