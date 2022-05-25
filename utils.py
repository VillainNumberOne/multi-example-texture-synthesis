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
