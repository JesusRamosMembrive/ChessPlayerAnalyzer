import math, numpy as np

def clean_json_numbers(obj):
    """
    Reemplaza NaN e ±Inf por None y convierte numpy.* a tipos nativos.
    Úsala justo antes de insertar JSON en la BD.
    """
    if isinstance(obj, dict):
        return {k: clean_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(v) for v in obj]
    # numpy -> python
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    # sanitizar
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj