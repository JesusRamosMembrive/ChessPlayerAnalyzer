import inspect
import functools
import logging

from app.logging_config import setup_logging

# Configurar logging estructurado (JSON)
setup_logging()


def trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Descubrir la función que llama
        caller = inspect.stack()[1].function

        logging.debug(f"Entrando a {func.__name__}() desde {caller}() con args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.debug(f"Saliendo de {func.__name__}() con resultado={result}")
        return result
    return wrapper