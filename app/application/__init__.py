"""
Application Layer - CQRS Implementation.

Esta capa contiene la implementación del patrón CQRS (Command Query Responsibility Segregation)
y orquesta los domain services para implementar casos de uso de negocio.

Componentes principales:
- Commands: Operaciones de escritura que cambian el estado del sistema
- Queries: Operaciones de lectura que no cambian el estado
- Use Cases: Lógica de orquestación que coordina domain services
- Handlers: Bridge entre la API (FastAPI) y los use cases
- Container: Sistema de inyección de dependencias
"""

# Commands y Queries
from .commands import *
from .queries import *

# Use Cases y Results
from .use_cases import *

# Handlers
from .handlers import *

# Dependency Injection
from .container import get_container, cleanup_container

__all__ = [
    # Container
    "get_container",
    "cleanup_container",
]