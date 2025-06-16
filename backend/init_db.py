from sqlmodel import SQLModel
from app.database import engine
from app import models

print("Creando tablas...")
SQLModel.metadata.create_all(engine)
print("✅ Tablas creadas exitosamente")
