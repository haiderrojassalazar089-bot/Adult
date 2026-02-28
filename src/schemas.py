"""
Definición de esquemas de entrada para la API de inferencia.

Responsabilidad:
- Validar estructura y tipos de datos recibidos en producción
- Garantizar consistencia con el dataset de entrenamiento
- Prevenir inputs malformados

Este módulo representa el contrato de datos en producción.
"""

from pydantic import BaseModel, Field


class AdultRequest(BaseModel):
    """
    Esquema de entrada para predicción de ingresos.

    Cada campo debe coincidir exactamente con
    las columnas originales del dataset Adult.
    """

    age: int = Field(..., ge=17, le=90)
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., ge=1, le=16)
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str