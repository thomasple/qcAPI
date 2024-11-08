import hashlib
import base64
import uuid
from enum import Enum

import numpy as np


from pydantic import BaseModel
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, PickleType

class RecordStatus(int, Enum):
    converged = 1
    failed = 0
    pending = -1


class Conformation(BaseModel):
    species: list[int]
    coordinates: list[list[float]]
    total_charge: int = 0
    energy: float | None = None
    forces: list[list[float]] | None = None
    dipole: list[float] | None = None
    mbis_charges: list[float] | None = None
    mbis_dipoles: list[list[float]] | None = None
    mbis_quadrupoles: list[list[float]] | None = None
    mbis_octupoles: list[list[float]] | None = None
    mbis_volumes: list[float] | None = None
    mbis_volume_ratios: list[float] | None = None
    mbis_valence_widths: list[float] | None = None
    wiberg_lowdin_indices: list[list[float]] | None = None
    mayer_indices: list[list[float]] | None = None

class QCRecord(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    conformation: Conformation = Field(sa_column=Column(PickleType))
    elapsed_time: float = -1.0
    converged: int = Field(default=-1, index=True)
    method: str = Field(default="none", index=True)
    basis: str = Field(default="none", index=True)
    restricted: bool = Field(default=True, index=True)
    error: str | None = None
    timestamp: float = Field(default=-1., index=True)

def get_conformation_id(conformation: Conformation) -> str:
    coordinates = tuple(np.round(conformation.coordinates, 4).flatten())
    species = tuple(conformation.species)
    total_charge = conformation.total_charge
    h = (coordinates, species, total_charge)

    b64 = base64.urlsafe_b64encode(hashlib.sha3_256(str(h).encode()).digest()).decode("utf-8")
    return b64

def get_record_id(conformation: Conformation, method: str, basis:str) -> str:
    b64 = get_conformation_id(conformation)
    return f"{method.lower()}_{basis.lower()}_{b64}"


class Worker(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hostname: str
    timestamp: float = Field(default=-1., index=True)