import datetime
import yaml
import uuid
from typing import Annotated
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request,Depends
from sqlmodel import SQLModel, func, col, select,delete, Session,create_engine,update
from sqlalchemy.orm import load_only
from fastapi.encoders import jsonable_encoder

from .utils import (
    Conformation,
    get_conformation_id,
    QCRecord,
    get_record_id,
    RecordStatus,
    Worker,
)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

sqlite_file_name = config.get("database_name", "test_database") + ".db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args, echo=False)


def get_session():
    with Session(engine) as session:
        yield session


def delete_all_workers():
    with Session(engine) as session:
        # session.query(Worker).delete()
        session.exec(delete(Worker))
        session.commit()


SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    delete_all_workers()
    yield


app = FastAPI(lifespan=lifespan)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


@app.get("/")
async def root(session: SessionDep, delay: float = 600.):
    num_records = session.exec(select(func.count()).select_from(QCRecord)).one()
    converged = session.exec(
        select(func.count()).select_from(QCRecord).where(QCRecord.converged == 1)
    ).one()
    pending = session.exec(
        select(func.count()).select_from(QCRecord).where(QCRecord.converged == -1)
    ).one()

    failed = num_records - converged - pending

    num_workers = session.exec(select(func.count()).select_from(Worker)).one()
    timestamp = datetime.datetime.now().timestamp()
    num_active_workers = session.exec(
        select(func.count())
        .select_from(Worker)
        .where(Worker.timestamp > timestamp - delay)
    ).one()

    return {
        "message": "qcAPI is running",
        "converged": converged,
        "pending": pending,
        "failed": failed,
        "num_workers": num_workers,
        "recently_active_workers": num_active_workers,
    }


@app.post("/conformation_id/")
async def return_conformation_id(conformation: Conformation):
    return get_conformation_id(conformation)


@app.put("/qc_result/{worker_id}")
async def create_item(
    worker_id: str, record: QCRecord, session: SessionDep, request: Request
):
    worker = session.get(Worker, uuid.UUID(worker_id))
    if worker is None:
        raise HTTPException(status_code=400, detail="Worker does not exist")
    session.delete(worker)
    session.commit()

    if record.converged < 0:
        return {"message": "Record not processed. Ignoring."}

    id = get_record_id(Conformation(**record.conformation), record.method, record.basis)
    if id != record.id:
        raise HTTPException(status_code=400, detail="ID does not match record")
    prev_record = session.get(QCRecord, id)
    if prev_record is None:
        raise HTTPException(status_code=400, detail="Record does not exist")
    if prev_record.converged == 1:
        raise HTTPException(status_code=210, detail="Record already converged")

    record.timestamp = datetime.datetime.now().timestamp()
    prev_record.sqlmodel_update(record)
    session.add(prev_record)
    session.commit()
    return {"message": "Record stored successfully. Thanks for your contribution!"}


@app.post("/populate/{method}/{basis}")
async def populate_db(
    basis: str,
    method: str,
    conformations: List[Conformation],
    session: SessionDep,
    force: bool = False,
):
    ids = []
    for conformation in conformations:
        id = get_record_id(conformation, method, basis)
        record = QCRecord(id=id, conformation=jsonable_encoder(conformation), method=method, basis=basis)
        prev_record = session.get(QCRecord, id)
        if prev_record is not None:
            if force or prev_record.converged != 1:
                prev_record.sqlmodel_update(record)
                session.add(prev_record)
        else:
            session.add(record)
        ids.append(id)
    session.commit()
    return {"message": "Data inserted successfully", "ids": ids}

@app.put("/reset_all_status/")
async def reset_all_status(session: SessionDep):
    session.exec(update(QCRecord).values(converged=-1))
    session.commit()
    return {"message": "All records reset to pending status"}

@app.put("/reset_failed/")
async def reset_failed(session: SessionDep):
    session.exec(update(QCRecord).where(QCRecord.converged == 0).values(converged=-1))
    session.commit()
    return {"message": "Failed records reset to pending status"}


@app.get("/get_record/{id}")
async def get_record(id: str, session: SessionDep):
    record = session.get(QCRecord, id)
    return record

@app.get("/get_record_status/{id}")
async def get_record_status(id: str, session: SessionDep, worker_id: str = None):
    record = session.get(QCRecord, id)

    if worker_id is not None:
        # update worker timestamp
        worker = session.get(Worker, uuid.UUID(worker_id))
        if worker is not None:
            worker.timestamp = datetime.datetime.now().timestamp()
            session.add(worker)
            session.commit()
            
    return record.converged


# remove record id
@app.delete("/delete_record/{id}")
async def delete_record(id: str, session: SessionDep):
    record = session.get(QCRecord, id)
    session.delete(record)
    session.commit()
    return {"message": "Record deleted successfully"}


# list all records ids
@app.get("/list_record_ids/")
async def list_record_ids(
    session: SessionDep,
    method: str = None,
    basis: str = None,
    status: RecordStatus = None,
):
    statement = select(QCRecord).options(load_only(QCRecord.id))
    if status is not None:
        statement = statement.where(QCRecord.converged == status)
    if method is not None:
        statement = statement.where(QCRecord.method == method)
    if basis is not None:
        statement = statement.where(QCRecord.basis == basis)
    records = session.exec(statement).all()
    ids = [record.id for record in records]
    return ids

# list all records ids
@app.get("/list_records/")
async def list_records(
    session: SessionDep,
    method: str = None,
    basis: str = None,
    status: RecordStatus = None,
):
    statement = select(QCRecord)
    if status is not None:
        statement = statement.where(QCRecord.converged == status)
    if method is not None:
        statement = statement.where(QCRecord.method == method)
    if basis is not None:
        statement = statement.where(QCRecord.basis == basis)
    records = session.exec(statement).all()
    return records


# get next record
@app.get("/get_next_record/")
async def get_next_record(session: SessionDep, request: Request):
    # get the first record that has not converged and sort by timestamp
    record = (
        session.query(QCRecord)
        .filter(QCRecord.converged == -1)
        .order_by(QCRecord.timestamp)
        .first()
    )
    # record = session.exec(
    #     select(QCRecord)
    #     .where(QCRecord.converged == -1)
    #     .order_by(QCRecord.timestamp)
    # ).first()
    if record is None:
        raise HTTPException(status_code=210, detail="No more records")
    timestamp = datetime.datetime.now().timestamp()
    client_host = f"{request.client.host}:{request.client.port}"
    worker = Worker(hostname=client_host, timestamp=timestamp)
    session.add(worker)

    record.timestamp = timestamp
    session.add(record)
    session.commit()
    session.refresh(record)
    return record, worker.id