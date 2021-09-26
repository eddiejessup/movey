import sqlalchemy
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    TIMESTAMP,
    ForeignKey,
    PickleType
)

metadata = MetaData()

pickle_type = PickleType()

run_table = Table(
    "run",
    metadata,
    Column('id', Integer, primary_key=True),
    Column(
        'created_at', TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sqlalchemy.sql.func.now(),
    ),
    Column('params', pickle_type, nullable=False),
)

env_table = Table(
    "env",
    metadata,
    Column('id', Integer, primary_key=True),
    Column('run_id', ForeignKey('run.id'), nullable=False),
    Column('step_view', Integer, nullable=False),
    Column('step_sim', Integer, nullable=False),
    Column('t_sim', Float, nullable=False),
)

chk_table = Table(
    "checkpoint",
    metadata,
    Column('id', Integer, primary_key=True),
    Column(
        'created_at', TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sqlalchemy.sql.func.now(),
    ),
    Column('run_id', ForeignKey('run.id'), nullable=False),
    Column('run_state', pickle_type, nullable=False),
    Column('sim_state', pickle_type, nullable=False),
)

seg_table = Table(
    "seg",
    metadata,
    Column('id', Integer, primary_key=True),
    Column('run_id', ForeignKey('run.id'), nullable=False),
    Column('x1', Float, nullable=False),
    Column('y1', Float, nullable=False),
    Column('x2', Float, nullable=False),
    Column('y2', Float, nullable=False),
)

agent_table = Table(
    "agent",
    metadata,
    Column('agent_id', Integer, nullable=False),
    Column('env_id', ForeignKey('env.id'), nullable=False),
    Column('rx', Float, nullable=False),
    Column('ry', Float, nullable=False),
    Column('ux', Float, nullable=False),
    Column('uy', Float, nullable=False),
    Column('vx', Float, nullable=False),
    Column('vy', Float, nullable=False),
)

def get_engine():
    engine = create_engine('postgresql+psycopg2://ejm@localhost/movey', future=True)
    metadata.create_all(engine)
    return engine
