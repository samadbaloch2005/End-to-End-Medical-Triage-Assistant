from alembic import context

from triagecare import models  # noqa: F401  (registers tables on Base.metadata)
from triagecare.config import get_settings
from triagecare.db import Base, make_engine

target_metadata = Base.metadata
url = get_settings().sqlalchemy_url


def run_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = make_engine(url)
    with engine.connect() as connection:
        # Batch mode lets ALTER TABLE style changes work on SQLite too.
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
