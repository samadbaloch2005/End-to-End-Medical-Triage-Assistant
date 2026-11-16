"""Project commands.

    python -m triagecare.cli setup        migrate, train the model if missing, add demo data
    python -m triagecare.cli migrate      apply database migrations
    python -m triagecare.cli train        (re)train the ML model
    python -m triagecare.cli seed         add demo data to an empty database
    python -m triagecare.cli create-admin USERNAME "Full Name"
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

from triagecare.config import ROOT_DIR, get_settings


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT_DIR / "migrations"))
    command.upgrade(cfg, "head")
    print("Database is up to date.")


def train() -> None:
    # Training scripts use paths relative to the project root.
    os.chdir(ROOT_DIR)
    if not Path("data/triage_records.csv").exists():
        import runpy

        runpy.run_path("data/generate_dataset.py", run_name="__main__")
    from src.train import train as run_training

    run_training()


def seed() -> None:
    from triagecare.db import SessionLocal
    from triagecare.seed import seed as run_seed

    with SessionLocal() as db:
        result = run_seed(db)
    if result.get("skipped"):
        print("Database already has users. Demo data not added.")
    else:
        print(f"Demo data added. Sign in as admin / nurse / reception / dr.hamid with password '{get_settings().demo_password}'.")


def create_admin(username: str, full_name: str) -> None:
    from sqlalchemy import select

    from triagecare.db import SessionLocal
    from triagecare.models import Role, User
    from triagecare.security import hash_password

    password = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.username == username)):
            sys.exit(f"User {username} already exists.")
        db.add(User(username=username, full_name=full_name, role=Role.ADMIN, password_hash=hash_password(password)))
        db.commit()
    print(f"Admin {username} created.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="triagecare")
    sub = parser.add_subparsers(dest="command", required=True)
    setup_p = sub.add_parser("setup", help="migrate, train if needed, seed demo data")
    setup_p.add_argument("--no-seed", action="store_true")
    sub.add_parser("migrate")
    sub.add_parser("train")
    sub.add_parser("seed")
    admin_p = sub.add_parser("create-admin")
    admin_p.add_argument("username")
    admin_p.add_argument("full_name")
    args = parser.parse_args(argv)

    if args.command == "setup":
        migrate()
        if not get_settings().model_path.exists():
            print("No model found, training one (takes under a minute)...")
            train()
        if not args.no_seed:
            seed()
    elif args.command == "migrate":
        migrate()
    elif args.command == "train":
        train()
    elif args.command == "seed":
        seed()
    elif args.command == "create-admin":
        create_admin(args.username, args.full_name)


if __name__ == "__main__":
    main()
