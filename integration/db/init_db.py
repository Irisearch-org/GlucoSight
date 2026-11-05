"""Create the `glucosight` database and its tables.

    python -m integration.db.init_db
"""

from integration import db


def main() -> None:
    db.init_schema()
    with db.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = sorted(next(iter(row.values())) for row in cur.fetchall())
    print(f"database '{db.db_name()}' ready, tables: {', '.join(tables)}")


if __name__ == "__main__":
    main()
