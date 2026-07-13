import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.expanduser("~/birder_sightings.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_species TEXT,
                confidence REAL,
                top_5 TEXT,
                ebird_info TEXT,
                common_name TEXT,
                bird_id TEXT,
                lat REAL,
                lon REAL,
                gps_fix TEXT,
                temp_f REAL,
                humidity REAL,
                batt_v REAL,
                sensor_utc TEXT,
                created_at TEXT,
                synced INTEGER DEFAULT 0
            )
        """)


def save_pending(predicted_species, confidence, top_5, sensor=None):
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO pending_sightings
                (predicted_species, confidence, top_5, lat, lon, gps_fix,
                 temp_f, humidity, batt_v, sensor_utc, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            predicted_species,
            confidence,
            json.dumps(top_5),
            sensor.get("lat") if sensor else None,
            sensor.get("lon") if sensor else None,
            sensor.get("fix") if sensor else None,
            sensor.get("temp_f") if sensor else None,
            sensor.get("hum") if sensor else None,
            sensor.get("batt_v") if sensor else None,
            sensor.get("utc") if sensor else None,
            datetime.now(timezone.utc).isoformat(),
        ))
    print(f"Sighting saved locally (no wifi): {predicted_species}")


def get_pending():
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM pending_sightings WHERE synced = 0 ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def mark_synced(row_id):
    with _get_conn() as conn:
        conn.execute("UPDATE pending_sightings SET synced = 1 WHERE id = ?", (row_id,))
