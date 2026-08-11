""" This mv_parser.py file is a modified version of the pricer/parser.py file.
It is used to parse the football player data and create a dataset of football players with their market values. """

from mv_estimator.mv_items import PlayerItem
from datetime import date
from typing import Optional
import math

MIN_VALUE = 10_000  # mirrors pricer/parser.py's price sanity filter, adapted to euros

def _valid(x) -> bool:
    """True if x is present and not NaN. NaN is truthy in Python, so a plain
    `if row.get(...)` check silently lets missing numeric fields through."""
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return True

def _age(dob_str: Optional[str]) -> Optional[int]:
    if not dob_str:
        return None
    try:
        dob = date.fromisoformat(dob_str[:10])
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None

def parse(row: dict) -> Optional[PlayerItem]:
    """
    Parse a raw players.csv row into a PlayerItem, or None if it doesn't
    meet the quality bar (missing/invalid value, missing position, etc).
    Mirrors pricer/parser.py's parse(datapoint, category).
    """
    value = row.get("market_value_in_eur")
    position = row.get("position")
    name = row.get("name")

    if not _valid(value) or value < MIN_VALUE:  # also tidies up the value check for consistency
        return None
    if not _valid(position) or position == "Unknown":
        return None
    if not _valid(name):
        return None

    age = _age(row.get("date_of_birth"))
    lines = [f"Name: {name}", f"Position: {position}"]

    sub_position = row.get("sub_position")
    if _valid(sub_position):
        lines[-1] += f" ({sub_position})"
    if _valid(age):
        lines.append(f"Age: {age}")
    if _valid(row.get("foot")):
        lines.append(f"Preferred foot: {row['foot']}")
    if _valid(row.get("height_in_cm")):
        lines.append(f"Height: {int(row['height_in_cm'])} cm")
    if _valid(row.get("country_of_citizenship")):
        lines.append(f"Nationality: {row['country_of_citizenship']}")
    if _valid(row.get("current_club_name")):
        lines.append(f"Current club: {row['current_club_name']}")
    if _valid(row.get("contract_expiration_date")):
        lines.append(f"Contract until: {str(row['contract_expiration_date'])[:10]}")
    
    full = "\n".join(lines)

    return PlayerItem(name=name, position=position, value=float(value), full=full, summary=full)
