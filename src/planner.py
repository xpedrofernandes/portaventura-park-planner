"""Greedy same-day ride scheduler.

Takes a Constraints object (see extract.py), filters data/rides.json down to
rides that satisfy the height and exclude_tags constraints, then greedily
walks the visitor's time window: at each step, predict the queue wait (via
the trained LightGBM model in models/) for every remaining ride *at the time
the visitor would actually arrive there* (accounting for a 5-minute walk
within a zone or 10 minutes between zones), and go to whichever ride has the
lowest predicted wait -- with a small bonus for rides matching prefer_tags so
close calls favor them. Rides themselves take 5 minutes.

This is a greedy heuristic (cheapest-next-ride), not a globally optimal tour --
it won't always minimize total wait over the whole day, but it's simple and
fast, and re-evaluates every step against the live model prediction.

Usage:
    python src/planner.py   # runs the built-in example
"""

import datetime as dt
import functools
import json

import lightgbm as lgb
import pandas as pd

from extract import Constraints

RIDES_PATH = "data/rides.json"
MODEL_PATH = "models/wait_time_lgbm.txt"
ATTENDANCE_PATH = "data/raw/attendance.csv"
WEATHER_PATH = "data/raw/weather_data.csv"

FEATURES = ["ride", "hour", "day_of_week", "month", "attendance", "temp", "rain"]

# The model was only trained on rides operating roughly 9am-10pm; clip
# prediction hours to that range rather than extrapolating outside it.
MODEL_MIN_HOUR = 9
MODEL_MAX_HOUR = 22

DEFAULT_ARRIVAL = "09:00"
DEFAULT_END = "22:00"

WALK_SAME_ZONE_MIN = 5
WALK_DIFF_ZONE_MIN = 10
RIDE_DURATION_MIN = 5

# Rides matching a prefer_tag are ranked as if their predicted wait were this
# much shorter, so they win close calls without overriding a much shorter
# wait elsewhere.
PREFERENCE_BONUS_MIN = 5.0


def load_rides(path: str = RIDES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str = MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def filter_rides(rides: list[dict], constraints: Constraints) -> list[dict]:
    """Drop rides too tall for the stated max height, and any with an excluded tag."""
    exclude = set(constraints.exclude_tags)
    out = []
    for ride in rides:
        if (
            constraints.max_height_cm is not None
            and ride["min_height_cm"] is not None
            and ride["min_height_cm"] > constraints.max_height_cm
        ):
            continue
        if exclude & set(ride["tags"]):
            continue
        out.append(ride)
    return out


@functools.lru_cache(maxsize=1)
def _all_ride_names() -> tuple[str, ...]:
    # The model's "ride" categorical codes were fixed at training time to the
    # sorted order of ALL rides in the training data (i.e. all of
    # data/rides.json), regardless of which subset survives filtering here --
    # so this must always use the full, unfiltered ride list.
    return tuple(sorted(r["name"] for r in load_rides()))


@functools.lru_cache(maxsize=1)
def _monthly_defaults() -> dict[int, dict[str, float]]:
    """Historical monthly-average attendance/temp/rain, used as stand-ins for
    an actual future day's attendance and weather (which we have no way to
    know in advance)."""
    attendance = pd.read_csv(ATTENDANCE_PATH)
    attendance = attendance[attendance["FACILITY_NAME"] == "PortAventura World"].copy()
    attendance["USAGE_DATE"] = pd.to_datetime(attendance["USAGE_DATE"])
    attendance_by_month = attendance.groupby(attendance["USAGE_DATE"].dt.month)["attendance"].mean()

    weather = pd.read_csv(WEATHER_PATH, usecols=["dt_iso", "temp", "rain_1h"])
    weather["date"] = pd.to_datetime(
        weather["dt_iso"].str.replace(" UTC", "", regex=False), utc=True
    ).dt.tz_localize(None)
    daily = weather.groupby(weather["date"].dt.date).agg(
        temp=("temp", "mean"), rain=("rain_1h", lambda x: x.fillna(0).sum())
    )
    daily.index = pd.to_datetime(daily.index)
    weather_by_month = daily.groupby(daily.index.month)[["temp", "rain"]].mean()

    defaults = {}
    for month in range(1, 13):
        defaults[month] = {
            "attendance": float(attendance_by_month.get(month, attendance_by_month.mean())),
            "temp": float(weather_by_month["temp"].get(month, weather_by_month["temp"].mean())),
            "rain": float(weather_by_month["rain"].get(month, weather_by_month["rain"].mean())),
        }
    return defaults


def _parse_hhmm(hhmm: str) -> int:
    hours, minutes = map(int, hhmm.split(":"))
    return hours * 60 + minutes


def _format_hhmm(minutes_since_midnight: int) -> str:
    hours, minutes = divmod(minutes_since_midnight, 60)
    return f"{hours:02d}:{minutes:02d}"


def _predict_wait(
    booster: lgb.Booster,
    ride_name: str,
    minute_of_day: int,
    day_of_week: int,
    month: int,
    attendance: float,
    temp: float,
    rain: float,
) -> float:
    hour = min(max(minute_of_day // 60, MODEL_MIN_HOUR), MODEL_MAX_HOUR)
    row = pd.DataFrame(
        [
            {
                "ride": ride_name,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "attendance": attendance,
                "temp": temp,
                "rain": rain,
            }
        ]
    )
    row["ride"] = row["ride"].astype(pd.CategoricalDtype(categories=_all_ride_names()))
    prediction = booster.predict(row[FEATURES])[0]
    return max(0.0, float(prediction))


def build_schedule(
    constraints: Constraints,
    rides: list[dict] | None = None,
    booster: lgb.Booster | None = None,
    date: dt.date | None = None,
) -> list[dict]:
    """Greedily order the rides that satisfy `constraints` between arrival and
    end time, minimizing predicted wait at each step. Returns a list of
    {time, ride, zone, predicted_wait_min} dicts in visit order."""
    rides = load_rides() if rides is None else rides
    booster = load_model() if booster is None else booster
    date = date or dt.date.today()

    candidates = {r["name"]: r for r in filter_rides(rides, constraints)}

    arrival = _parse_hhmm(constraints.arrival_time or DEFAULT_ARRIVAL)
    end = _parse_hhmm(constraints.end_time or DEFAULT_END)
    day_of_week = date.weekday()
    month = date.month
    defaults = _monthly_defaults()[month]
    prefer = set(constraints.prefer_tags)

    current_time = arrival
    current_zone = None
    schedule = []

    while candidates:
        best = None
        for name, ride in candidates.items():
            if current_zone is None:
                travel = 0
            elif ride["zone"] == current_zone:
                travel = WALK_SAME_ZONE_MIN
            else:
                travel = WALK_DIFF_ZONE_MIN

            arrive_at = current_time + travel
            if arrive_at >= end:
                continue  # no time left to reach this ride

            predicted_wait = _predict_wait(
                booster,
                name,
                arrive_at,
                day_of_week,
                month,
                defaults["attendance"],
                defaults["temp"],
                defaults["rain"],
            )
            is_preferred = bool(prefer & set(ride["tags"]))
            score = predicted_wait - (PREFERENCE_BONUS_MIN if is_preferred else 0.0)

            if best is None or score < best["score"]:
                best = {
                    "name": name,
                    "ride": ride,
                    "arrive_at": arrive_at,
                    "predicted_wait": predicted_wait,
                    "score": score,
                }

        if best is None:
            break  # nothing reachable before end_time

        schedule.append(
            {
                "time": _format_hhmm(best["arrive_at"]),
                "ride": best["name"],
                "zone": best["ride"]["zone"],
                "predicted_wait_min": round(best["predicted_wait"], 1),
            }
        )

        current_time = best["arrive_at"] + round(best["predicted_wait"]) + RIDE_DURATION_MIN
        current_zone = best["ride"]["zone"]
        del candidates[best["name"]]

    return schedule


def print_schedule(schedule: list[dict]) -> None:
    if not schedule:
        print("No rides fit the given constraints and time window.")
        return

    print(f"{'Time':<6} {'Ride':<20} {'Zone':<18} {'Predicted wait':>15}")
    for stop in schedule:
        print(f"{stop['time']:<6} {stop['ride']:<20} {stop['zone']:<18} {stop['predicted_wait_min']:>13.1f}m")

    total_wait = sum(s["predicted_wait_min"] for s in schedule)
    print(f"\n{len(schedule)} rides, {total_wait:.1f} predicted total queue minutes.")


def main() -> None:
    example_constraints = Constraints(
        arrival_time="10:00",
        end_time="16:00",
        max_height_cm=None,
        exclude_tags=["water"],
        prefer_tags=["thrill", "coaster"],
    )
    print(f"Constraints: {example_constraints}\n")

    schedule = build_schedule(example_constraints)
    print_schedule(schedule)


if __name__ == "__main__":
    main()
