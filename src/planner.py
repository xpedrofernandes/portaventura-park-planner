"""Same-day ride scheduler.

Takes a Constraints object (see extract.py), filters data/rides.json down to
rides that satisfy the height and exclude_tags constraints, then builds a
visit schedule using the trained LightGBM model (models/) to predict queue
wait *at the time the visitor would actually arrive* at each ride. Rides
matching prefer_tags are a hard priority tier -- scheduled before any
non-matching ride, for as long as a preferred one is still reachable before
end_time -- not just a tiebreaker. Rides themselves take 5 minutes.

Two schedulers are provided:

- build_schedule_zoned (the default, exposed as build_schedule): works
  through one zone at a time -- cheapest-wait-first within it, prefer_tags
  still a hard priority tier -- and only moves to another zone once the
  current one is exhausted (nothing left in it reachable before end_time).
  When moving, an adjacent zone (data/zones.json) is preferred over a
  distant one. Every zone is tried as the starting zone; whichever full
  schedule has the lowest total (wait + walk) time cost wins. This avoids
  the park-wide criss-crossing build_schedule_greedy is prone to, where
  chasing the single cheapest ride anywhere can send the visitor back and
  forth between zones, burning walk time and letting queues grow in transit.
- build_schedule_greedy: the original park-wide approach -- at each step,
  picks the single cheapest-wait reachable ride anywhere in the park. Kept
  only so its cost can be compared against the zoned version (see
  compare_greedy_vs_zoned).

Both are greedy heuristics, not a globally optimal tour -- they won't always
minimize total wait over the whole day, but they're simple, fast, and
re-evaluate every step against the live model prediction.

Usage:
    python src/planner.py   # runs the built-in example, including a
                             # greedy-vs-zoned comparison
"""

import datetime as dt
import functools
import json

import lightgbm as lgb
import pandas as pd

from extract import Constraints

RIDES_PATH = "data/rides.json"
ZONES_PATH = "data/zones.json"
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


def load_rides(path: str = RIDES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_zones(path: str = ZONES_PATH) -> dict:
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


def _travel_minutes(
    from_zone: str | None,
    to_zone: str,
    zone_walk_minutes: dict[str, dict[str, int]] | None,
) -> int:
    """Minutes to walk from `from_zone` to `to_zone`. `from_zone is None` means
    "haven't picked a first ride yet" -- no travel cost. Same zone is always
    WALK_SAME_ZONE_MIN. Different zones use the real per-pair walk time from
    data/zones.json when given, else the flat legacy WALK_DIFF_ZONE_MIN
    (build_schedule_greedy's park-wide approximation)."""
    if from_zone is None:
        return 0
    if to_zone == from_zone:
        return WALK_SAME_ZONE_MIN
    if zone_walk_minutes is not None:
        return zone_walk_minutes[from_zone][to_zone]
    return WALK_DIFF_ZONE_MIN


def _cheapest_in(
    pool: dict[str, dict],
    current_time: int,
    from_zone: str | None,
    end: int,
    booster: lgb.Booster,
    day_of_week: int,
    month: int,
    defaults: dict[str, float],
    zone_walk_minutes: dict[str, dict[str, int]] | None,
) -> dict | None:
    """Cheapest-predicted-wait reachable ride in `pool`, or None if nothing in
    it can be reached before `end`."""
    best = None
    for name, ride in pool.items():
        travel = _travel_minutes(from_zone, ride["zone"], zone_walk_minutes)
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

        if best is None or predicted_wait < best["predicted_wait"]:
            best = {
                "name": name,
                "ride": ride,
                "arrive_at": arrive_at,
                "predicted_wait": predicted_wait,
            }
    return best


def _preferred_pool(pool: dict[str, dict], prefer: set[str]) -> dict[str, dict]:
    """Subset of `pool` matching a prefer_tags tag, or `pool` unchanged if
    `prefer` is empty or nothing in `pool` matches."""
    if not prefer:
        return pool
    preferred = {name: ride for name, ride in pool.items() if prefer & set(ride["tags"])}
    return preferred if preferred else pool


def _make_stop(best: dict) -> dict:
    return {
        "time": _format_hhmm(best["arrive_at"]),
        "ride": best["name"],
        "zone": best["ride"]["zone"],
        "predicted_wait_min": round(best["predicted_wait"], 1),
    }


def build_schedule_greedy(
    constraints: Constraints,
    rides: list[dict] | None = None,
    booster: lgb.Booster | None = None,
    date: dt.date | None = None,
) -> list[dict]:
    """Park-wide greedy scheduler: at each step, picks whichever remaining
    ride anywhere in the park has the lowest predicted wait (prefer_tags
    still a hard priority tier), regardless of zone. Kept for comparison
    against build_schedule_zoned -- see compare_greedy_vs_zoned. This is the
    approach that can send the visitor criss-crossing between zones."""
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
        # Preferred rides are a hard priority tier, not a tiebreaker: as long
        # as any prefer_tags ride is still reachable before end_time, it's
        # scheduled next (cheapest wait among preferred candidates) rather
        # than losing to a cheaper non-preferred ride. Only once no preferred
        # candidate is reachable does selection fall back to all remaining
        # rides.
        best = None
        if prefer:
            preferred = {name: r for name, r in candidates.items() if prefer & set(r["tags"])}
            if preferred:
                best = _cheapest_in(preferred, current_time, current_zone, end, booster, day_of_week, month, defaults, None)

        if best is None:
            best = _cheapest_in(candidates, current_time, current_zone, end, booster, day_of_week, month, defaults, None)

        if best is None:
            break  # nothing reachable before end_time

        schedule.append(_make_stop(best))
        current_time = best["arrive_at"] + round(best["predicted_wait"]) + RIDE_DURATION_MIN
        current_zone = best["ride"]["zone"]
        del candidates[best["name"]]

    return schedule


def _run_zoned_pass(
    start_zone: str,
    candidates: dict[str, dict],
    arrival: int,
    end: int,
    booster: lgb.Booster,
    day_of_week: int,
    month: int,
    defaults: dict[str, float],
    prefer: set[str],
    zone_walk_minutes: dict[str, dict[str, int]],
    adjacency: dict[str, list[str]],
) -> list[dict]:
    """One zone-clustered pass starting in `start_zone`. Mutates `candidates`
    (caller must pass a fresh copy per start zone)."""
    current_time = arrival
    active_zone = start_zone  # which zone's candidates we're working through
    from_zone = None  # None => no ride picked yet, so the first one is free of travel cost
    schedule = []

    while candidates:
        zone_pool = {name: r for name, r in candidates.items() if r["zone"] == active_zone}
        best = None
        if zone_pool:
            best = _cheapest_in(
                _preferred_pool(zone_pool, prefer),
                current_time, from_zone, end, booster, day_of_week, month, defaults, zone_walk_minutes,
            )

        if best is None:
            # Current zone exhausted (empty, or nothing left in it reachable
            # before end_time) -- move on. Prefer an adjacent zone that still
            # has something in it; only fall back to a distant zone if no
            # adjacent one does.
            adjacent_with_candidates = {
                name: r for name, r in candidates.items() if r["zone"] in adjacency.get(active_zone, [])
            }
            move_pool = adjacent_with_candidates if adjacent_with_candidates else candidates
            if not move_pool:
                break

            best = _cheapest_in(
                _preferred_pool(move_pool, prefer),
                current_time, from_zone, end, booster, day_of_week, month, defaults, zone_walk_minutes,
            )
            if best is None:
                break  # nothing reachable anywhere before end_time

        schedule.append(_make_stop(best))
        current_time = best["arrive_at"] + round(best["predicted_wait"]) + RIDE_DURATION_MIN
        active_zone = best["ride"]["zone"]
        from_zone = active_zone
        del candidates[best["name"]]

    return schedule


def build_schedule_zoned(
    constraints: Constraints,
    rides: list[dict] | None = None,
    booster: lgb.Booster | None = None,
    date: dt.date | None = None,
    zones: dict | None = None,
) -> list[dict]:
    """Zone-clustered scheduler (the default -- see build_schedule). Works
    through one zone at a time: cheapest-predicted-wait first within it,
    prefer_tags still a hard priority tier, and only moves to another zone
    once the current one has nothing left reachable before end_time. When
    moving, an adjacent zone (data/zones.json) is preferred over a distant
    one. Every zone is tried as the starting zone; whichever full schedule
    has the lowest total (wait + walk) time cost is returned (ties broken by
    more rides completed)."""
    rides = load_rides() if rides is None else rides
    booster = load_model() if booster is None else booster
    zones = load_zones() if zones is None else zones
    date = date or dt.date.today()

    base_candidates = {r["name"]: r for r in filter_rides(rides, constraints)}
    if not base_candidates:
        return []

    arrival = _parse_hhmm(constraints.arrival_time or DEFAULT_ARRIVAL)
    end = _parse_hhmm(constraints.end_time or DEFAULT_END)
    day_of_week = date.weekday()
    month = date.month
    defaults = _monthly_defaults()[month]
    prefer = set(constraints.prefer_tags)

    best_schedule: list[dict] | None = None
    best_cost = None
    best_num_rides = -1

    for start_zone in zones["zones"]:
        schedule = _run_zoned_pass(
            start_zone,
            dict(base_candidates),
            arrival, end, booster, day_of_week, month, defaults, prefer,
            zones["walk_minutes"], zones["adjacent"],
        )
        stats = schedule_stats(schedule)
        cost = stats["total_wait_min"] + stats["total_walk_min"]

        if (
            best_schedule is None
            or cost < best_cost
            or (cost == best_cost and stats["num_rides"] > best_num_rides)
        ):
            best_schedule = schedule
            best_cost = cost
            best_num_rides = stats["num_rides"]

    return best_schedule


def build_schedule(
    constraints: Constraints,
    rides: list[dict] | None = None,
    booster: lgb.Booster | None = None,
    date: dt.date | None = None,
    zones: dict | None = None,
) -> list[dict]:
    """Default planner entry point -- see build_schedule_zoned."""
    return build_schedule_zoned(constraints, rides=rides, booster=booster, date=date, zones=zones)


def schedule_stats(schedule: list[dict]) -> dict:
    """Total wait/walk minutes, zone-change count, and ride count for a
    schedule returned by either scheduler. Walk time between consecutive
    stops is reconstructed exactly from their `time`/`predicted_wait_min`
    fields (mirroring the arithmetic both schedulers use to advance the
    clock), so this works on any schedule of that shape without needing the
    scheduler to record it separately."""
    if not schedule:
        return {"num_rides": 0, "total_wait_min": 0.0, "total_walk_min": 0.0, "zone_changes": 0}

    total_wait = sum(stop["predicted_wait_min"] for stop in schedule)
    total_walk = 0.0
    zone_changes = 0
    for prev, nxt in zip(schedule, schedule[1:]):
        prev_end = _parse_hhmm(prev["time"]) + round(prev["predicted_wait_min"]) + RIDE_DURATION_MIN
        total_walk += max(0, _parse_hhmm(nxt["time"]) - prev_end)
        if nxt["zone"] != prev["zone"]:
            zone_changes += 1

    return {
        "num_rides": len(schedule),
        "total_wait_min": round(total_wait, 1),
        "total_walk_min": round(total_walk, 1),
        "zone_changes": zone_changes,
    }


def compare_greedy_vs_zoned(
    constraints: Constraints,
    rides: list[dict] | None = None,
    booster: lgb.Booster | None = None,
    date: dt.date | None = None,
    zones: dict | None = None,
) -> dict:
    """Build both schedules for the same constraints and return their stats
    side by side, e.g. for reporting the zoned version's improvement."""
    rides = load_rides() if rides is None else rides
    booster = load_model() if booster is None else booster
    zones = load_zones() if zones is None else zones

    greedy_schedule = build_schedule_greedy(constraints, rides=rides, booster=booster, date=date)
    zoned_schedule = build_schedule_zoned(constraints, rides=rides, booster=booster, date=date, zones=zones)

    return {
        "greedy": {"schedule": greedy_schedule, **schedule_stats(greedy_schedule)},
        "zoned": {"schedule": zoned_schedule, **schedule_stats(zoned_schedule)},
    }


def print_comparison(comparison: dict) -> None:
    g, z = comparison["greedy"], comparison["zoned"]
    print(f"{'':<22} {'Rides':>7} {'Wait (min)':>12} {'Walk (min)':>12} {'Zone changes':>14}")
    print(f"{'Greedy (park-wide)':<22} {g['num_rides']:>7} {g['total_wait_min']:>12.1f} {g['total_walk_min']:>12.1f} {g['zone_changes']:>14}")
    print(f"{'Zoned (clustered)':<22} {z['num_rides']:>7} {z['total_wait_min']:>12.1f} {z['total_walk_min']:>12.1f} {z['zone_changes']:>14}")


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

    print("=== Zoned schedule (default) ===")
    schedule = build_schedule(example_constraints)
    print_schedule(schedule)

    print("\n=== Greedy vs. zoned comparison, same request ===")
    comparison = compare_greedy_vs_zoned(example_constraints)
    print_comparison(comparison)


if __name__ == "__main__":
    main()
