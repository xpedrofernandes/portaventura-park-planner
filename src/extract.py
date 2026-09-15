"""Extract structured ride-planning constraints from a natural-language request.

Calls Claude Haiku with a forced tool call so the response is a schema-
validated JSON object rather than free text, then re-validates it in Python
before returning.

Usage:
    python src/extract.py                     # runs the 3 built-in test cases
    python src/extract.py "some request text" # runs on your own text
"""

import json
import os
import sys
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from tags import ALLOWED_TAGS

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = f"""\
You extract ride-planning constraints from a theme park visitor's natural-language request.

The park opens at 09:00 and closes at 22:00.

Only populate a field when the request states or clearly implies it; otherwise omit that field \
from the tool call entirely. Never fill an unknown field with a placeholder string such as \
"<UNKNOWN>", "N/A", or "" -- omitting it is always correct, even for a request that has \
is_planning_request=true but doesn't specify that particular detail (e.g. "we only have 2 \
hours" implies planning intent but gives no clock time, so arrival_time/end_time are both \
omitted, not filled with a placeholder).
- arrival_time / end_time: 24-hour "HH:MM" time strings for when the visit starts/ends. Apply \
these conventions:
  - "from opening" / "when the gates open" -> arrival_time "09:00".
  - "until close" / "until the park closes" / "staying until close" -> end_time "22:00".
  - "morning" as a time-of-day window (e.g. "morning only") -> end_time "12:00".
  - "after dinner" describing when the visitor arrives -> arrival_time "19:00".
  - A dinner reservation, restaurant booking, or any time the visitor must leave by or be done \
by is an end_time, never an arrival_time -- e.g. "we have a reservation at 19:30" means \
end_time "19:30". Only set arrival_time when the request says when the visitor arrives, \
starts, or enters the park.
- max_height_cm: if the visitor mentions a rider's height (e.g. a child), the height in \
centimeters. This means the visitor can only go on rides whose minimum height requirement \
is at or below this value. Convert other units (feet/inches, meters) to centimeters. \
"toddler" -> 90. "small child" -> 110. If multiple children/heights are mentioned, use the \
shortest one -- it's the binding constraint.
- exclude_tags / prefer_tags: category tags for ride types to avoid or prefer. You may ONLY \
use tags from this fixed list: {", ".join(ALLOWED_TAGS)}. Pick the closest matching tag(s) \
for anything the visitor implies (e.g. "big drop" -> "thrill" and/or "heights"); do not invent \
other tags. Always include these two fields, using an empty list if nothing applies.
- is_planning_request: true only if the request contains at least one of: an arrival or end \
time you can resolve to a clock time or one of the named conventions above (e.g. "morning", \
"until close"), party/child details (who's coming, ages), a height constraint, or ride \
preferences/exclusions. false otherwise -- including a vague relative duration with no \
anchoring clock time (e.g. "we only have 2 hours", "just a quick visit") if nothing else in \
the request resolves to arrival_time/end_time/max_height_cm/exclude_tags/prefer_tags either; a \
general question about the park ("what time does the park close?", "is the food any good?"); \
small talk; or unrelated/gibberish text. Always include this field.
"""

TOOL_SCHEMA = {
    "name": "extract_ride_constraints",
    "description": "Record the ride-planning constraints found in the visitor's request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "arrival_time": {
                "type": "string",
                "description": "24-hour HH:MM time the visitor arrives. Omit if not mentioned.",
            },
            "end_time": {
                "type": "string",
                "description": "24-hour HH:MM time the visit ends. Omit if not mentioned.",
            },
            "max_height_cm": {
                "type": "integer",
                "description": "Max rider height in cm implied by the request. Omit if not mentioned.",
            },
            "exclude_tags": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_TAGS},
                "description": "Ride tags to avoid. Empty list if none.",
            },
            "prefer_tags": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_TAGS},
                "description": "Ride tags to prefer. Empty list if none.",
            },
            "is_planning_request": {
                "type": "boolean",
                "description": (
                    "True if the request expresses actual visit-planning intent (arrival/end "
                    "time, party/height details, ride preferences). False for general "
                    "questions, small talk, or unrelated/gibberish text."
                ),
            },
        },
        "required": ["exclude_tags", "prefer_tags", "is_planning_request"],
    },
}

_TIME_RE = __import__("re").compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class ConstraintValidationError(ValueError):
    pass


@dataclass
class Constraints:
    arrival_time: str | None = None
    end_time: str | None = None
    max_height_cm: int | None = None
    exclude_tags: list[str] = field(default_factory=list)
    prefer_tags: list[str] = field(default_factory=list)
    is_planning_request: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "Constraints":
        arrival_time = data.get("arrival_time")
        end_time = data.get("end_time")
        max_height_cm = data.get("max_height_cm")
        exclude_tags = data.get("exclude_tags", [])
        prefer_tags = data.get("prefer_tags", [])
        is_planning_request = data.get("is_planning_request", True)

        for label, value in (("arrival_time", arrival_time), ("end_time", end_time)):
            if value is not None and not _TIME_RE.match(value):
                raise ConstraintValidationError(f"{label!r} is not HH:MM: {value!r}")

        if max_height_cm is not None:
            if not isinstance(max_height_cm, int) or max_height_cm <= 0:
                raise ConstraintValidationError(f"max_height_cm must be a positive int: {max_height_cm!r}")

        if not isinstance(exclude_tags, list) or not all(isinstance(t, str) for t in exclude_tags):
            raise ConstraintValidationError(f"exclude_tags must be a list of strings: {exclude_tags!r}")
        if not isinstance(prefer_tags, list) or not all(isinstance(t, str) for t in prefer_tags):
            raise ConstraintValidationError(f"prefer_tags must be a list of strings: {prefer_tags!r}")
        if not isinstance(is_planning_request, bool):
            raise ConstraintValidationError(
                f"is_planning_request must be a bool: {is_planning_request!r}"
            )

        allowed = set(ALLOWED_TAGS)
        exclude_tags = list(dict.fromkeys(t.lower() for t in exclude_tags if t.lower() in allowed))
        prefer_tags = list(dict.fromkeys(t.lower() for t in prefer_tags if t.lower() in allowed))

        return cls(
            arrival_time=arrival_time,
            end_time=end_time,
            max_height_cm=max_height_cm,
            exclude_tags=exclude_tags,
            prefer_tags=prefer_tags,
            is_planning_request=is_planning_request,
        )


def extract_constraints(request: str, client: anthropic.Anthropic | None = None) -> Constraints:
    if client is None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (expected in .env)")
        client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "extract_ride_constraints"},
        messages=[{"role": "user", "content": request}],
        # This SDK build's typed `messages.create` signature doesn't expose
        # `temperature` directly (checked via inspect.signature) -- pass it
        # through extra_body so it still reaches the API.
        extra_body={"temperature": 0},
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return Constraints.from_dict(tool_use.input)


TEST_CASES = [
    "We're arriving at 10am and need to leave by 3pm. My daughter is 115cm tall, "
    "so nothing too big for her, and please avoid anything with a big drop.",
    "I love roller coasters and spinning rides, but my partner gets seasick on water rides "
    "so let's skip those. We'll be there most of the day.",
    "Just a chill afternoon from 1pm to 5pm, nothing too intense, we'd rather see shows "
    "and easy family rides.",
]


def main() -> None:
    if len(sys.argv) > 1:
        requests_to_run = [" ".join(sys.argv[1:])]
    else:
        requests_to_run = TEST_CASES

    for i, request in enumerate(requests_to_run, 1):
        print(f"--- Test case {i} ---")
        print(f"Request: {request}")
        constraints = extract_constraints(request)
        print(json.dumps(constraints.__dict__, indent=2))
        print()


if __name__ == "__main__":
    main()
