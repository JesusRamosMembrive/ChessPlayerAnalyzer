import json, pathlib

_ECO_PATH = pathlib.Path(__file__).with_suffix(".json")
ECO_NAMES: dict[str, str] = json.loads(_ECO_PATH.read_text(encoding="utf-8"))