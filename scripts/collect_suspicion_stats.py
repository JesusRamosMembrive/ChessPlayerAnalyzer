import json
import re
from collections import defaultdict
from pathlib import Path

DEBUG_DIR = Path('debug_results')
ARCHIVE_DIR = Path('archives')
OUTPUT_PATH = Path('data') / 'suspicion_priors.json'

RATING_BUCKET = 200
EXP_BUCKET = 50


# nested dict: rating_bucket -> exp_bucket -> {'suspicious': int, 'total': int}
stats = defaultdict(lambda: defaultdict(lambda: {'suspicious': 0, 'total': 0}))

for dbg_file in DEBUG_DIR.glob('*.json'):
    raw = dbg_file.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        end = raw.find('}\n')
        if end != -1:
            try:
                data = json.loads(raw[: end + 1])
            except json.JSONDecodeError:
                continue
        else:
            end = raw.find('}')
            if end != -1:
                try:
                    data = json.loads(raw[: end + 1])
                except json.JSONDecodeError:
                    continue
            else:
                continue
    player = dbg_file.stem.rsplit('_', 1)[0]
    game_id = data.get('game_id')
    analyzed_at = data.get('analyzed_at')
    if game_id is None or analyzed_at is None:
        continue
    date_str = analyzed_at.split(' ')[0].replace('-', '')
    matches = sorted(ARCHIVE_DIR.glob(f'{player}_{date_str}_*.json'))
    if not matches:
        continue
    archive_data = json.loads(matches[0].read_text())
    if not isinstance(archive_data, list) or game_id >= len(archive_data):
        continue
    game = archive_data[game_id]
    pgn = game.get('pgn', '')
    color = 'White' if game.get('white') == player else 'Black'
    m = re.search(rf'\[{color}Elo "(\d+)"\]', pgn)
    rating = int(m.group(1)) if m else None
    if rating is None:
        continue
    experience = int(game_id)
    is_suspicious = False
    for key in ('suspicious_quality', 'suspicious_timing', 'suspicious_opening'):
        val = data.get(key)
        if isinstance(val, str):
            val = val.lower() == 'true'
        if val:
            is_suspicious = True
            break
    if not is_suspicious and data.get('overall_suspicion_score', 0) > 0:
        is_suspicious = True

    r_bucket = f"{(rating // RATING_BUCKET) * RATING_BUCKET}-{(rating // RATING_BUCKET) * RATING_BUCKET + RATING_BUCKET - 1}"
    e_bucket = f"{(experience // EXP_BUCKET) * EXP_BUCKET}-{(experience // EXP_BUCKET) * EXP_BUCKET + EXP_BUCKET - 1}"
    bucket = stats[r_bucket][e_bucket]
    bucket['total'] += 1
    if is_suspicious:
        bucket['suspicious'] += 1

# Convert counts to alpha/beta parameters
output = {}
for r_bucket, e_dict in stats.items():
    output[r_bucket] = {}
    for e_bucket, counts in e_dict.items():
        s = counts['suspicious']
        t = counts['total']
        output[r_bucket][e_bucket] = {
            'suspicious': s,
            'total': t,
            'alpha': s + 1,
            'beta': t - s + 1,
        }

OUTPUT_PATH.parent.mkdir(exist_ok=True)
with open(OUTPUT_PATH, 'w') as fh:
    json.dump(output, fh, indent=2, sort_keys=True)

print(f"Wrote {OUTPUT_PATH}")
