# gen_eco_table.py  (ejecútalo en tu máquina local con cualquier Python)
import csv, json, urllib.request, pathlib


# Open the ECO CSV file in the folder
with open("eco_table.csv", "r", encoding="utf-8") as f:
    csv_text = f.read()

reader = csv.DictReader(csv_text.splitlines())

eco_names = {row["ECO Code"]: row["Name"] for row in reader}
path = pathlib.Path("eco_table.json")
path.write_text(json.dumps(eco_names, indent=2, ensure_ascii=False))
print(f"Wrote {len(eco_names)} ECO entries to {path}")