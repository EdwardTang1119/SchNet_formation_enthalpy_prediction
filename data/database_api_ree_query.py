import requests
import time
import json
import os

# ----------------------- Configuration -----------------------
REE_ELEMENTS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}

API_URL = "https://oqmd.org/oqmdapi/formationenergy"
FIELDS = "delta_e,unit_cell,sites,composition,natoms"
LIMIT = 500
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60  # seconds
SAVE_INTERVAL = 3  # Save every 5 offsets
SAVE_PATH = "oqmd_ree_molecules_resume.json"
OFFSET_PATH = "oqmd_ree_offset_resume.txt"

# ----------------------- Helper Functions -----------------------

def contains_ree(composition):
    try:
        keys = "".join(composition.keys()) if isinstance(composition, dict) else str(composition)
    except Exception:
        keys = str(composition)
    return any(el in keys for el in REE_ELEMENTS)

def fetch_page(offset):
    params = {
        "fields": FIELDS,
        "limit": LIMIT,
        "offset": offset,
        "format": "json"
    }
    retries = 0
    backoff = 1
    while retries < MAX_RETRIES:
        try:
            r = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "10"))
                print(f"⚠️ Rate limited at offset {offset}. Waiting {wait}s...")
                time.sleep(wait)
                retries += 1
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.RequestException as e:
            print(f"⚠️ Error fetching offset {offset}: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            retries += 1
            backoff *= 2
    print(f"❌ Failed to fetch offset {offset} after {MAX_RETRIES} retries.")
    return None  # mark for retry

def save_progress(collected, offset):
    unique = {}
    for entry in collected:
        entry_id = entry.get("entry_id")
        if entry_id is not None:
            unique[entry_id] = entry

    if not unique:
        print(f"⚠️ Warning: No REE entries to save at offset {offset}.")
        return

    try:
        with open(SAVE_PATH, "w") as f:
            json.dump(list(unique.values()), f, indent=2)
        with open(OFFSET_PATH, "w") as f:
            f.write(str(offset))
        print(f"💾 Saved {len(unique)} REE entries at offset {offset}.")
    except Exception as e:
        print(f"❌ Error saving to file: {e}")

def load_previous_progress():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            collected = json.load(f)
        print(f"🔄 Loaded {len(collected)} previously collected entries from file.")
    else:
        collected = []

    if os.path.exists(OFFSET_PATH):
        with open(OFFSET_PATH, "r") as f:
            offset = int(f.read().strip())
        print(f"🔁 Resuming from saved offset: {offset}")
    else:
        offset = 0
    return collected, offset

# ----------------------- Main Loop -----------------------

def main():
    collected, offset = load_previous_progress()
    skipped = 0
    failed_offsets = []

    collected_dict = {e["entry_id"]: e for e in collected if "entry_id" in e}

    while True:
        data = fetch_page(offset)
        if data is None:
            print(f"⚠️ Fetch failed at offset {offset}. Will retry after main pass.")
            failed_offsets.append(offset)
            offset += LIMIT
            continue

        if not data:
            print(f"🛑 No more data at offset {offset}. Ending loop.")
            break

        for entry in data:
            comp = entry.get("composition", {})
            if contains_ree(comp):
                eid = entry.get("entry_id")
                if eid is not None:
                    collected_dict[eid] = entry
            else:
                skipped += 1

        print(f"✅ Fetched offset {offset}: Total REE collected={len(collected_dict)}, skipped={skipped}")

        if (offset // LIMIT) % SAVE_INTERVAL == 0:
            save_progress(list(collected_dict.values()), offset)

        offset += LIMIT
        time.sleep(0.2)

    # Retry failed offsets
    if failed_offsets:
        print(f"\n🔁 Retrying failed offsets: {failed_offsets}")
        for off in failed_offsets:
            data = fetch_page(off)
            if data:
                for entry in data:
                    comp = entry.get("composition", {})
                    if contains_ree(comp):
                        eid = entry.get("entry_id")
                        if eid is not None:
                            collected_dict[eid] = entry
                    else:
                        skipped += 1
                print(f"✅ Recovered offset {off}. Total REE collected={len(collected_dict)}")
            else:
                print(f"❌ Still failed offset {off} after retry.")

    # Final save
    save_progress(list(collected_dict.values()), offset)
    print(f"\n🎉 Finished. Total unique REE entries: {len(collected_dict)} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
