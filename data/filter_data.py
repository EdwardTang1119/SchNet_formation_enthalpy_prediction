import numpy as np
import re

def frac_to_cart(frac_coords, lattice):
    """Convert fractional to cartesian coordinates."""
    frac = np.array(frac_coords)
    lattice = np.array(lattice)
    cart = np.dot(frac, lattice)
    return cart.tolist()

def parse_site_string(site_str):
    """Parse site string like 'Lu @ 0.222 0.778 0.334'."""
    # Regex captures element and fractional coords
    m = re.match(r'(\w+)\s*@\s*([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)', site_str)
    if m:
        element = m.group(1)
        frac_coords = [float(m.group(i)) for i in range(2, 5)]
        return element, frac_coords
    else:
        raise ValueError(f"Cannot parse site string: {site_str}")

def convert_oqmd_to_mp_format(entry):
    lattice = entry['unit_cell']
    delta_e = entry['delta_e']

    sites_mp = []
    for site_str in entry['sites']:
        element, frac_coords = parse_site_string(site_str)
        xyz = frac_to_cart(frac_coords, lattice)
        site_mp = {
            "species": [{"element": element, "occu": 1}],
            "xyz": xyz
        }
        sites_mp.append(site_mp)

    filtered_entry = {
        "structure": {
            "sites": sites_mp
        },
        "formation_energy_per_atom": delta_e
    }
    return filtered_entry
import json

input_oqmd_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\oqmd_ree_molecules_inference.json'  # your raw OQMD JSON
output_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\oqmd_inference_filtered.json'

with open(input_oqmd_path, 'r') as f:
    data_oqmd = json.load(f)

filtered_data = []
for entry in data_oqmd:
    try:
        filtered_entry = convert_oqmd_to_mp_format(entry)
        filtered_data.append(filtered_entry)
    except Exception as e:
        print(f"Skipping entry due to error: {e}")

with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Converted {len(filtered_data)} OQMD entries to MP-like format")
