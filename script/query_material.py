import tensorflow as tf
from collections import Counter
from periodictable import elements
import re

# === TFRecord path ===
TFRECORD_PATH = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\oqmd_ree_dataset.tfrecord'


# === Parse TFRecord ===
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'Z': tf.io.VarLenFeature(tf.int64),
        'R': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'n': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    Z = tf.sparse.to_dense(parsed['Z'])
    return Z.numpy()


# === Convert formula string to atomic number counts ===
def formula_to_counter(formula):
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)
    counter = Counter()
    for (symbol, count) in matches:
        count = int(count) if count else 1
        try:
            Z = elements.symbol(symbol).number
            counter[Z] += count
        except Exception as e:
            print(f"❌ Invalid element symbol: {symbol}")
            return None
    return counter


# === Compare two element counters ===
def same_composition(counter1, counter2):
    return counter1 == counter2


# === Main query function ===
def query_formula_in_tfrecord(formula, tfrecord_path):
    target_counter = formula_to_counter(formula)
    if target_counter is None:
        return

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    matches = []

    print(f"🔍 Searching for formula: {formula} ({dict(target_counter)})")

    for i, raw_record in enumerate(dataset):
        try:
            Z = parse_tfrecord_fn(raw_record)
            mol_counter = Counter(Z)
            if same_composition(mol_counter, target_counter):
                matches.append((i, list(Z)))
        except Exception as e:
            print(f"⚠️ Error parsing entry {i}: {e}")

    if matches:
        print(f"\n✅ Found {len(matches)} match(es):")
        for idx, zlist in matches:
            elements_str = ' '.join(f"{elements[z].symbol}" for z in zlist)
            print(f" - Entry {idx}: {zlist} → {elements_str}")
    else:
        print("\n❌ No matching composition found.")


# === Run query ===
if __name__ == '__main__':
    formula_input = input("Enter chemical formula (e.g., Nd2O3): ").strip()
    query_formula_in_tfrecord(formula_input, TFRECORD_PATH)
