import json
import tensorflow as tf
from periodictable import elements as pt_elements  # pip install periodictable

# === Update your paths here ===
JSON_PATH = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\oqmd_inference_filtered.json'  # JSON with only structure and formation_energy_per_atom
TFRECORD_PATH = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\oqmd_inference_dataset.tfrecord'

# === TFRecord serialization helpers ===
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(Z, R, y, n):
    feature = {
        'Z': _int64_feature(Z),
        'R': _float_feature(R),
        'y': _float_feature([y]),
        'n': _int64_feature([n]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# === Parse a single entry ===
def parse_entry(entry):
    sites = entry['structure']['sites']
    Z = []
    R = []

    for site in sites:
        element = site['species'][0]['element']
        Z.append(pt_elements.symbol(element).number)
        R.extend(site['xyz'])  # flatten x, y, z

    n = len(Z)
    y = entry['formation_energy_per_atom']

    return Z, R, y, n

# === Main conversion function ===
def json_to_tfrecord(json_path, tfrecord_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, entry in enumerate(data):
            try:
                Z, R, y, n = parse_entry(entry)
            except Exception as e:
                print(f"Skipping entry {i} due to error: {e}")
                continue

            example = serialize_example(Z, R, y, n)
            writer.write(example)

            if i % 100 == 0:
                print(f"Processed {i} / {len(data)} entries")

    print(f"\nTFRecord written to: {tfrecord_path}")

if __name__ == "__main__":
    json_to_tfrecord(JSON_PATH, TFRECORD_PATH)
