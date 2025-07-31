import os
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from mendeleev import element
from tqdm import tqdm

# Paths
TFRECORD_PATH = r"C:\Users\edwar\PycharmProjects\GNN_Formation_Enthalpy\Dual_Message_GNN\oqmd_ree_dataset.tfrecord"
SCALER_DIR = r"/Dual_Message_GNN/training_data/scaler"
DATASET_DIR = r"/Dual_Message_GNN/training_data/dataset"
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------- Parse TFRecord --------------------
def parse_data(data):
    features = {
        'Z': tf.io.VarLenFeature(tf.int64),
        'R': tf.io.VarLenFeature(tf.float32),
        'n': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(data, features)
    Z = tf.sparse.to_dense(parsed['Z'])
    R = tf.reshape(tf.sparse.to_dense(parsed['R']), [-1, 3])
    return Z, R, parsed['y'], parsed['n']

def load_data(path):
    dataset = tf.data.TFRecordDataset(path).map(parse_data)
    Z_list, R_list, y_list, n_list = [], [], [], []
    for Z, R, y, n in tqdm(dataset, desc="Loading TFRecord"):
        Z_list.append(Z.numpy())
        R_list.append(R.numpy())
        y_list.append(y.numpy())
        n_list.append(n.numpy())
    print("TFRecord loading complete!")
    return np.array(Z_list, dtype=object), np.array(R_list, dtype=object), np.array(y_list), np.array(n_list)

Z_data, R_data, y_data, n_data = load_data(TFRECORD_PATH)

# -------------------- Optimized Feature Extraction --------------------
def build_element_lookup():
    lookup = {}
    for z in range(1, 119):  # Elements 1 to 118
        try:
            e = element(z)
            block_val = ord(e.block[0]) - ord('a') if e.block else 0
            lookup[z] = [
                e.electronegativity() or 0.0,
                e.covalent_radius or 0.0,
                e.atomic_volume or 0.0,
                e.nvalence() or 0.0,
                block_val
            ]
        except Exception:
            lookup[z] = [0.0, 0.0, 0.0, 0.0, 0.0]
    return lookup

element_lookup = build_element_lookup()

def atomic_features(Z):
    return np.array([
        element_lookup.get(int(z), [0.0] * 5) for z in tqdm(Z, desc="Extracting Atomic Features")
    ], dtype=np.float32)

# -------------------- Flatten and Process --------------------
Z_flat = np.concatenate(Z_data)       # shape [N] — atomic numbers
R_flat = np.concatenate(R_data)       # shape [N, 3] — positions
index = np.repeat(np.arange(len(n_data)), n_data)  # shape [N]
y = y_data                             # shape [M]

X = atomic_features(Z_flat)           # shape [N, 5]

# -------------------- Normalize --------------------
X_scaler = StandardScaler()
X_all_scaled = X_scaler.fit_transform(X)
pickle.dump(X_scaler, open(os.path.join(SCALER_DIR, "X_scaler.pkl"), "wb"))

y_scaler = StandardScaler()
y_all_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
pickle.dump(y_scaler, open(os.path.join(SCALER_DIR, "y_scaler.pkl"), "wb"))

# Save processed data
np.save(os.path.join(DATASET_DIR, "X_scaled.npy"), X_all_scaled)
np.save(os.path.join(DATASET_DIR, "R_scaled.npy"), R_flat)
np.save(os.path.join(DATASET_DIR, "idx_all.npy"), index)
np.save(os.path.join(DATASET_DIR, "y_scaled.npy"), y_all_scaled)
np.save(os.path.join(DATASET_DIR, "Z_.npy"), Z_flat)   # ✅ Save atomic numbers for later use

print("✅ Normalized data saved")
