import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import Counter
from periodictable import elements
from model import SchNetModel  # your model class here

# === Paths: update these to your local files ===
X_scaled_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\dataset\X_scaled.npy'
R_scaled_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\dataset\R_scaled.npy'
batch_idx_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\dataset\idx_all.npy'
y_scaled_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\dataset\y_scaled.npy'

X_scaler_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\scaler\X_scaler.pkl'
y_scaler_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\scaler\y_scaler.pkl'

model_weights_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\model\schnet_model_oqmd_db.weights.h5'

tfrecord_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\inference_test\oqmd_inference_dataset.tfrecord'

output_csv_path = r'C:\Users\edwar\Chemistry_GNN\formation_enthalpy\inference\predictions_with_formulas_from_tfrecord.csv'


# --- Utility functions ---
def pad_array(arr, shape, pad_value=0):
    padded = np.full(shape, pad_value, dtype=arr.dtype)
    padded[:arr.shape[0], ...] = arr
    return padded

def atomic_nums_to_formula(atomic_nums):
    counts = Counter(atomic_nums)
    formula = ''
    for z in sorted(counts):
        if z > 0:
            try:
                symbol = elements[int(z)].symbol
            except Exception:
                symbol = f"Unknown({z})"
            count = counts[z]
            formula += f"{symbol}{count if count > 1 else ''}"
    return formula

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'Z': tf.io.VarLenFeature(tf.int64),
        'R': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'n': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    Z = tf.sparse.to_dense(parsed['Z'])
    return Z, parsed['y'], parsed['n']


def get_formulas_from_tfrecord(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    formulas = []
    ys = []
    ns = []
    for raw_record in dataset:
        Z, y, n = parse_tfrecord_fn(raw_record)
        Z = Z.numpy()
        formula = atomic_nums_to_formula(Z)
        formulas.append(formula)
        ys.append(y.numpy())
        ns.append(n.numpy())
    return formulas, np.array(ys), np.array(ns)


def main():

    # --- Load scaled inputs ---
    X_scaled = np.load(X_scaled_path)
    R_scaled = np.load(R_scaled_path)
    batch_idx = np.load(batch_idx_path)
    y_scaled = np.load(y_scaled_path)

    # --- Group atoms by molecule ---
    num_mols = np.max(batch_idx) + 1
    mol_X, mol_R, mol_batch = [], [], []
    max_atoms = 0

    for mol_id in range(num_mols):
        mask = batch_idx == mol_id
        mol_X.append(X_scaled[mask])
        mol_R.append(R_scaled[mask])
        mol_batch.append(np.full(np.sum(mask), mol_id, dtype=np.int32))
        max_atoms = max(max_atoms, np.sum(mask))

    X_padded = np.array([pad_array(x, (max_atoms, x.shape[1])) for x in mol_X], dtype=np.float32)
    R_padded = np.array([pad_array(r, (max_atoms, r.shape[1])) for r in mol_R], dtype=np.float32)
    batch_padded = np.array([pad_array(b, (max_atoms,), pad_value=-1) for b in mol_batch], dtype=np.int32)

    print(f"Padded inputs shape: X={X_padded.shape}, R={R_padded.shape}, batch={batch_padded.shape}")

    # --- Load model ---
    num_features = X_padded.shape[2]
    model = SchNetModel(hidden_dim=64, num_interactions=3)
    _ = model((X_padded[:1], R_padded[:1], batch_padded[:1]))  # Build model
    model.load_weights(model_weights_path)
    print("Model weights loaded.")

    # --- Predict ---
    pred_scaled = model.predict((X_padded, R_padded, batch_padded))

    # --- Load y scaler and inverse transform ---
    with open(y_scaler_path, 'rb') as f:
        y_scaler = pickle.load(f)

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    mae = np.mean(np.abs(pred - y_true))
    print(f"MAE (eV/atom): {mae:.4f}")

    # --- Calculate eV per molecule ---
    natoms = np.sum(batch_padded != -1, axis=1)
    pred_eVmolecule = pred * natoms
    true_eVmolecule = y_true * natoms
    abs_error = np.abs(pred_eVmolecule - true_eVmolecule)

    # --- Get formulas from TFRecord ---
    print("Reading formulas from TFRecord...")
    formulas, ys_tfrecord, ns_tfrecord = get_formulas_from_tfrecord(tfrecord_path)
    if len(formulas) != num_mols:
        print(f"Warning: TFRecord formula count ({len(formulas)}) does not match batch count ({num_mols})")

    # --- Save to CSV ---
    df = pd.DataFrame({
        'Formula': formulas,
        'True_eV_per_molecule': true_eVmolecule,
        'Predicted_eV_per_molecule': pred_eVmolecule,
        'Absolute_Error': abs_error
    })

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved predictions + formulas CSV to: {output_csv_path}")

    # --- Plot predicted vs true ---
    plt.figure(figsize=(6,6))
    plt.scatter(true_eVmolecule, pred_eVmolecule, color='blue', alpha=0.6, label='Predictions')
    plt.plot([min(true_eVmolecule), max(true_eVmolecule)],
             [min(true_eVmolecule), max(true_eVmolecule)],
             'r--', label='Ideal: y = x')

    X_reg = true_eVmolecule.reshape(-1,1)
    reg = LinearRegression().fit(X_reg, pred_eVmolecule)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(pred_eVmolecule, reg.predict(X_reg))

    x_line = np.linspace(min(true_eVmolecule), max(true_eVmolecule), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, color='green', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}")

    plt.xlabel("True Formation Enthalpy (eV/molecule)")
    plt.ylabel("Predicted Formation Enthalpy (eV/molecule)")
    plt.title(f"Prediction vs True (MAE = {np.mean(abs_error):.3f} eV/molecule)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
