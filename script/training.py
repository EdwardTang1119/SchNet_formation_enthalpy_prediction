import os
import numpy as np
import tensorflow as tf
from model import SchNetModel
from sklearn.model_selection import train_test_split

# === Paths ===
DATASET_DIR = r"C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\dataset"
SCALER_DIR = r"C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\scaler"
X_PATH = os.path.join(DATASET_DIR, "X_scaled.npy")
R_PATH = os.path.join(DATASET_DIR, "R_scaled.npy")
IDX_PATH = os.path.join(DATASET_DIR, "idx_all.npy")
Y_PATH = os.path.join(DATASET_DIR, "y_scaled.npy")
MODEL_SAVE_PATH = r"C:\Users\edwar\Chemistry_GNN\formation_enthalpy\model\schnet_model_oqmd_db.weights.h5"

# === Load Data ===
X = np.load(X_PATH)
R = np.load(R_PATH)
batch_idx = np.load(IDX_PATH)
y = np.load(Y_PATH)

print(f"✅ Loaded shapes: X = {X.shape}, R = {R.shape}, batch = {batch_idx.shape}, y = {y.shape}")

# === Group atoms by molecule ===
num_mols = np.max(batch_idx) + 1
mol_X, mol_R, mol_batch, mol_y = [], [], [], []

print("📦 Grouping atoms per molecule...")
for mol_id in range(num_mols):
    mask = batch_idx == mol_id
    mol_X.append(X[mask])
    mol_R.append(R[mask])
    mol_batch.append(np.full(np.sum(mask), mol_id, dtype=np.int32))
    mol_y.append(y[mol_id])
print("✅ Grouping complete.")

# === Train/Validation Split ===
train_data, val_data = train_test_split(
    list(zip(mol_X, mol_R, mol_batch, mol_y)),
    test_size=0.2,
    random_state=42
)

# === Dataset Creation Function ===
def create_dataset(data, batch_size):
    def generator():
        for x, r, b, y_val in data:
            yield (x.astype(np.float32), r.astype(np.float32), b.astype(np.int32)), np.float32(y_val)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    return ds.padded_batch(
        batch_size,
        padded_shapes=(
            ([None, 5], [None, 3], [None]),
            []
        ),
        padding_values=(
            (tf.constant(0, tf.float32),
             tf.constant(0, tf.float32),
             tf.constant(-1, tf.int32)),
            tf.constant(0, tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

# === Build Datasets ===
BATCH_SIZE = 32
train_dataset = create_dataset(train_data, BATCH_SIZE)
val_dataset = create_dataset(val_data, BATCH_SIZE)

# === Build & Compile Model ===
num_features = X.shape[1]
model = SchNetModel(num_features=num_features, hidden_dim=64, num_interactions=3)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse', metrics=['mae'])

# === Callbacks ===
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# === Train ===
EPOCHS = 100
print("🚀 Training started...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)
print("✅ Training complete!")

# === Save Best Model ===
model.save_weights(MODEL_SAVE_PATH)
print(f"💾 Model weights saved to {MODEL_SAVE_PATH}")

# === Evaluate on Validation Set ===
print("📊 Evaluating on validation set...")
val_loss, val_mae = model.evaluate(val_dataset, verbose=0)
print(f"✅ Final Normalized Validation MAE: {val_mae:.6f}")
