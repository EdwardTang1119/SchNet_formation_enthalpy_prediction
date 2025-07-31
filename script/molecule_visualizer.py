import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
from periodictable import elements as pt
import matplotlib.cm as cm

# === CONFIG ===
TFRECORD_PATH = r"C:\Users\edwar\Chemistry_GNN\formation_enthalpy\data\test_db\oqmd_ree_dataset.tfrecord"
BOND_SCALE = 1.3  # multiplier for sum of covalent radii to detect bonds

# === Parse TFRecord example ===
def parse_example(example_proto):
    features = {
        'Z': tf.io.VarLenFeature(tf.int64),
        'R': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.FixedLenFeature([1], tf.float32),
        'n': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    Z = tf.sparse.to_dense(parsed['Z']).numpy()
    R = tf.sparse.to_dense(parsed['R']).numpy().reshape((-1, 3))
    y = parsed['y'][0].numpy()
    n = parsed['n'][0].numpy()
    return Z, R, y, n

# === Load molecule by index ===
def load_molecule(index):
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    for i, raw_record in enumerate(dataset):
        if i == index:
            return parse_example(raw_record)
    raise IndexError("Index out of range")

# === Bond detection without PBC using covalent radii ===
def find_bonds(Z, R, scale=BOND_SCALE):
    bonds = []
    n = len(Z)
    R = np.array(R)
    for i in range(n):
        for j in range(i + 1, n):
            z1, z2 = Z[i], Z[j]
            r1 = pt[z1].covalent_radius or 1.0
            r2 = pt[z2].covalent_radius or 1.0
            max_dist = scale * (r1 + r2)
            dist = np.linalg.norm(R[i] - R[j])
            if dist <= max_dist:
                bonds.append((i, j))
    return bonds

# === Distinct color palette using matplotlib tab20 ===
def get_color_map(Z_list):
    unique_Z = sorted(set(Z_list))
    cmap = cm.get_cmap('tab20', len(unique_Z))
    color_dict = {z: cmap(i) for i, z in enumerate(unique_Z)}
    return color_dict

# === Get atom counts per element symbol from Z list ===
def get_atom_counts(Z_list):
    symbols = []
    for z in Z_list:
        try:
            symbols.append(pt[z].symbol)
        except Exception:
            symbols.append(f"Z={z}")
    return Counter(symbols)

# === Create legend patches with color + element symbol + count ===
def get_legend_patches(color_dict, atom_counts):
    patches = []
    for z, color in color_dict.items():
        try:
            symbol = pt[z].symbol
        except Exception:
            symbol = f"Z={z}"
        count = atom_counts.get(symbol, 0)
        label = f"{symbol} ({count})"
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    return patches

# === Draw molecule with bonds and color-coded atoms ===
def draw_molecule(ax, Z, R, bonds, color_dict, atom_counts):
    ax.clear()
    ax.set_title("Molecule Structure", fontsize=14)
    ax.set_xlabel("X (Å)", fontsize=12)
    ax.set_ylabel("Y (Å)", fontsize=12)
    ax.set_zlabel("Z (Å)", fontsize=12)
    ax.grid(True)

    R = np.array(R)
    # Plot atoms with color by element
    for i, (z, coord) in enumerate(zip(Z, R)):
        ax.scatter(*coord, s=350, color=color_dict[z], alpha=0.9, edgecolors='k', linewidths=0.6)
        ax.text(*coord, f"{pt[z].symbol if z in pt else z}", fontsize=10, ha='center', va='center', color='black')

    # Plot bonds
    for i, j in bonds:
        ax.plot([R[i][0], R[j][0]],
                [R[i][1], R[j][1]],
                [R[i][2], R[j][2]],
                color='gray', linewidth=1.5, alpha=0.6)

    # Legend with counts
    patches = get_legend_patches(color_dict, atom_counts)
    ax.legend(handles=patches, loc='upper right', fontsize=10, title="Elements")

    # Equal aspect ratio
    max_range = (R.max(axis=0) - R.min(axis=0)).max()
    mid = R.mean(axis=0)
    lim = [[m - max_range / 2, m + max_range / 2] for m in mid]
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_zlim(lim[2])
    ax.set_box_aspect([1, 1, 1])

# === GUI App ===
class MoleculeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("SchNet Molecule Visualizer")
        self.root.geometry("1000x900")  # Bigger window

        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(pady=10)

        tk.Label(self.entry_frame, text="Enter Molecule Index:").pack(side=tk.LEFT)
        self.entry = tk.Entry(self.entry_frame)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.button = tk.Button(self.entry_frame, text="Show Molecule", command=self.show_molecule)
        self.button.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(pady=10)

        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_molecule(self):
        idx_str = self.entry.get()
        if not idx_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid integer index.")
            return
        idx = int(idx_str)

        try:
            # Load molecule from TFRecord
            Z, R, y, n = load_molecule(idx)

            # Find bonds
            bonds = find_bonds(Z, R)

            # Get colors
            color_dict = get_color_map(Z)

            # Count atoms per element symbol
            atom_counts = get_atom_counts(Z)

            # Draw molecule plot with counts in legend
            draw_molecule(self.ax, Z, R, bonds, color_dict, atom_counts)
            self.canvas.draw()

            # Show info label
            count_str = ", ".join(f"{el}: {c}" for el, c in atom_counts.items())
            info_text = (f"Atoms total: {n} | Per element: {count_str}\n"
                         f"Formation Enthalpy per Atom: {y:.5f} eV")
            self.info_label.config(text=info_text)

        except IndexError:
            messagebox.showerror("Error", f"Molecule index {idx} not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{e}")

# === Run the GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MoleculeViewer(root)
    root.mainloop()
