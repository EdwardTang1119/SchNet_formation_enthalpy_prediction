# === Import Stuff ===
import tensorflow as tf
from tensorflow.keras import layers

# === Radial Basis Function Layer ===
"""
“This layer transforms each atomic pair distance within the cutoff range into a fixed length vector by 
comparing it to a set of RBF centers, which are evenly distributed across the cutoff. Outputs a tensor of 
shape: [num_edges, num_rbf], for better distance-to-energy relationship learning and scaled message passing
"""
class RBFExpansion(tf.keras.layers.Layer):  # making a neural network layer by inheriting TensorFlow.layer
    def __init__(self, num_rbf=64, cutoff=10.0, gamma=10.0):    # pass in parameters: num_rbf, cutoff, gamma
        super().__init__()  # intialize

        self.num_rbf = num_rbf  # amount of RBF used per distance
        self.cutoff = cutoff    # maximum distance of interaction between 2 atoms (10 Angstroms)
        self.gamma = gamma  # controls the stretch of each RBF

        centers = tf.linspace(0.0, cutoff, num_rbf) # generates the centers of each RBF from 0 to cuttoff (10) based on num_rbf
        self.centers = tf.reshape(centers, [1, num_rbf])  # reshapes the centers tensor to [1, num_rbf] for broadcasting

    def call(self, d):  # call function takes parameter of d： a tensor of distances between atoms
        d_expanded = tf.expand_dims(d, -1)  # reshapes tensor to [num_edges, 1] for broadcasting
        return tf.exp(-self.gamma * tf.square(d_expanded - self.centers))   # applies RBF function to every distance in d_expanded, outputs [num_edges, num_rbf] tensor


# === Message Passing Layer ===
"""
This layer is responsible for message passing of edge features. Pass the RBF encoded distance into 2 dense MLP to transform
raw RBF distances into edge feature, perform message pass from edge to atom, then aggregate the edge features into atom after 
a 2 layer dense MLP that updates the atom features after aggregating the message received from neighboring atoms.
"""
class InteractionBlock(tf.keras.layers.Layer):  # making a neural network layer by inheriting TensorFlow.layer
    def __init__(self, hidden_dim): # pass in parameter: hidden_dim (dimension of the feature vector)
        super().__init__()  # intialize

        self.dense_rbf = layers.Dense(hidden_dim)   # dense layer that takes in the RBF encoded distance features and applying learned weights and bias, transform into edge features with shape of hidden_dim
        self.dense_pair = layers.Dense(hidden_dim)  # second dense layer that further transform the edge feature, increase expressiveness of edge features

        self.dense_atom = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='swish'),   # non-linear transformation using swish
            layers.Dense(hidden_dim)    # linear projection to match output tensor shape of atom_feat
        ])  # 2 layer dense MLP that updates the atom features after aggregating the message received from neighboring atoms

    def call(self, atom_feat, pair_feat, recv_idx):   # call function takes parameters - atom_feat: individual atom features, pair_feat: RBF encoded distance feature, send/recv_idx: index of message passing atoms

        edge_messages = self.dense_rbf(pair_feat)   # input the RBF encoded distance vector [num_edges, hidden_dim] and apply dense_rbf, outputs edge feature vector of shape [num_edges, hidden_dim]
        edge_messages = self.dense_pair(edge_messages)     # takes the transformed edge features vector and apply another linear transformation, outputs refined edge feature vector of shape [num_edges, hidden_dim]

        aggregated = tf.math.unsorted_segment_sum(
            edge_messages,
            recv_idx,
            tf.shape(atom_feat)[0]
        )   # aggregate all message received with summation, shape: [num_atoms, hidden_dim]

        return atom_feat + self.dense_atom(aggregated)  # applies dense_atom to the aggregated message and then adding it to the atom features [num_atoms, hidden_dim]


# === SchNet Model ===
"""
Builds the message passing framework and assigning batch idx to map the atom sender and receiver for distance calculation 
and message passing. Use embedding MLP layers to learn atomic features and a 2 layer MLP to predict the final formation 
enthalpy. 
"""
class SchNetModel(tf.keras.Model):  # create SchNet architecture, inherit from TensorFlow.model
    def __init__(self, hidden_dim=64, num_interactions=3):    # pass in parameters - hidden_dim: number of hidden dimensions per layer, num_interactions: number of edge_message passing
        super().__init__()  # initialize

        self.embedding = layers.Dense(hidden_dim)   # set up an MLP to apply learned weights and biases on the atomic features transforms it into a feature vector of hidden_dims
        self.interactions = [InteractionBlock(hidden_dim) for _ in range(num_interactions)] # sets a new interaction block based on the num_interactions, edge_feature message passing hops
        self.rbf = RBFExpansion()   # set RBF function layer
        self.cutoff = 10.0  # sets interaction cutoff distance

        self.out_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='swish'),   # use dense non-linear Swish activation function first layer
            layers.Dense(1) # second dense layer outputs the predicted formation enthalpy
        ])  # prediction MLP that takes in all the feature vectors (atomic and distance) and predicts the formation enthalpy based on learned weights and biases

    def call(self, inputs): # pass in inputs: X, R, batch
        X, R, batch = inputs  # shape - X: [B, N, F](batch idx, atoms, features), R: [B, N, 3](batch idx, atoms, 3D coords of atoms), batch: [B, N](batch idx, atoms)

        def process_single_molecule(args):  # process atoms one molecule at a time based on batch index
            Xi, Ri, bi = args  # shapes - Xi: [N, F], Ri: [N, 3], bi: [N](mask for valid atoms)

            mask = tf.not_equal(bi, -1)       # sets the mask to anything but -1, shape: [N]
            Xi = tf.boolean_mask(Xi, mask)    # removes any padding atoms with mask, shape: [N', F]
            Ri = tf.boolean_mask(Ri, mask)    # removes any padding atoms with mask, shape: [N', 3]

            num_atoms = tf.shape(Ri)[0] # gets the amount of atoms in the molecule
            send_idx, recv_idx = tf.meshgrid(tf.range(num_atoms), tf.range(num_atoms), indexing='ij')   # builds a 2D grid that represents all pairs of atom
            send_idx = tf.reshape(send_idx, [-1])  # flattens the send_idx to a 1D vector, shape: [num_edges]
            recv_idx = tf.reshape(recv_idx, [-1])  # flattens the recv_idx to a 1D vector, shape: [num_edges]

            R_send = tf.gather(Ri, send_idx)       # gets the coordinate of the sender atom, shape: [num_edges, 3]
            R_recv = tf.gather(Ri, recv_idx)       # gets the coordinate of the receiver atom, shape: [num_edges, 3]
            d = tf.norm(R_send - R_recv, axis=1)   # calculates the distance between the sender and receiver atom, shape: [num_edges]

            rbf_feat = self.rbf(d)                 # call RBFExpansion to apply RBF expansion layer on the distance, shape: [num_edges, num_rbf]

            h = self.embedding(Xi)                 # applies the dense embedding MLP to the atomic features (Xi), shape: [num_atoms, hidden_dim]
            for interaction in self.interactions:   # loops through each interaction blocks (3 stacks) and update atomic features
                h = interaction(h, rbf_feat, recv_idx)    # updates atomic features with geometric messages from other edges

            y = tf.reduce_mean(self.out_mlp(h), axis=0)  # predicts the formation enthalpy per atom using out_mlp and averaging it over every atom, shape: [1] (eV/atom)
            return y    # return prediction

        outputs = tf.map_fn(process_single_molecule, (X, R, batch), fn_output_signature=tf.float32) # repeat the prediction process for every molecule in dataset
        return outputs  # returns the predictions
