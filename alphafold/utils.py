import tensorflow as tf
import numpy as np
from Bio.PDB import PDBParser
from typing import Dict, List
import os
import numpy as np

# Define the amino acid vocabulary and mappings
VOCABULARY = (
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'Z', 'B', 'U', 'O', '-'
)

# Create mappings from char to index and index to char
CHAR_TO_INT = {char: i for i, char in enumerate(VOCABULARY)}
INT_TO_CHAR = {i: char for i, char in enumerate(VOCABULARY)}

def parse_a3m(a3m_file_path):
    """Parses an A3M file to extract sequences and deletion information."""
    sequences = []
    deletion_matrices = []
    with open(a3m_file_path, 'r') as f:
        lines = f.readlines()

    header = True
    current_sequence = []
    current_deletion_matrix = []

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if not header: # Finished processing a sequence block
                sequences.append(''.join(current_sequence))
                deletion_matrices.append(np.array(current_deletion_matrix))
            # Reset for new sequence
            current_sequence = []
            current_deletion_matrix = []
            header = False
        else:
            # Process sequence line, handling lowercase for deletions
            seq_line = []
            del_line = []
            for char in line:
                if char.isupper() or char == '-': # Regular amino acid or gap
                    seq_line.append(char)
                    del_line.append(0) # No deletion
                elif char.islower(): # Deletion
                    seq_line.append(char.upper()) # Keep the uppercase for the sequence, deletion handled by matrix
                    del_line.append(1) # Mark as deletion
            current_sequence.extend(seq_line)
            current_deletion_matrix.extend(del_line)

    # Add the last sequence block
    if current_sequence:
        sequences.append(''.join(current_sequence))
        deletion_matrices.append(np.array(current_deletion_matrix))

    return sequences, deletion_matrices

def one_hot_encode_msa(msa_sequences, vocabulary=VOCABULARY):
    """One-hot encodes MSA sequences."""
    num_sequences = len(msa_sequences)
    sequence_length = len(msa_sequences[0])
    vocab_size = len(vocabulary)
    char_to_int = {char: i for i, char in enumerate(vocabulary)}

    one_hot_msa = np.zeros((num_sequences, sequence_length, vocab_size), dtype=np.float32)

    for i, seq in enumerate(msa_sequences):
        for j, char in enumerate(seq):
            if char in char_to_int:
                one_hot_msa[i, j, char_to_int[char]] = 1.0
            else:
                # Handle unknown characters, typically mapped to a specific index (e.g., 'X')
                # For now, we'll map to 'X' if it exists in vocab, otherwise ignore/error
                if 'X' in char_to_int:
                    one_hot_msa[i, j, char_to_int['X']] = 1.0
                # else: print(f"Warning: Unknown character '{char}' encountered and not mapped.")
    return one_hot_msa

def calculate_msa_weights(one_hot_msa, gap_character='-'):
    """Calculates MSA sequence weights based on sequence similarity. (Simplified version)
    AlphaFold uses a more sophisticated clustering approach. This is a basic approach.
    """
    num_sequences, sequence_length, vocab_size = one_hot_msa.shape
    weights = np.ones(num_sequences, dtype=np.float32)

    # For simplicity, we'll use a basic weighting strategy: downweighting identical sequences.
    # AlphaFold's actual weighting involves clustering at a certain percentage identity
    # and then assigning weights based on cluster size.
    # This is a placeholder that can be replaced with a more complex implementation if needed.

    # Calculate pairwise sequence identity (ignoring gaps for this simple version)
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            # Count matches, ignoring gaps in *both* sequences
            matches = 0
            total_residues = 0
            seq_i = np.argmax(one_hot_msa[i], axis=-1)
            seq_j = np.argmax(one_hot_msa[j], axis=-1)

            for k in range(sequence_length):
                char_i = INT_TO_CHAR[seq_i[k]]
                char_j = INT_TO_CHAR[seq_j[k]]

                if char_i != gap_character and char_j != gap_character:
                    total_residues += 1
                    if char_i == char_j:
                        matches += 1

            if total_residues > 0:
                identity = matches / total_residues
                if identity > 0.9: # Example threshold for downweighting
                    # This is a very simplistic heuristic. AlphaFold uses a more rigorous method.
                    # For now, let's just mark sequences as similar if they exceed a threshold.
                    # A more proper implementation would involve clustering.
                    pass # Placeholder for actual weight adjustment logic

    # Return uniform weights for now, as a proper implementation requires clustering.
    # This part needs to be updated with a clustering algorithm like MaxCluster.
    return weights / np.sum(weights) * num_sequences # Normalize to sum to number of sequences initially

def get_msa_features(a3m_file_path):
    """Combines MSA parsing, one-hot encoding, deletion matrix, and weight calculation.
    Returns one-hot MSA, deletion matrix, and MSA weights.
    """
    sequences, deletion_matrices = parse_a3m(a3m_file_path)

    if not sequences:
        return None, None, None, None

    # The query sequence is typically the first one in the A3M file
    query_sequence = sequences[0].replace('-', '') # Remove gaps for true query length
    query_sequence_length = len(query_sequence)

    # Ensure all deletion matrices are padded/truncated to a consistent length
    # This assumes that the `parse_a3m` returns deletion_matrices of the same length as sequences
    # and that all sequences have the same length after parsing (common for A3M block alignments)
    msa_length = len(sequences[0])
    processed_deletion_matrices = np.array([np.pad(dm, (0, msa_length - len(dm)), 'constant')[:msa_length] for dm in deletion_matrices])


    one_hot_msa = one_hot_encode_msa(sequences)
    msa_mask = np.where(np.sum(one_hot_msa, axis=-1) > 0, 1.0, 0.0) # Mask valid positions

    # Simplified MSA weights (as discussed in the function's docstring)
    msa_weights = calculate_msa_weights(one_hot_msa)

    return one_hot_msa, processed_deletion_matrices, msa_mask, msa_weights, query_sequence_length

# --- Data Loading and Dataset Creation ---
def _get_ground_truth_structure(pdb_file_path: str, sequence_length: int) -> Dict[str, tf.Tensor]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ground_truth', pdb_file_path)

    ATOM_NAMES_PER_RESIDUE = ['N', 'CA', 'C', 'O', 'CB']
    num_atoms_per_res = len(ATOM_NAMES_PER_RESIDUE)

    true_coords_np = np.zeros((sequence_length, num_atoms_per_res, 3), dtype=np.float32)
    atom_mask_np = np.zeros((sequence_length, num_atoms_per_res), dtype=np.float32)
    true_frames_np = np.zeros((sequence_length, 12), dtype=np.float32)

    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        print(f"Warning: No chains found in {pdb_file_path}")
        return {
            'true_coords': tf.constant(true_coords_np, dtype=tf.float32),
            'true_frames': tf.constant(true_frames_np, dtype=tf.float32),
            'atom_mask': tf.constant(atom_mask_np, dtype=tf.float32)
        }
    chain = chains[0]

    for residue in chain.get_residues():
        res_id = residue.get_id()
        res_seq_num = res_id[1]

        array_idx = res_seq_num - 1

        if 0 <= array_idx < sequence_length:
            for atom_idx, atom_name in enumerate(ATOM_NAMES_PER_RESIDUE):
                if residue.has_id(atom_name):
                    true_coords_np[array_idx, atom_idx] = residue[atom_name].get_coord()
                    atom_mask_np[array_idx, atom_idx] = 1.0

            if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                n_coord = residue['N'].get_coord()
                ca_coord = residue['CA'].get_coord()
                c_coord = residue['C'].get_coord()

                x_axis_vec = n_coord - ca_coord
                z_axis_vec = np.cross(c_coord - ca_coord, x_axis_vec)
                y_axis_vec = np.cross(z_axis_vec, x_axis_vec)

                norm_x = np.linalg.norm(x_axis_vec)
                norm_y = np.linalg.norm(y_axis_vec)
                norm_z = np.linalg.norm(z_axis_vec)

                x_axis = x_axis_vec / norm_x if norm_x > 1e-6 else np.zeros_like(x_axis_vec)
                y_axis = y_axis_vec / norm_y if norm_y > 1e-6 else np.zeros_like(y_axis_vec)
                z_axis = z_axis_vec / norm_z if norm_z > 1e-6 else np.zeros_like(z_axis_vec)

                rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)
                translation_vector = ca_coord

                frame_flat = np.concatenate([rotation_matrix.reshape(-1), translation_vector])
                true_frames_np[array_idx] = frame_flat

    return {
        'true_coords': tf.constant(true_coords_np, dtype=tf.float32),
        'true_frames': tf.constant(true_frames_np, dtype=tf.float32),
        'atom_mask': tf.constant(atom_mask_np, dtype=tf.float32)
    }

def _get_ground_truth_labels(protein_id: str, sequence_length: int, distogram_bins: int, msa_vocab_size: int, plddt_bins: int) -> Dict[str, tf.Tensor]:
    true_distogram_labels = tf.random.uniform((sequence_length, sequence_length, distogram_bins), dtype=tf.float32)
    true_distogram_labels = tf.nn.softmax(true_distogram_labels, axis=-1)

    num_masked_msa_positions = tf.cast(sequence_length * 0.15, tf.int32)
    # Ensure num_masked_msa_positions is at least 1 for the dummy example
    num_masked_msa_positions = tf.maximum(1, num_masked_msa_positions)
    true_masked_msa_labels = tf.random.uniform((num_masked_msa_positions,), minval=0, maxval=msa_vocab_size, dtype=tf.int32)

    true_plddt_labels = tf.random.uniform((sequence_length, plddt_bins), dtype=tf.float32)
    true_plddt_labels = tf.nn.softmax(true_plddt_labels, axis=-1)

    true_chi_angles = tf.random.uniform((sequence_length, 7, 2), dtype=tf.float32)
    true_experiment_flags = tf.constant(np.ones(sequence_length), dtype=tf.float32)

    return {
        'true_distogram_labels': true_distogram_labels,
        'true_masked_msa_labels': true_masked_msa_labels,
        'true_plddt_labels': true_plddt_labels,
        'true_chi_angles': true_chi_angles,
        'true_experiment_flags': true_experiment_flags
    }
import numpy as np

def generate_relative_positional_encodings(sequence_length, max_relative_distance=32):
    """
    Generates relative positional encodings for a given sequence length.
    AlphaFold 2 uses a bucketed encoding for relative distances.
    """
    # Create a matrix where each element (i, j) is the difference j - i
    relative_positions = np.arange(sequence_length)[:, None] - np.arange(sequence_length)[None, :]

    # AlphaFold 2 typically buckets these relative distances.
    # For simplicity, we'll start with a direct encoding and can refine later.
    # A common way is to clip distances and then one-hot encode or use sinusoidal encodings.

    # Example of simple bucketing/clipping
    # Distances are clipped to a max_relative_distance (e.g., -32 to 32) and then potentially binned.
    relative_positions = np.clip(relative_positions, -max_relative_distance, max_relative_distance)

    # A more sophisticated approach would map these to an embedding space,
    # similar to how relative position embeddings are handled in Transformers.
    # For now, we'll return the clipped integer values as a starting point.
    # These integers would then typically be embedded into a higher-dimensional space.

    return relative_positions

def generate_pairwise_features(query_sequence_length, msa_one_hot=None):
    """
    Generates pairwise features for the AlphaFold 2 model.
    This includes relative positional encodings and a placeholder for co-evolutionary signals.

    Args:
        query_sequence_length (int): Length of the query protein sequence.
        msa_one_hot (np.ndarray, optional): One-hot encoded MSA. Used for co-evolutionary signals.

    Returns:
        np.ndarray: A 2D array of relative positional encodings.
        np.ndarray: A placeholder for co-evolutionary features (e.g., predicted contact map).
    """
    # 1. Relative Positional Encodings
    relative_pos_encoding = generate_relative_positional_encodings(query_sequence_length)
    # This 'relative_pos_encoding' would typically be passed through an embedding layer
    # to get a d_pair x L x L tensor, where L is sequence length.
    # For now, it's just L x L integer matrix.

    # 2. Co-evolutionary Signals (Placeholder)
    # AlphaFold 2's Evoformer implicitly learns co-evolutionary signals.
    # However, some initial features might provide a 'head start' or bias.
    # A common approach before Evoformer was direct coupling analysis (DCA) to predict contact maps.
    # For this implementation, we'll create a dummy placeholder. In a full AlphaFold implementation,
    # these might come from an initial attention layer or be implicitly handled by the Evoformer.

    co_evolutionary_features = np.zeros((query_sequence_length, query_sequence_length, 1), dtype=np.float32)
    if msa_one_hot is not None and msa_one_hot.shape[0] > 1: # Requires more than one sequence for co-evolution
        # A very basic, illustrative co-evolutionary signal: average amino acid product
        # This is NOT how AlphaFold does it, but serves as a conceptual placeholder.
        # Actual co-evolutionary signals would be more complex, e.g., derived from covariance matrices.
        msa_mean = np.mean(msa_one_hot, axis=0) # Average residue frequencies at each position
        # Simplified pairwise residue co-occurrence
        # Taking the outer product of mean frequencies can give a very rough idea of co-occurrence
        # for demonstration purposes. Shape (L, 26) @ (26, L) -> (L, L)
        # A proper implementation would look at covariance or mutual information.
        # For now, keep it simple and note that real AlphaFold handles this differently.
        # co_evolutionary_features = np.einsum('ia,jb->ijab', msa_mean, msa_mean)
        # This would result in a (L, L, V, V) tensor. AlphaFold typically uses something like an outer product
        # of MSA features, which are then passed to the Evoformer.
        pass # Keep as zero for now, as direct derivation here is complex and not AF2's core approach

    print(f"Generated relative positional encoding of shape: {relative_pos_encoding.shape}")
    print(f"Generated co-evolutionary features placeholder of shape: {co_evolutionary_features.shape}")

    return relative_pos_encoding, co_evolutionary_features

def parse_pdb_file(pdb_file_path, chain_id=None):
    """Parses a PDB file and extracts C-alpha coordinates and residue types.

    Args:
        pdb_file_path (str): Path to the PDB file.
        chain_id (str, optional): Specific chain ID to extract. If None, extracts first chain.

    Returns:
        tuple: A tuple containing:
            - list: List of (residue_name, residue_number).
            - np.ndarray: A 2D numpy array of C-alpha coordinates (N, 3).
            - list: List of one-letter amino acid codes.
    """
    parser = PDBParser(QUIET=True) # QUIET=True suppresses warnings
    structure = parser.get_structure('template', pdb_file_path)

    atom_coords = []
    residue_data = []
    seq = []

    # AlphaFold primarily uses C-alpha atoms for initial structure representation
    # and works with a simplified amino acid vocabulary.
    AA_MAP = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'UNK': 'X' # Unknown amino acid
    }

    # Iterate over models, chains, residues, and atoms
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for residue in chain:
                    # Only consider standard amino acids
                    if residue.has_id('CA') and residue.get_resname() in AA_MAP:
                        ca_atom = residue['CA']
                        atom_coords.append(ca_atom.get_coord())
                        residue_data.append((residue.get_resname(), residue.get_id()[1]))
                        seq.append(AA_MAP.get(residue.get_resname(), 'X'))
                # If a specific chain_id was requested, break after processing it
                if chain_id is not None:
                    break
        # Break after the first model as AlphaFold typically uses one model per template
        break

    if not atom_coords:
        print(f"Warning: No C-alpha atoms or valid residues found in {pdb_file_path} for chain {chain_id}.")
        return [], np.array([]), []

    return residue_data, np.array(atom_coords), seq

def get_template_features(target_sequence, pdb_templates_info, pdb_files_dir, max_templates=4):
    """Extracts and processes features from selected PDB templates.

    Args:
        target_sequence (str): The amino acid sequence of the target protein.
        pdb_templates_info (list): List of dictionaries, each containing 'pdb_id', 'chain_id', 'evalue', 'identity'
                                   and alignment details (e.g., query_start, query_end, template_start, template_end).
        pdb_files_dir (str): Directory where PDB files are stored.
        max_templates (int): Maximum number of templates to use.

    Returns:
        dict: A dictionary containing various template features:
            - 'template_coords': (Num_templates, L, 3) C-alpha coordinates.
            - 'template_mask': (Num_templates, L) Mask indicating valid C-alpha positions.
            - 'template_aa': (Num_templates, L) One-hot encoded amino acid types.
            - 'template_distances': (Num_templates, L, L) Pairwise C-alpha distances.
            - 'template_orientations': (Num_templates, L, L, K) Placeholder for pairwise orientations.
    """
    # For simplicity, we'll assume pdb_templates_info is sorted by quality (e.g., evalue/identity)
    # and only process up to max_templates.
    selected_templates = pdb_templates_info[:max_templates]

    num_templates = len(selected_templates)
    target_length = len(target_sequence)

    # Initialize tensors for storing features
    # Pad with zeros or a special value if a template doesn't cover the full target or is missing
    template_coords_list = [] # List of (L, 3) arrays
    template_mask_list = []   # List of (L,) arrays
    template_aa_list = []     # List of (L,) arrays (integer encoded AA)

    for template_info in selected_templates:
        pdb_id = template_info['pdb_id']
        chain_id = template_info['chain_id']

        # Assuming PDB files are named pdbid.pdb (e.g., 1xyz.pdb)
        pdb_file_path = os.path.join(pdb_files_dir, f"{pdb_id.lower()}.pdb")

        if not os.path.exists(pdb_file_path):
            print(f"PDB file not found for {pdb_id}. Skipping template.")
            # Add dummy data if PDB file is missing
            template_coords_list.append(np.zeros((target_length, 3), dtype=np.float32))
            template_mask_list.append(np.zeros(target_length, dtype=np.float32))
            template_aa_list.append(np.zeros(target_length, dtype=np.int32)) # 0 for padding/unknown
            continue

        residue_data, ca_coords, template_seq_list = parse_pdb_file(pdb_file_path, chain_id)

        # Placeholder for alignment logic:
        # In a real AlphaFold implementation, you would use the alignment information
        # (e.g., from BLAST output) to map template residues to target residues.
        # For this conceptual implementation, we'll assume a perfect 1:1 mapping
        # and padding if the template is shorter or longer.

        current_template_coords = np.zeros((target_length, 3), dtype=np.float32)
        current_template_mask = np.zeros(target_length, dtype=np.float32)
        current_template_aa = np.zeros(target_length, dtype=np.int32)

        # Simplified mapping (assumes template starts from query_start and aligns sequentially)
        # This needs robust implementation based on actual sequence alignment from template search
        # For now, let's just take the first N residues if template is long enough
        # or pad with zeros if shorter.
        num_aligned_res = min(target_length, len(ca_coords))
        current_template_coords[:num_aligned_res] = ca_coords[:num_aligned_res]
        current_template_mask[:num_aligned_res] = 1.0

        # Convert template_seq_list to integer indices using CHAR_TO_INT
        int_aa_map = {char: i for i, char in enumerate(VOCABULARY)}
        current_template_aa_indices = [int_aa_map.get(aa, int_aa_map.get('X', 0)) for aa in template_seq_list[:num_aligned_res]]
        current_template_aa[:num_aligned_res] = np.array(current_template_aa_indices, dtype=np.int32)

        template_coords_list.append(current_template_coords)
        template_mask_list.append(current_template_mask)
        template_aa_list.append(current_template_aa)

    # Stack all template features into numpy arrays
    # If fewer than max_templates are found, pad with zeros
    if len(template_coords_list) < max_templates:
        for _ in range(max_templates - len(template_coords_list)):
            template_coords_list.append(np.zeros((target_length, 3), dtype=np.float32))
            template_mask_list.append(np.zeros(target_length, dtype=np.float32))
            template_aa_list.append(np.zeros(target_length, dtype=np.int32))

    final_template_coords = np.stack(template_coords_list, axis=0)
    final_template_mask = np.stack(template_mask_list, axis=0)
    final_template_aa = np.stack(template_aa_list, axis=0)

    # Calculate pairwise distances (Euclidean distance between C-alpha atoms)
    # Shape: (Num_templates, L, L)
    template_distances = np.linalg.norm(
        final_template_coords[:, :, None, :] - final_template_coords[:, None, :, :],
        axis=-1
    )

    # Placeholder for pairwise orientations (much more complex, involving rotation matrices/dihedrals)
    # This would typically be a tensor of shape (Num_templates, L, L, K) where K is number of orientation features
    template_orientations = np.zeros((num_templates, target_length, target_length, 1), dtype=np.float32)

    return {
        'template_coords': final_template_coords,
        'template_mask': final_template_mask,
        'template_aa': final_template_aa, # Integer-encoded AA
        'template_distances': template_distances,
        'template_orientations': template_orientations
    }

def generate_sequence_positional_encoding(sequence_length, max_absolute_position=None):
    """
    Generates sequence-level positional encodings for a given sequence length.
    This can be simple integer indices or a more complex sinusoidal encoding.

    Args:
        sequence_length (int): The length of the protein sequence.
        max_absolute_position (int, optional): Maximum absolute position for normalization or binning.
                                               If None, uses sequence_length.

    Returns:
        np.ndarray: A 1D array of positional encodings (L,).
    """
    # AlphaFold 2 typically uses relative positional encodings within the pairwise features
    # and might embed absolute residue indices as part of initial sequence features.
    # For this step, we'll generate a simple integer encoding of the residue index.

    if max_absolute_position is None:
        max_absolute_position = sequence_length

    # Simple integer positional encoding (0 to L-1)
    position_encoding = np.arange(sequence_length, dtype=np.int32)

    # Alternative: Sinusoidal positional encoding (common in Transformers)
    # This is often done in a higher dimension, e.g., (L, embedding_dim)
    # For simplicity, returning the integer index. If needed, this can be expanded
    # to a full sinusoidal embedding matrix or a binned/categorical embedding.
    # For example, to normalize:
    # normalized_positions = position_encoding / max_absolute_position

    print(f"Generated sequence-level positional encoding of shape: {position_encoding.shape}")

    return position_encoding

def get_positional_features(query_sequence_length):
    """
    Orchestrates the generation of all positional features, including sequence-level
    and potentially relative positional encodings (if not already handled by pairwise).

    Args:
        query_sequence_length (int): Length of the query protein sequence.

    Returns:
        dict: A dictionary containing positional features.
    """
    # Sequence-level positional encoding
    sequence_pos_encoding = generate_sequence_positional_encoding(query_sequence_length)

    # Relative positional encodings are already generated as part of `generate_pairwise_features`.
    # They are typically used directly as pairwise features rather than separate 'positional features'.
    # So, for this function, we primarily focus on sequence-level positional features.

    return {
        'sequence_position_encoding': sequence_pos_encoding,
    }

print("Positional encoding generation functions defined.")

def get_input_features(a3m_file_path, target_sequence, pdb_templates_info=None, pdb_files_dir=None, max_templates=4):
    """Conbines all generated features into a single input structure.

    Args:
        a3m_file_path (str): Path to the A3M file for MSA features.
        target_sequence (str): The amino acid sequence of the target protein.
        pdb_templates_info (list, optional): List of dictionaries for PDB templates.
        pdb_files_dir (str, optional): Directory where PDB files are stored.
        max_templates (int): Maximum number of templates to use.

    Returns:
        dict: A dictionary containing all combined input features for the AlphaFold 2 model.
    """
    # 1. Get MSA Features
    # one_hot_msa: (N_seq, L, V)
    # processed_deletion_matrices: (N_seq, L)
    # msa_mask: (N_seq, L)
    # msa_weights: (N_seq,)
    # query_sequence_length: scalar
    one_hot_msa, processed_deletion_matrices, msa_mask, msa_weights, query_sequence_length = get_msa_features(a3m_file_path)

    if one_hot_msa is None:
        print("Error: Could not extract MSA features.")
        return {}

    # 2. Get Pairwise Residue Features
    # relative_pos_encoding: (L, L)
    # co_evolutionary_features: (L, L, 1) (placeholder)
    relative_pos_encoding, co_evolutionary_features = generate_pairwise_features(query_sequence_length, msa_one_hot=one_hot_msa)

    # 3. Get Template Features
    template_features = {}
    if pdb_templates_info and pdb_files_dir:
        template_features = get_template_features(target_sequence, pdb_templates_info, pdb_files_dir, max_templates)
    else:
        print("No PDB templates provided or directory not found. Skipping template feature extraction.")
        # Initialize empty/dummy template features if none are provided
        num_templates = max_templates
        target_length = query_sequence_length
        template_features = {
            'template_coords': np.zeros((num_templates, target_length, 3), dtype=np.float32),
            'template_mask': np.zeros((num_templates, target_length), dtype=np.float32),
            'template_aa': np.zeros((num_templates, target_length), dtype=np.int32),
            'template_distances': np.zeros((num_templates, target_length, target_length), dtype=np.float32),
            'template_orientations': np.zeros((num_templates, target_length, target_length, 1), dtype=np.float32)
        }

    # 4. Get Positional Encodings
    # sequence_position_encoding: (L,)
    positional_features = get_positional_features(query_sequence_length)

    # Combine all features into a single dictionary
    # The exact structure might vary slightly based on the model's input expectations
    # This is a general aggregation.
    combined_features = {
        'msa_one_hot': one_hot_msa,
        'deletion_matrix': processed_deletion_matrices,
        'msa_mask': msa_mask,
        'msa_weights': msa_weights,
        'query_sequence_length': query_sequence_length,
        'relative_pos_encoding': relative_pos_encoding,
        'co_evolutionary_features': co_evolutionary_features,
        **template_features,
        **positional_features
    }

    print("All input features combined successfully.")
    return combined_features

def load_protein_data(protein_id: str, data_paths: Dict[str, str],
                      distogram_bins: int = 64, msa_vocab_size: int = 22, plddt_bins: int = 50,
                      max_templates: int = 4) -> Dict[str, tf.Tensor]:
    a3m_file_path = os.path.join(data_paths['a3m_dir'], f"{protein_id}.a3m")
    pdb_file_path = os.path.join(data_paths['pdb_dir'], f"{protein_id.lower()}.pdb")
    template_info_path = os.path.join(data_paths['template_info_dir'], f"{protein_id}_templates.json")

    if not os.path.exists(a3m_file_path):
        print(f"Error: A3M file not found for {protein_id} at {a3m_file_path}")
        return {}
    if not os.path.exists(pdb_file_path):
        print(f"Error: PDB file not found for {protein_id} at {pdb_file_path}")
        return {}

    pdb_templates_info = []
    if os.path.exists(template_info_path):
        pdb_templates_info = [{'pdb_id': '1abc', 'chain_id': 'A', 'evalue': 1e-10, 'identity': 0.5}] * 2 # Dummy

    sequences, _ = parse_a3m(a3m_file_path)
    if not sequences: return {}
    target_sequence = sequences[0].replace('-', '')

    input_features_dict = get_input_features(
        a3m_file_path=a3m_file_path,
        target_sequence=target_sequence,
        pdb_templates_info=pdb_templates_info,
        pdb_files_dir=data_paths['pdb_dir'],
        max_templates=max_templates
    )

    if not input_features_dict:
        return {}

    query_sequence_length = int(input_features_dict['query_sequence_length'].numpy())
    del input_features_dict['query_sequence_length']

    ground_truth_structure_dict = _get_ground_truth_structure(
        pdb_file_path=pdb_file_path,
        sequence_length=query_sequence_length
    )

    ground_truth_labels_dict = _get_ground_truth_labels(
        protein_id=protein_id,
        sequence_length=query_sequence_length,
        distogram_bins=distogram_bins,
        msa_vocab_size=msa_vocab_size,
        plddt_bins=plddt_bins
    )

    residue_mask = tf.constant(np.ones(query_sequence_length), dtype=tf.float32)

    full_example = {
        **input_features_dict,
        **ground_truth_structure_dict,
        **ground_truth_labels_dict,
        'residue_mask': residue_mask,
        'pair_mask': tf.cast(tf.ones((query_sequence_length, query_sequence_length)), dtype=tf.float32)
    }

    return full_example

# --- Dummy Data Creation Function ---
def create_dummy_data(temp_dir, protein_id, sequence_length, num_msa_sequences):
    os.makedirs(os.path.join(temp_dir, 'a3m_dir'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'pdb_dir'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'template_info_dir'), exist_ok=True)

    with open(os.path.join(temp_dir, 'a3m_dir', f'{protein_id}.a3m'), 'w') as f:
        f.write(f">{protein_id}\n")
        f.write("A" * sequence_length + "\n")
        for i in range(num_msa_sequences - 1):
            f.write(f">seq{i}\n")
            f.write("G" * sequence_length + "\n")

    with open(os.path.join(temp_dir, 'pdb_dir', f'{protein_id.lower()}.pdb'), 'w') as f:
        atom_serial = 1
        for i in range(sequence_length):
            res_seq_num = i + 1
            x_base = float(i * 5.0)
            y_base = float(i * 5.0 + 1)
            z_base = float(i * 5.0 + 2)

            f.write(f"ATOM  {atom_serial:>5}  N   ALA A{res_seq_num:>4}    {x_base:8.3f}{y_base:8.3f}{z_base:8.3f}  1.00 10.00           N  \n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:>5}  CA  ALA A{res_seq_num:>4}    {x_base+1:8.3f}{y_base+1:8.3f}{z_base+1:8.3f}  1.00 10.00           C  \n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:>5}  C   ALA A{res_seq_num:>4}    {x_base+2:8.3f}{y_base+2:8.3f}{z_base+2:8.3f}  1.00 10.00           C  \n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:>5}  O   ALA A{res_seq_num:>4}    {x_base+3:8.3f}{y_base+3:8.3f}{z_base+3:8.3f}  1.00 10.00           O  \n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:>5}  CB  ALA A{res_seq_num:>4}    {x_base+4:8.3f}{y_base+4:8.3f}{z_base+4:8.3f}  1.00 10.00           C  \n")
            atom_serial += 1
        f.write("END\n")

    with open(os.path.join(temp_dir, 'template_info_dir', f'{protein_id}_templates.json'), 'w') as f:
        f.write("[]")

    return {
        'a3m_dir': os.path.join(temp_dir, 'a3m_dir'),
        'pdb_dir': os.path.join(temp_dir, 'pdb_dir'),
        'template_info_dir': os.path.join(temp_dir, 'template_info_dir')
    }

def create_alphafold_dataset(
    protein_ids: List[str],
    data_paths: Dict[str, str],
    batch_size: int,
    distogram_bins: int = 64,
    msa_vocab_size: int = 22,
    plddt_bins: int = 50,
    max_templates: int = 4,
    shuffle_buffer_size: int = 1000,
    num_parallel_calls: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:

    def data_generator():
        for protein_id in protein_ids:
            yield load_protein_data(
                protein_id=protein_id,
                data_paths=data_paths,
                distogram_bins=distogram_bins,
                msa_vocab_size=msa_vocab_size,
                plddt_bins=plddt_bins,
                max_templates=max_templates
            )

    dummy_example = load_protein_data(protein_ids[0], data_paths, distogram_bins, msa_vocab_size, plddt_bins, max_templates)
    if not dummy_example:
        raise ValueError("Could not load dummy example to infer dataset output signature.")

    output_signature = {}
    for key, value in dummy_example.items():
        spec_shape = [None] * len(value.shape)

        if key == 'msa_one_hot':
            spec_shape[-1] = len(VOCABULARY)
        elif key == 'co_evolutionary_features':
            spec_shape[-1] = 1
        elif key == 'template_coords':
            spec_shape[0] = max_templates
            spec_shape[-1] = 3
        elif key == 'template_mask':
            spec_shape[0] = max_templates
        elif key == 'template_aa':
            spec_shape[0] = max_templates
        elif key == 'template_distances':
            spec_shape[0] = max_templates
        elif key == 'template_orientations':
            spec_shape[0] = max_templates
            spec_shape[-1] = 1
        elif key == 'true_coords':
            spec_shape[-2] = 5 # N_atom is fixed to 5
            spec_shape[-1] = 3
        elif key == 'true_frames':
            spec_shape[-1] = 12
        elif key == 'atom_mask':
            spec_shape[-1] = 5 # N_atom is fixed to 5
        elif key == 'true_distogram_labels':
            spec_shape[-1] = distogram_bins
        elif key == 'true_plddt_labels':
            spec_shape[-1] = plddt_bins
        elif key == 'true_chi_angles':
            spec_shape[-2] = 7 # Max_Chi_Angles is fixed to 7
            spec_shape[-1] = 2

        output_signature[key] = tf.TensorSpec(shape=spec_shape, dtype=value.dtype)

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size) \
                     .prefetch(buffer_size=num_parallel_calls)

    padded_shapes = {
        key: output_signature[key].shape for key in output_signature
    }
    padding_values = {
        key: tf.constant(0, dtype=output_signature[key].dtype) for key in output_signature
    }

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=True
    )

    return dataset