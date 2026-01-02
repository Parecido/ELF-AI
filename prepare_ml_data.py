import glob
import numpy as np
import os
import pandas as pd
import pwfn

BOHR_TO_ANGSTROM = 0.529177

def extract_tables(lines, start_keyword1, start_keyword2):
    """
    Parse a text file (already split into lines) and extract two tables that appear
    after two different start markers.

    The function assumes:
      - Table 1 begins right after a line containing start_keyword1
      - Table 2 begins right after a line containing start_keyword2

    Args:
        lines (list[str]): Full file contents split into lines.
        start_keyword1 (str): Marker line that indicates the start of table 1.
        start_keyword2 (str): Marker line that indicates the start of table 2.

    Returns:
        tuple[list[str], list[str]]: (table1_lines, table2_lines) where each is a list
        of stripped, non-empty lines belonging to the corresponding table.
    """
    table1, table2 = [], []
    capture_table1 = capture_table2 = False

    for line in lines:
        if start_keyword1 in line:
            capture_table1, capture_table2 = True, False
            continue
        if start_keyword2 in line:
            capture_table2, capture_table1 = True, False
            continue
        
        if capture_table1:
            if line.strip() == "" and len(table1) > 5:
                capture_table1 = False
            else:
                table1.append(line.strip())
        
        if capture_table2:
            if line.strip() == "" and len(table2) > 5:
                capture_table2 = False
            else:
                table2.append(line.strip())

    table1 = [line for line in table1 if line]
    table2 = [line for line in table2 if line]
    return table1, table2

def create_dataframe(table1_lines, table2_lines):
    """
    Convert the parsed table lines into a single DataFrame.

    Table 1 is assumed to contain attractor/basin info including:
      - coordinates (x, y, z)
      - basin type (int)
      - basin name/atoms label (string)

    Table 2 is assumed to contain at least a numeric population field "N"
    located at parts[3] after splitting.

    Args:
        table1_lines (list[str]): Lines belonging to table 1.
        table2_lines (list[str]): Lines belonging to table 2.

    Returns:
        pd.DataFrame: Combined DataFrame with columns:
          X, Y, Z, basin_type, basin_name, N
        where each row corresponds to a basin/attractor.
    """
    data1 = []
    for line in table1_lines:
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        basin_type = int(parts[-1])
        basin_atoms = parts[5]
        data1.append([x, y, z, basin_type, basin_atoms])

    data2 = []
    for line in table2_lines:
        parts = line.split()
        N = float(parts[3])
        data2.append([N])

    df1 = pd.DataFrame(data1, columns=["X", "Y", "Z", "basin_type", "basin_name"])
    df2 = pd.DataFrame(data2, columns=["N"])
    return pd.concat([df1, df2], axis=1)

def read_wfn_atoms(file_path):
    """
    Load a .wfn (wavefunction) file using pwfn and return atomic data as a DataFrame.

    The pwfn parser is expected to expose:
      - wavefunction.at_name: iterable of element symbols (e.g., ["C", "H", ...])
      - wavefunction.at_position: array-like of shape (n_atoms, 3) in bohr

    Args:
        file_path (str): Path to the .wfn file.

    Returns:
        pd.DataFrame: Atom table with columns:
          atom_id (0-based), element, X, Y, Z
        where coordinates are in bohr.
    """
    with open(file_path, "r") as f:
        wavefunction = pwfn.loads(f.read())
        atom_names = list(wavefunction.at_name)
        atom_positions = wavefunction.at_position
        X, Y, Z = np.array_split(atom_positions, 3, axis=1)
        ids = np.arange(len(atom_names), dtype=int)

    return pd.DataFrame({
        "atom_id": ids,
        "element": atom_names,
        "X": X.ravel(),
        "Y": Y.ravel(),
        "Z": Z.ravel()
    })

def find_two_min_distances(df, atoms):
    """
    For each basin point, compute distances to all atoms and store:
      - the smallest (min1) and second smallest (min2) distances (Å)
      - the corresponding atom ids and element names for those closest atoms

    Distances are computed between basin coordinates (df: X,Y,Z) and atom coordinates
    (atoms: X,Y,Z). Both are assumed to be in bohr and then converted to Å.

    Args:
        df (pd.DataFrame): Basin table with columns X,Y,Z.
        atoms (pd.DataFrame): Atom table with columns atom_id, element, X,Y,Z.

    Returns:
        pd.DataFrame: df with extra columns:
          min1, min2, min1_atom_id, min2_atom_id, min1_atom_name, min2_atom_name
    """
    def euclidean_distance(row1, row2):
        X1, Y1, Z1 = row1["X"], row1["Y"], row1["Z"]
        X2, Y2, Z2 = row2["X"], row2["Y"], row2["Z"]
        distance = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)
        return distance * BOHR_TO_ANGSTROM

    min1, min2 = [], []
    min1_id, min2_id = [], []
    min1_name, min2_name = [], []

    for _, row1 in df.iterrows():
        distances = []
        for _, row2 in atoms.iterrows():
            distance = euclidean_distance(row1, row2)
            distances.append((distance, int(row2["atom_id"]), row2["element"]))
        distances.sort()
        min1.append(distances[0][0]); min2.append(distances[1][0])
        min1_id.append(distances[0][1]); min2_id.append(distances[1][1])
        min1_name.append(distances[0][2]); min2_name.append(distances[1][2])
    
    df["min1"] = min1
    df["min2"] = min2
    df["min1_atom_id"] = min1_id
    df["min2_atom_id"] = min2_id
    df["min1_atom_name"] = min1_name
    df["min2_atom_name"] = min2_name
    return df

def calculate_distance_between_atoms(df, atoms):
    """
    Compute the interatomic distance r (Å) between the two atoms closest to each basin.

    Steps:
      1) For each basin row, fetch the two closest atoms by atom_id columns
         (min1_atom_id, min2_atom_id)
      2) Apply simple filters to remove basins that likely represent non-bonding regions:
           - basin_type == 5 (often non-bonded / lone-pair)
           - very asymmetric nearest-neighbor distances:
             min2 > 1.5 * min1 and min1 > 0.1 Å
      3) Compute the distance between those two atoms and store as r
      4) Add a simple "bond" label by sorting element symbols (e.g., "CH", "HO", "CC")
      5) Drop rows where r is NaN

    Args:
        df (pd.DataFrame): Basin DataFrame with nearest-atom columns already added.
        atoms (pd.DataFrame): Atom DataFrame with coordinates.

    Returns:
        pd.DataFrame: Filtered DataFrame with added columns:
          r (Å) and bond (string)
    """
    def euclidean_distance(row, atoms_df):
        atom1 = atoms_df.iloc[row["min1_atom_id"]]
        atom2 = atoms_df.iloc[row["min2_atom_id"]]

        if row["basin_type"] == 5:
            return np.nan
        if row["min2"] > 1.5 * row["min1"] and row["min1"] > 0.1:
            return np.nan

        X1, Y1, Z1 = atom1["X"], atom1["Y"], atom1["Z"]
        X2, Y2, Z2 = atom2["X"], atom2["Y"], atom2["Z"]
        distance = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)
        return distance * BOHR_TO_ANGSTROM

    def get_bond_name(row):
        atom1 = row["min1_atom_name"]
        atom2 = row["min2_atom_name"]
        return "".join(sorted([atom1, atom2]))

    df["r"] = df.apply(lambda x: euclidean_distance(x, atoms), axis=1)
    df["bond"] = df.apply(lambda x: get_bond_name(x), axis=1)
    df = df.dropna()
    return df

def merge_attractors(df):
    """
    Merge multiple attractors/basins that correspond to the same atom pair.

    This groups by a canonical (element, id) tuple of the two closest atoms, ensuring
    that (A,B) and (B,A) end up in the same group.

    Aggregation:
      - bond: first (same for the group)
      - r: mean (average bond length across attractors)
      - N: sum (total population across attractors)

    Args:
        df (pd.DataFrame): Basin DataFrame containing min1/min2 atom info, r, and N.

    Returns:
        pd.DataFrame: One row per unique atom pair with columns:
          bond, r, N, atom1_name, atom2_name, atom1_id, atom2_id
        with r and N rounded to 2 decimals.
    """
    def sorted_atoms(row):
        atoms = [
            (row["min1_atom_name"], row["min1_atom_id"]),
            (row["min2_atom_name"], row["min2_atom_id"]),
        ]
        atoms.sort(key=lambda x: (x[0], x[1]))
        return tuple(atoms)

    df["sorted_atoms"] = df.apply(sorted_atoms, axis=1)

    grouped = df.groupby("sorted_atoms", as_index=False).agg({
        "bond": "first",
        "r": "mean",
        "N": "sum"
    })

    grouped["atom1_name"] = grouped["sorted_atoms"].apply(lambda t: t[0][0])
    grouped["atom2_name"] = grouped["sorted_atoms"].apply(lambda t: t[1][0])
    grouped["atom1_id"] = grouped["sorted_atoms"].apply(lambda t: t[0][1])
    grouped["atom2_id"] = grouped["sorted_atoms"].apply(lambda t: t[1][1])

    grouped = grouped.drop(columns=["sorted_atoms"]).reset_index(drop=True)
    grouped["r"] = grouped["r"].round(2)
    grouped["N"] = grouped["N"].round(2)
    return grouped

def add_environments(bonds_df, atoms_df, env_size: int = 5, cutoff_ang: float = 2.0):
    """
    Add a simple local chemical environment description around each bond endpoint.

    For each bond (atom1_id - atom2_id):
      - For atom1_id, look for neighboring atoms within cutoff_ang (Å), excluding atom2_id
      - For atom2_id, look for neighboring atoms within cutoff_ang (Å), excluding atom1_id
      - Convert each neighbor into a pair label "XY" where X is center element and Y is
        neighbor element (sorted alphabetically), e.g.:
          center=C neighbor=H -> "CH"
          center=O neighbor=H -> "HO"
      - Exclude "HH" labels
      - Sort labels deterministically and keep up to env_size labels
      - Pad with NaN if fewer than env_size neighbors exist

    Outputs columns:
      a_env1..a_env{env_size} for atom1 environment
      b_env1..b_env{env_size} for atom2 environment

    Args:
        bonds_df (pd.DataFrame): Bond-level DataFrame with atom1_id and atom2_id columns.
        atoms_df (pd.DataFrame): Atom DataFrame with atom_id, element, X,Y,Z in bohr.
        env_size (int): Number of environment labels to store per bond end.
        cutoff_ang (float): Neighbor cutoff radius in Å.

    Returns:
        pd.DataFrame: Copy of bonds_df with added environment columns.
    """
    id_to_elem = atoms_df.set_index("atom_id")["element"].to_dict()
    coords = atoms_df.set_index("atom_id")[["X", "Y", "Z"]].to_dict("index")

    def dist_angstrom(id1: int, id2: int) -> float:
        c1 = coords[id1]; c2 = coords[id2]
        dx = c1["X"] - c2["X"]; dy = c1["Y"] - c2["Y"]; dz = c1["Z"] - c2["Z"]
        return (dx*dx + dy*dy + dz*dz) ** 0.5 * BOHR_TO_ANGSTROM

    all_ids = set(atoms_df["atom_id"].astype(int).tolist())

    def env_for(center_id: int, exclude_id: int):
        center_elem = id_to_elem[center_id]
        pairs = []
        for nid in all_ids:
            if nid == center_id or nid == exclude_id:
                continue
            if dist_angstrom(center_id, nid) < cutoff_ang:
                neigh_elem = id_to_elem[nid]
                pair_label = "".join(sorted([center_elem, neigh_elem]))  # e.g., CH, HO, CC
                if pair_label == "HH":
                    continue
                pairs.append((pair_label, nid))

        pairs.sort(key=lambda t: (t[0], t[1]))

        labels = [p for (p, _) in pairs[:env_size]]
        while len(labels) < env_size:
            labels.append(np.nan)
        return labels

    a_env_cols = [[] for _ in range(env_size)]
    b_env_cols = [[] for _ in range(env_size)]

    for _, row in bonds_df.iterrows():
        a = int(row["atom1_id"]); b = int(row["atom2_id"])
        a_labels = env_for(a, b)
        b_labels = env_for(b, a)
        for i in range(env_size):
            a_env_cols[i].append(a_labels[i])
            b_env_cols[i].append(b_labels[i])

    out = bonds_df.copy()
    for i in range(env_size):
        k = i + 1
        out[f"a_env{k}"] = a_env_cols[i]
        out[f"b_env{k}"] = b_env_cols[i]

    return out

if __name__ == "__main__":
    results = pd.DataFrame()

    # Find all *.top files
    files = glob.glob(f"dft/*.top")
    files = [os.path.splitext(os.path.basename(file))[0] for file in files]

    # Parse each *.top file
    for file_name in files:
        print(f"Parsing filename {file_name}...")

        file_top = f"dft/{file_name}.top"
        file_wfn = f"dft/{file_name}.wfn"

        with open(file_top, "r") as file:
            lines = file.readlines()

        keyword1 = "elapsed time in attractor assignment"
        keyword2 = "basin               vol.    pop.    pab     paa    pbb     sigma2  std. dev."

        table1_lines, table2_lines = extract_tables(lines, keyword1, keyword2)
        atoms_df = read_wfn_atoms(file_wfn)

        df = create_dataframe(table1_lines, table2_lines)
        df = find_two_min_distances(df, atoms_df)
        df = calculate_distance_between_atoms(df, atoms_df)
        df = merge_attractors(df)
        df = add_environments(df, atoms_df)

        df["mol_id"] = file_name
        results = pd.concat((results, df), ignore_index=True)

    # Save entire dataset
    results.to_csv("data/datasets.csv", index=False)
