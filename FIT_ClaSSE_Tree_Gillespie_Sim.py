"""
Birth-Death Cladogenetic Tree Simulator (Forward-in-Time)
=========================================================

Started     : April 10, 2025  
Last Update : May 17, 2025  
Version     : V3 (Cladogenic changes supported)

This script simulates the evolution of lineages under a birth-death process with two states and
cladogenetic state changes. It also provides tree visualization and exports the final tree in 
Newick format and as a distance matrix for further phylogenetic analyses.

Dependencies:
-------------
- numpy
- matplotlib
- seaborn
- pandas
- scipy

Outputs:
--------
- Time-stamped tree matrix
- Observed lineage states
- Newick formatted tree
- Ancestor state transitions

Author:
-------
Erwan H., MSc student in Rennes University (Rennes, France)
"""

# ========== Libraries ==========
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as clust
import json

# For reproducibility
np.random.seed(8980)

# ========== Functions ==========

def choose_event(s, lam1, mu1, lam2, mu2, q12, q21):
    """
    Choose the next event (speciation, extinction, transition) and lineage index.

    Parameters:
    -----------
    s : list
        Current states of all lineages.
    lam1, lam2 : float
        Speciation rates for states 1 and 2.
    mu1, mu2 : float
        Extinction rates.
    q12, q21 : float
        Transition rates between states.

    Returns:
    --------
    event : tuple
        Selected event and lineage index.
    """
    events = []
    rates = []

    for lineage, state in enumerate(s):
        if state == 1:
            events += [("speciation", lineage), ("extinction", lineage), ("transition_to_2", lineage)]
            rates  += [lam1, mu1, q12]
        elif state == 2:
            events += [("speciation", lineage), ("extinction", lineage), ("transition_to_1", lineage)]
            rates  += [lam2, mu2, q21]

    total = sum(rates)
    if total == 0:
        return None, 0

    probs = [r / total for r in rates]
    idx = np.random.choice(len(events), p=probs)

    return events[idx]

def apply_event_on_s(s, choosed_event, lam):
    """
    Apply the chosen event to the state list `s`.

    Parameters:
    -----------
    s : list
        List of lineage states.
    choosed_event : tuple
        Selected event and index.
    lam : np.ndarray
        Speciation transition tensor.

    Returns:
    --------
    s : list
        Updated states list.
    """
    event, i = choosed_event

    if event == "speciation":
        lami = sum(lam[s[i]-1][j][k] for j in range(2) for k in range(2))
        probs = [lam[s[i]-1][0][0]/lami, 2*lam[s[i]-1][0][1]/lami, lam[s[i]-1][1][1]/lami]
        clad_options = [[1,1], [1,2], [2,2]]
        clad_index = np.random.choice(len(clad_options), p=probs)
        clad = clad_options[clad_index]

        rd = np.random.choice([0,1], p=[0.5,0.5])
        s[i] = clad[rd]
        s.append(clad[-rd])
    elif event == "extinction":
        s[i] = 0
    elif event == "transition_to_2":
        s[i] = 2
    elif event == "transition_to_1":
        s[i] = 1

    return s

def apply_event_on_M(M, choosed_event, t):
    """
    Apply the event to the distance matrix M.

    Parameters:
    -----------
    M : np.ndarray
        Shared time matrix.
    choosed_event : tuple
        Selected event and index.
    t : float
        Current simulation time.

    Returns:
    --------
    M : np.ndarray
        Updated matrix.
    """
    event, i = choosed_event
    M = np.asarray(M)
    
    if event != "speciation":
        np.fill_diagonal(M, t)
        return M

    n = M.shape[0]
    New_M = np.zeros((n + 1, n + 1))
    New_M[:n, :n] = M
    New_M[n, :n] = M[i, :]
    New_M[:n, n] = M[:, i]
    New_M[n, i] = New_M[i, n] = t
    np.fill_diagonal(New_M, t)

    return New_M

# ========== Initialization ==========
s = [1]
M = np.full((1, 1), 0)

# Simulation parameters
TMax = 5.5

lam1 = 0.5
lam2 = 1
q12 = 1
q21 = 0.1

mu1 = 0.045
mu2 = 0.045

rho1 = 0.75
rho2 = 0.75

# Compute cladogenetic rates
lam1_11 = lam1*(1-q12)**2
lam1_21 = lam1*q12
lam1_22 = lam1*q12**2
lam2_22 = lam2*(1-q21)**2
lam2_21 = lam2*q21
lam2_11 = lam2*q21**2

lam = np.zeros((2, 2, 2))
lam[0][0][0] = lam1_11
lam[0][1][0] = lam[0][0][1] = lam1_21
lam[0][1][1] = lam1_22
lam[1][0][0] = lam2_11
lam[1][1][0] = lam[1][0][1] = lam2_21
lam[1][1][1] = lam2_22

# ========== Simulation ==========
times = [0]
n_state1 = [s.count(1)]
n_state2 = [s.count(2)]
n_extinct = [s.count(0)]

t = 0
AnceState = {}

while t < TMax and (1 in set(s) or 2 in set(s)):
    lam_tot = sum((lam1 + mu1 + q12) if state == 1 else (lam2 + mu2 + q21) if state == 2 else 0 for state in s)
    tau = np.random.exponential(1 / lam_tot)
    t += tau

    next_event = choose_event(s, lam1, mu1, lam2, mu2, q12, q21)
    event, i = next_event
    if event == 'speciation':
        AnceState[t] = s[i] - 1

    s = apply_event_on_s(s, next_event, lam)
    M = apply_event_on_M(M, next_event, t)

    times.append(t)
    n_state1.append(s.count(1))
    n_state2.append(s.count(2))
    n_extinct.append(s.count(0))

# ========== Sampling ==========
states = [0, 1, 2]
sampS = [
    np.random.choice(states, p=[1 - rho1, rho1, 0]) if x == 1
    else np.random.choice(states, p=[1 - rho2, 0, rho2]) if x == 2
    else 0
    for x in s
]

# Remove unsampled
index_to_remove = [i for i, x in enumerate(sampS) if x == 0]
obsM2 = np.delete(M, index_to_remove, axis=0)
obsM2 = np.delete(obsM2, index_to_remove, axis=1)
obsS = np.delete(s, index_to_remove)
obsS = np.asarray([0 if s[i] == 1 else 1 for i in range(len(obsS))])

# ========== Visualization ==========
# Lineages over time
plt.plot(times, n_state1, '-g', label='State 1', alpha=0.8)
plt.plot(times, n_state2, '-b', label='State 2', alpha=0.8)
plt.plot(times, n_extinct, '-r', label='Extinct', alpha=0.8)
plt.grid()
plt.legend(loc='best')
plt.title("Number of Lineages Over Time")
plt.xlabel("Time")
plt.ylabel("Count")
plt.show()

# Tree clustering
M_dist = abs(obsM2 - t)
condensed = squareform(M_dist)
linkage_matrix = linkage(condensed, method='complete')

fig, ax = plt.subplots()
tree = clust.dendrogram(linkage_matrix, ax=ax, color_threshold=0, link_color_func=lambda k: 'black')
leaf_order = tree['leaves']
for i, leaf_idx in enumerate(leaf_order):
    x = 5 + i * 10
    y = 0.07
    color = 'red' if obsS[leaf_idx] == 0 else 'green'
    ax.plot(x, y, 'o', color=color, markersize=8)
plt.xlabel("Time")
plt.ylabel("Lineages")
plt.title("Simulated Phylogenetic Tree")
plt.show()

# ========== Export ==========
np.savetxt("~/shared_time_matrix.txt", obsM2, delimiter=",")
np.savetxt("~/lineages.txt", obsS.astype(int), fmt="%d")

# Ancestor state tracking
with open("~/AnceState.json", "w") as f:
    json.dump(AnceState, f)

# ========== Newick Conversion ==========
def get_newick(node, newick="", parentdist=0, leaf_names=None):
    """
    Recursively convert a scipy ClusterNode to a Newick formatted string.

    Parameters:
    -----------
    node : ClusterNode
    newick : str
        Current state of the string.
    parentdist : float
        Distance from parent node.
    leaf_names : list
        Names of leaves.

    Returns:
    --------
    str : Newick formatted string.
    """
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{parentdist - node.dist:.6f}{newick}"
    else:
        if newick:
            newick = f"):{parentdist - node.dist:.6f}{newick}"
        else:
            newick = ");"
        left = get_newick(node.get_left(), newick, node.dist, leaf_names)
        right = get_newick(node.get_right(), f",{left}", node.dist, leaf_names)
        return f"({right}"

# To Newick
obsM2 = np.loadtxt("~/shared_time_matrix.txt", delimiter=",")
obsS = np.loadtxt("~/lineages.txt", dtype=int)
M_dist = np.max(obsM2) - obsM2
condensed = squareform(M_dist)
linkage_matrix = linkage(condensed, method='complete')
tree, _ = to_tree(linkage_matrix, rd=True)
leaf_names = [f"t{i}_s{state}" for i, state in enumerate(obsS)]
newick_str = get_newick(tree, "", tree.dist, leaf_names)

# Save Newick string
with open("~/tree.nwk", "w") as f:
    f.write(newick_str + "\n")
