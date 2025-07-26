#######################
# Required libraries  #
#######################
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import json

#####################
# Define parameters #
#####################
##############
# Parameters #
##############
TMax = 5.5

lam1 = 0.5
lam2 = 1

q12 = 1
q21 = 0.1

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

mu1 = 0.045
mu2 = 0.045

rho1 = 0.75
rho2 = 0.75

mu = [mu1, mu2]  # extinction rates
q = [q12, q21]    # anagenetic transition rates
rho = [rho1, rho2]  # sampling probabilities
pi = [0.5, 0.5]   # root state prior

###################################
# Import tree matrix and lineages #
###################################
M = np.loadtxt("/home/r1/Documents/Master/M1/STAGE/DATA/shared_time_matrix.txt", delimiter=",")
s = np.loadtxt("/home/r1/Documents/Master/M1/STAGE/DATA/lineages.txt", delimiter=",", dtype=int)

with open("/home/r1/Documents/Master/M1/STAGE/DATA/AnceState.json", "r") as f:
    AnceState = json.load(f)


labels = list(range(len(s)))
df = pd.DataFrame(M, index=labels, columns=labels)
df2 = df.__deepcopy__()
##########################################
#     Get existing nodes in the tree     #
##########################################

def get_nodes(df):
    SharedTimes = sorted(np.unique(df.values), reverse=True)
    T = df.values.max()
    step = 0
    nodes = {}
    while len(df) > 1:
        step += 1
        tau = T - SharedTimes[step]

        tri_sup = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        i_label, j_label = tri_sup.stack().idxmax()
        i_pos = df.index.get_loc(i_label)
        j_pos = df.index.get_loc(j_label)

        to_rename, to_remove = (i_label, j_label) if i_pos < j_pos else (j_label, i_label)
        new_label = f"{i_label},{j_label}"

        nodes[new_label] = tau
    
        np.fill_diagonal(df.values, tau)
        df = df.drop(index=to_remove, columns=to_remove)
        df = df.rename(index={to_rename: new_label}, columns={to_rename: new_label})
    return nodes

##########################################
#       Get parent node of a node        #
##########################################

def parent_node(node, nodes):
    tips = set(node.split(','))
    nodes1 = nodes.copy()
    nodes = [*nodes]
    
    if node == max([n for n in nodes if tips.issubset(set(n.split(',')))], key=lambda x: len(x.split(','))) :
        print("No parent node for the root")
        return node
    
    else:
        parent = min([n for n in nodes if tips.issubset(set(n.split(','))) and tips != set(n.split(','))], key=lambda x: len(x.split(',')))

    return parent

def parent_node_time(node, nodes, T):
    tips = set(node.split(','))
    nodes1 = nodes.copy()
    nodes = [*nodes]
    
    if node == max([n for n in nodes if tips.issubset(set(n.split(',')))], key=lambda x: len(x.split(','))) :
        time = T
    
    else:
        time = nodes1[min([n for n in nodes if tips.issubset(set(n.split(','))) and tips != set(n.split(','))], key=lambda x: len(x.split(',')))]

    return time

##########################################
#       Get sister edge of a edge        #
##########################################

def get_sister_edge(node, parent, nodes):
    """
    Given a node and its parent, return the sister node (i.e., the other child of the parent).
    
    Parameters:
    - node (str): The name of the current node, with tip labels separated by commas (e.g., "A,B").
    - parent (str): The name of the parent node, also with tip labels separated by commas.
    - nodes (list of str): A list of all node names in the tree.

    Returns:
    - str: The sister node name
    """
    node_set = set(node.split(','))
    parent_set = set(parent.split(','))
    sister_set = parent_set - node_set
    #print("Sister clade tip set:", sister_set)

    # Try to find the node in `nodes` that exactly matches the sister set
    for candidate in nodes:
        candidate_set = set(candidate.split(','))
        if candidate_set == sister_set:
            return candidate

    return int(sister_set.pop())

######################################################################################################
######################################################################################################

#                                             Step 1                                                 #

######################################################################################################
######################################################################################################

##########################################
# ODE System for E and D (backward-time) #
##########################################
def dEdt_back(t, y, lam, mu, q):
    E = y[0:2]
    dE = [0.0, 0.0]

    for i in range(2):
        lam_sum = sum(lam[i][j][k] for j in range(2) for k in range(2))
        q_sum = q[i]
        mu_i = mu[i]

        cladExt_term = sum(lam[i][j][k] * E[j] * E[k] for j in range(2) for k in range(2))
        shift_term = q[i] * E[1 - i]
        dE[i] = mu_i - (lam_sum + q_sum + mu_i) * E[i] + shift_term + cladExt_term

    return dE

def ode_backward(t, y, lam, mu, q):
    E = y[0:2]
    D = y[2:4]
    dE = [0.0, 0.0]
    dD = [0.0, 0.0]

    for i in range(2):
        lam_sum = sum(lam[i][j][k] for j in range(2) for k in range(2))
        q_sum = q[i] if i == 0 else q[1]
        mu_i = mu[i]

        cladExt_term = sum(lam[i][j][k] * E[j] * E[k] for j in range(2) for k in range(2))
        shift_term = q[i] * E[1 - i]
        dE[i] = mu_i - (lam_sum + q_sum + mu_i) * E[i] + shift_term + cladExt_term

        shift_term_D = q[i] * D[1 - i]
        hiddenSpec_term_D = D_init = sum(2 * lam[i][j][k] * D[j] * E[k] for j in range(2) for k in range(2))
        dD[i] = -(lam_sum + q_sum + mu_i) * D[i] + shift_term_D + hiddenSpec_term_D

    return dE + dD


def integrate_backward(tau, edgeLen, D_left, D_right, lam, mu, q, rho):
    E_init = [1 - rho[0], 1 - rho[1]]
    y0 = E_init

    solE = solve_ivp(
        fun=lambda t, y: dEdt_back(t, y, lam, mu, q),
        t_span=(0, tau),
        y0=y0,
        method='RK45',
        t_eval=np.linspace(0, tau, 100)
    )

    E_initTau = list(solE.y[:,-1])
    #print(f"Einit = {E_initTau}")
    D_init = [sum(2*lam[i][j][k] * D_left[j] * D_right[k] for j in range(2) for k in range(2)) for i in range(2)]
    y0tau = E_initTau + D_init

    sol = solve_ivp(
        fun=lambda t, y: ode_backward(t, y, lam, mu, q),
        t_span=(0, edgeLen),
        y0=y0tau,
        method='RK45',
        t_eval=np.linspace(0, edgeLen, 100)
    )

    return list(sol.y[:, -1]), D_init

#########################################
# Post-order traversal implementation   #
#########################################

Dei = {}
SharedTimes = sorted(np.unique(df.values), reverse=True)
T = df.values.max()
df_diag0 = df.copy()


# Initialization from tips to 1st node
for tip in range(len(labels)):
    np.fill_diagonal(df_diag0.values, 0)
    times = df_diag0.loc[tip]
    node_time = T - times[times != 0].max()
    #print(f"node time = {node_time}")

    # Initial values
    state = s[tip]
    E0 = [1 - rho[0], 1 - rho[1]]
    D0 = [rho[i] if i == state else 0.0 for i in range(2)]
    y0 = E0 + D0
    #print(f"y0 = {y0}")

    # Solve along the edge
    sol = solve_ivp(
        fun=lambda t, y: ode_backward(t, y, lam, mu, q),
        t_span=(0, node_time),
        y0=y0,
        method='RK45',
        t_eval=np.linspace(0, node_time, 100)
    )
    #print(sol.y[2:, -1])

    # Store initial values and solution
    Dei[labels[tip]] = {
        "Tau": sol.y[2:, -1],
        "Tau0": D0
    }


# Tree traversal
nodes = get_nodes(df)
#print([*nodes])
step = 0
while len(df) > 1:
    step += 1
    tau = T - SharedTimes[step]
    #print(f"\nTau : {tau}")

    tri_sup = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    i_label, j_label = tri_sup.stack().idxmax()
    i_pos = df.index.get_loc(i_label)
    j_pos = df.index.get_loc(j_label)

    to_rename, to_remove = (i_label, j_label) if i_pos < j_pos else (j_label, i_label)
    new_label = f"{i_label},{j_label}"

    # Get parent node time
    parenTime = parent_node_time(new_label,nodes,T)
    
    # Get edge length (from node to parent node)
    edgeLen = parenTime - tau
    #print(edgeLen)
   
    # Get children values (closer to the node)
    D_left = Dei[i_label]["Tau"]
    D_right = Dei[j_label]["Tau"]

    # Solve E and D + derivatives
    result, dD0 = integrate_backward(tau, edgeLen, D_left, D_right, lam, mu, q, rho)

    Dei[new_label] = {
        "Tau": result[2:],
        "Tau0": dD0
    }

    #print(f"[{new_label}] D = {Dei[new_label]['Tau']}")

    np.fill_diagonal(df.values, SharedTimes)
    df = df.drop(index=to_remove, columns=to_remove)
    df = df.rename(index={to_rename: new_label}, columns={to_rename: new_label})

print(f"\nTraversal complete with {step} internal nodes")
#print(Dei)
#print(len([*Dei]))

######################################################################################################
######################################################################################################

#                                             Step 2                                                 #

######################################################################################################
######################################################################################################

### Compute root state
keys = [*Dei] # Unpacking key
root_key = keys[-1]
root_prob_state0 = pi[0] * Dei[root_key]['Tau0'][0]/(pi[0] * Dei[root_key]['Tau0'][0]+ pi[1] * Dei[root_key]['Tau0'][1])
root_prob_state1 = 1-root_prob_state0
print(f"\n The probablity that the root is in state 0 is {round(root_prob_state0,5)}")

######################################################################################################
######################################################################################################

#                                             Step 3                                                 #

######################################################################################################
######################################################################################################
def dDC_forw(t, y, lam, mu, q, E, T, t0e):
    D = y[0:2]
    dD = [0.0, 0.0]

    for i in range(2):
        i_to_j = sum(lam[i][i-1][k]*D[i]*E[k](T-(t+t0e))+ q[i]*D[i] for k in range(2))
        j_to_i = sum(lam[i-1][i][k]*D[i-1]*E[k](T-(t+t0e))+ q[i-1]*D[i-1] for k in range (2))
        dD[i] = - (i_to_j) + (j_to_i)

    return dD

def solve_Dc(t, edgeLen, Dinit, lam, mu, q, rho, T, t0e):
    E_init = [1 - rho[0], 1 - rho[1]]
    y0 = E_init

    solE = solve_ivp(
        fun=lambda t, y: dEdt_back(t, y, lam, mu, q),
        t_span=(0, T),
        y0=y0,
        method='RK45',
        t_eval=np.linspace(0, T, 100))

    E_tau_val = solE.y
    E_interp = [interp1d(solE.t, E_tau_val[state], kind='linear', fill_value='extrapolate') for state in range(2)]

    solD = solve_ivp(
        fun=lambda t, y: dDC_forw(t, y, lam, mu, q, E_interp, T, t0e),
        t_span=(0, edgeLen),
        y0=Dinit,
        method='RK45',
        t_eval=np.linspace(0, edgeLen, 100))

    return list(solD.y[:, -1])

##################
# Tree traversal #
##################
DC_ei = {root_key : {"t" : [root_prob_state0,root_prob_state1]}}

SharedTimes = sorted(np.unique(df2.values))
nodesT = dict((round(T-v,10),k) for k,v in get_nodes(df2).items())

#print(nodesT)

for t in SharedTimes[1:-1]:  # Skip the maxtime (diag)
    
    t = round(t,10)
    node = nodesT[t]
    #print(f"node at time {t} is {node}")

    parent = parent_node(node,nodes)
    #print(parent)

    sister_edge = get_sister_edge(node, parent, nodes)
    #print(sister_edge)
    
    t0e = parent_node_time(node,nodes,T)
    edgeLen = abs(t-t0e)

    De_prim = Dei[sister_edge]['Tau']
    Dc_p = DC_ei[parent]['t']

    DinitNum = [sum(2*lam[j][i][k]*Dc_p[j]*De_prim[k] for j in range(2) for k in range(2)) for i in range(2)]
    DinitDen = sum(2*lam[j][l][k]*Dc_p[j]*De_prim[k] for j in range(2) for l in range(2) for k in range(2))
    Dinit = [DinitNum[0]/DinitDen,DinitNum[1]/DinitDen]

    solD = solve_Dc(t,edgeLen,Dinit,lam,mu,q,rho,T,t0e)

    DC_ei[node] = {'t0e':Dinit, 't': solD} 

    #print(DC_ei[node])
#print(DC_ei)

######################################################################################################
######################################################################################################

#                                             Step 4                                                 #

######################################################################################################
######################################################################################################
As = {}
for node in DC_ei.keys():
    if node == root_key:
        As[root_key] = [root_prob_state0,root_prob_state1]
    else :
        prob_state0 = (DC_ei[node]['t'][0]*Dei[node]['Tau0'][0])/(DC_ei[node]['t'][0]*Dei[node]['Tau0'][0] + DC_ei[node]['t'][1]*Dei[node]['Tau0'][1])
        As[node] = [prob_state0,1-prob_state0]
    #print(f"node {node} was in state 0 with prob {As[node][0]}")

print(As)
print(f"\nPre-order traversal completed")
##################################################################################
#print(nodesT)
sampled_nodes_times = [*nodesT]
AnceState_rounded = dict((round(float(time), 10), state) for time, state in AnceState.items())
#print(AnceState_rounded)
AnceState_sampled = {
    time: state
    for time, state in AnceState_rounded.items()
    if time in sampled_nodes_times
}
print(AnceState_sampled)

df_AncesState = pd.DataFrame(list(AnceState_sampled.items()), columns=["time", "state"])
df_AncesState = df_AncesState.sort_values(by="time")
n = len(s)
df_AncesState["node"] = list(range(n + 1, 2 * n))

df_AncesState = pd.concat([df_AncesState,pd.DataFrame(As.values(), columns=["Pstate_0", "Pstate_1"])], axis=1)

print(df_AncesState)
df_AncesState.to_csv("/home/r1/Documents/Master/M1/STAGE/DATA/AnceState_final.csv", index=False)

#################################################################################################
