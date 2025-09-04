import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import product
import pandas as pd

#############################
# Define Master ODE System  #
#############################
def dPdt(t, P):
    dP = np.zeros_like(P)
    for i in range(2):  # state: 0 or 1
        for m1 in range(m1_max + 1):
            for m2 in range(m2_max + 1):
                u = idx(i, m1, m2)
                val = P[u]

                # LOSS: extinction + state-change + birth
                loss = (mu[i] + sum(q[i] if i != j else 0 for j in range(2)) +
                        sum(lam[i][j][k] for j in range(2) for k in range(2)))
                dP[u] -= loss * val

                # GAIN: from state change
                for j in range(2):
                    if j != i:
                        dP[u] += q[j] * P[idx(j, m1, m2)]

                # GAIN: from extinction (if all 0)
                if m1 == 0 and m2 == 0:
                    dP[u] += mu[i] * val

                # GAIN: from splitting
                for j in range(2):
                    for k in range(2):
                        lamx = lam[i][j][k]
                        for n1 in range(m1 + 1):
                            for n2 in range(m2 + 1):
                                if 0 <= n1 <= m1 and 0 <= n2 <= m2:
                                    u1 = idx(j, n1, n2)
                                    u2 = idx(k, m1 - n1, m2 - n2)
                                    dP[u] += lamx * P[u1] * P[u2]
    return dP

#####################
# Observed Clade    #
#####################
AsS = 1  # ancestral state (1-based)
obs_m1 = 0
obs_m2 = 43
T = 5.5

##############
# Parameters #
##############

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


# Max lineages to evaluate
m1_max = 75
m2_max = m1_max

# Discretized state space : index mapping
def idx(i, m1, m2):
    return i * (m1_max + 1) * (m2_max + 1) + m1 * (m2_max + 1) + m2

# Reverse map
def reverse_idx(index):
    i = index // ((m1_max + 1) * (m2_max + 1))
    rem = index % ((m1_max + 1) * (m2_max + 1))
    m1 = rem // (m2_max + 1)
    m2 = rem % (m2_max + 1)
    return i, m1, m2

# Total number of ODEs
N = 2 * (m1_max + 1) * (m2_max + 1)

f1_list=[0.1,0.5,0.9]

for i in range(3):
    #####################
    # Initial Condition #
    #####################
    f1 = f1_list[i]
    f2 = 1-f1
    f = [f1,f2]
    P0 = np.zeros(N)
    P0[idx(0, 0, 0)] = (1 - rho[0]) * f[0]
    P0[idx(1, 0, 0)] = (1 - rho[1]) * f[1]
    P0[idx(0, 1, 0)] = rho[0] * f[0]
    P0[idx(0, 0, 1)] = rho[1] * f[1]

    #############################
    # Integrate from t = 0 to T #
    #############################
    sol = solve_ivp(dPdt, [0, T], P0, method='RK45')

    # Final distribution
    P_final = sol.y[:, -1]

    # Reshape to extract matrix for ancestral state
    P0_matrix = P_final[: (m1_max + 1) * (m2_max + 1)].reshape((m1_max + 1, m2_max + 1))
    P1_matrix = P_final[(m1_max + 1) * (m2_max + 1):].reshape((m1_max + 1, m2_max + 1))

    # Choose correct ancestral state (convert 1-based AsS to 0-based index)
    Pm = P0_matrix if AsS == 1 else P1_matrix

    ###########################
    # Compute one-sided p-val #
    ###########################
    p_value = 1 - np.sum(Pm[n1,n2] for n1 in range(obs_m1) for n2 in range(obs_m2))
    print(f"P-value for observed (m1={obs_m1}, m2={obs_m2}) = {p_value:.8f}")


    ##############
    # Save Pm    #
    ##############
    file_name = f"/home/user/f0_{f1}&f1_{f2}.csv"
    df = pd.DataFrame(Pm)
    df.to_csv(file_name, sep='\t')

#####################
# Optional Heatmap  #
#####################

plt.figure(figsize=(10, 8))
plt.imshow(np.log(Pm + 1e-50), origin='lower', aspect='auto',
        extent=[0, m2_max, 0, m1_max], cmap='viridis')
plt.colorbar(label='log(Pi,n1,n2)')
plt.scatter(obs_m2, obs_m1, color='red', label='Observed (n1, n2)')
plt.xlabel("# State 2 lineages")
plt.ylabel("# State 1 lineages")
plt.title("Joint PMF: Lineage Counts")
plt.legend()
plt.tight_layout()
plt.show()
