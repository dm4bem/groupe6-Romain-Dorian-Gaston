import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import dm4bem

# Dimensions de la pièce
L, L2, H = 2.5, 2, 3  # Longueur et hauteur de la pièce (m)
S_wall1 = L * H
S_wall2 = L2 * H         # Surface de chaque mur vertical
S_wall=S_wall1+S_wall2
S_floor = S_roof = L**2 # Surface sol et plafond
V_air = L**2 * H        # Volume intérieur

# Données absorbtion
E = 300 # solar irradiation; in W/m^2
alphawout=0.25
alphawindow=0.38
alpharoof=alphafloor=0.2

#Température
To, Ti = 10, 22

# Matériaux
air = {'Density': 1.2, 'Specific heat': 1000}

concrete = {'Conductivity': 1.4,
            'Density': 2300,
            'Specific heat': 880,
            'Width': 0.24,
            'Surface': 3 * S_wall}

insulation = {'Conductivity': 0.027,
              'Density': 55.0,
              'Specific heat': 1210,
              'Width': 0.06,
              'Surface': 3 * S_wall}

glass = {'Conductivity': 1.4,
         'Density': 2500,
         'Specific heat': 1210,
         'Width': 0.04,
         'Surface': S_wall}  

floor = {'Conductivity': 0.15,
         'Density': 700,
         'Specific heat': 1500,
         'Width': 0.02,
         'Surface': S_floor}

roof = {'Conductivity': 0.25,
        'Density': 850,
        'Specific heat': 1090,
        'Width': 0.013,
        'Surface': S_roof}

# Compilation dans le DataFrame
wall = pd.DataFrame.from_dict({'Concrete': concrete,
                               'Insulation': insulation,
                               'Glass': glass,
                               'Floor': floor,
                               'Roof': roof},
                              orient='index')

# Coefficients radiatifs & convectifs
ε_wLW, ε_gLW = 0.85, 0.90
σ = 5.67e-8



# Facteur de vue mur vers vitrage
Fwg = glass['Surface'] / concrete['Surface']

# Échange radiatif LW
Tm = 273 + 10
GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * concrete['Surface']
GLW12 = 4 * σ * Tm**3 * Fwg * concrete['Surface']
GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * glass['Surface']
GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

# Ventilation
ACH = 1
Va_dot = ACH / 3600 * V_air
Gv = air['Density'] * air['Specific heat'] * Va_dot

# P-contrôleur désactivé
Kp = 0

# Capacité thermique totale
C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
C['Air'] = air['Density'] * air['Specific heat'] * V_air

# Températures nœuds
θ = [f'θ{i}' for i in range(19)]
q = [f'q{i}' for i in range(29)]  

hi, ho = 8., 25.
λ_c, w_c = concrete['Conductivity'], concrete['Width']
λ_i, w_i = insulation['Conductivity'], insulation['Width']
λ_g, w_g = glass['Conductivity'], glass['Width']
λ_f, w_f = floor['Conductivity'], floor['Width']
λ_r, w_r = roof['Conductivity'], roof['Width']
S1, S2, S3 = S_wall1, S_wall2, S_wall1
S_floor = floor['Surface']
S_roof = roof['Surface']
S_door = S1-S2
λ_door = 0.22
w_door = 0.04
S_wind = glass['Surface']

# Matrice d'incidence A 
A=np.zeros([29, 19])
A[0, 0] = 1
A[1, 0] = -1
A[1, 1] = 1
A[2, 1] = -1
A[2, 2] = 1
A[3, 3] = 1
A[4, 3] = -1
A[4, 4] = 1
A[5, 4] = -1
A[5, 5] = 1
A[6, 5] = -1
A[6, 6] = 1
A[7, 6] = -1
A[7, 2] = 1
A[8, 7] = 1
A[9, 7] = -1
A[9, 8] = 1
A[10, 8] = -1
A[10, 9] = 1
A[11, 9] = -1
A[11, 2] = 1
A[12, 9] = -1
A[12, 6] = 1
A[13, 9] = -1
A[13, 1] = 1
A[14, 10] = 1
A[15, 10] = -1
A[15, 11] = 1
A[16, 11] = -1
A[16, 12] = 1
A[17, 12] = -1
A[17, 2] = 1
A[18, 12] = -1
A[18, 6] = 1
A[19, 12] = -1
A[19, 1] = 1
A[20, 2] = 1
A[21, 13] = 1
A[22, 13] = -1
A[22, 14] = 1
A[23, 14] = -1
A[23, 15] = 1
A[24, 15] = -1
A[24, 2] = 1
A[25, 16] = 1
A[26, 16] = -1
A[26, 17] = 1
A[27, 17] = -1
A[27, 18] = 1
A[28, 18] = -1
A[28, 2] = 1

pd.DataFrame(A, index=q, columns=θ)

G = np.hstack([
    # Fenêtre
    hi * S_wind,
    (λ_g / (2 * w_g)) * S_wind,
    ho * S_wind,
    
    # Mur 1
    hi * S1,
    (λ_c / (2 * w_c)) * S1,
    (λ_i / w_i) * S1,
    (λ_c / (2 * w_c)) * S1,
    ho * S1,

    # Mur 2
    hi * S2,
    (λ_c / (2 * w_c)) * S2,
    (λ_i / w_i) * S2,
    ho * S2,

    # Rayonnement
    GLW, GLW, GLW, GLW,

    # Mur 3
    hi * S3,
    (λ_c / (2 * w_c)) * S3,
    (λ_i / w_i) * S3,
    ho * S3,

    # Porte
    (2 * hi + λ_door / w_door) * S_door,

    # Sol
    hi * S_floor,
    (λ_f / (2 * w_f)) * S_floor,
    (λ_f / (2 * w_f)) * S_floor,
    ho * S_floor,

    # Plafond
    hi * S_roof,
    (λ_r / (2 * w_r)) * S_roof,
    (λ_r / (2 * w_r)) * S_roof,
    ho * S_roof
])
# print("Taille de G :", G.shape)
# print("Taille attendue :", len(q))

pd.DataFrame(G, index=q)

# Capacité thermique par nœud 

neglect_air_glass = True

if neglect_air_glass:
    C = np.array([0, concrete['Density'] * concrete['Specific heat'] * concrete['Surface'] * concrete['Width'], 0, insulation['Density'] * insulation['Specific heat'] * insulation['Surface'] * insulation['Width'], 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
else:
    C = np.array([0, concrete['Density'] * concrete['Specific heat'] * concrete['Surface'] * concrete['Width'], 0, insulation['Density'] * insulation['Specific heat'] * insulation['Surface'] * insulation['Width'], 0, 0, glass['Density'] * glass['Specific heat'] * glass['Surface'] * glass['Width'], 0 ,0, 0, 0, 0, air['Density'] * air['Specific heat'] * V_air, 0, 0, 0, 0, 0, 0, 0])
pd.DataFrame(C, index=θ)

# Sources de température (b) et flux (f)
b = pd.Series(['To', 0, 0, 'To', 0, 0, 0, 0, 'Ti', 0, 0, 0, 0, 0, 'Ti', 0, 0, 0, 0, 0, 'Ti', 'Ti',0,0,0,'Ti',0,0,0],
              index=q)

f = pd.Series(['Φ1', 0, 0, 'Φ2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 'Φ3', 0, 0, 'Φ4', 0, 0],
              index=θ)


# Température d’intérêt : air intérieur
y = np.zeros(19)            # nodes
y[[2]] = 1              # nodes (temperatures) of interest 

pd.DataFrame(y, index=θ)

# Création du modèle thermique complet
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

A_np = A.values            # (29, 19)
G_mat = np.diag(G.values)  # (29, 29) matrice diagonale à partir de G (vecteur)
b_np = b.values            # (29,)
f_np = f.values            # (19,)


bss = np.zeros(29)        # temperature sources b for steady state
bss[[0, 3]] = 10     # outdoor temperature To : 10 °C
bss[[8, 14, 20, 21, 25]] = 22         # indoor temperature Ti: 22 °C

fss = np.zeros(19)        # flow-rate sources f for steady state


diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)
θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
# θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ b + f)

print(f'θss = {np.around(θss, 2)} °C')

# State-space rep
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

bT = np.array([10, 10, 22, 22, 22, 22, 22])     # [To, To, Ti, Ti, Ti, Ti, Ti]
fQ = np.array([0, 0, 0, 0])             # [Φ1, Φ2, Φ3, Φ4]
uss = np.hstack([bT, fQ]) 

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)

yss = (-Cs @ inv_As @ Bs + Ds) @ uss                        # output vector for state space are the temperatures θ5 and θ10
yss = float(yss.values[0])

print(f'yss_θ3 = {yss:.2f} °C')
print(f'Error between DAE and state-space: {abs(θss[2] - yss):.2e} °C')

