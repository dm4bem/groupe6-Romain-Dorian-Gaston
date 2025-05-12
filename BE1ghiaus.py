import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
θ = [f'θ{i}' for i in range(30)]
q = [f'q{i}' for i in range(39)]  # Mis à jour pour correspondre aux 13 branches

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

# Matrice G
G = np.zeros(29)

# Fenêtre : branches 0–2
G[0] = hi * S_wind            # Convection sur la fenêtre (0)
G[1] = (λ_g / (2 * w_g)) * S_wind  # Conduction à travers le vitrage (1)
G[2] = ho * S_wind            # Convection à l'extérieur (2)

# Mur 1 : branches 3–7
G[3] = hi * S1                # Convection sur le Mur 1 (3)
G[4] = (λ_c / (2 * w_c)) * S1    # Conduction à travers le béton (4)
G[5] = (λ_i / w_i) * S1       # Conduction à travers l'isolation (5)
G[6] = (λ_c / (2 * w_c)) * S1    # Conduction à travers le béton (6)
G[7] = hi * S1                # Convection à l'extérieur (7)

# Mur 2 : branches 8–11
G[8] = hi * S2                # Convection sur le Mur 2 (8)
G[9] = (λ_c / (2 * w_c)) * S2    # Conduction à travers le béton (9)
G[10] = (λ_i / w_i) * S2      # Conduction à travers l'isolation (10)
G[11] = hi * S2               # Convection à l'extérieur (11)

# Rayonnements (branches 12, 13, 18, 19)
G[12] = G[13] = G[18] = G[19] = GLW  # Rayonnement

# Mur 3 : branches 14–17
G[14] = hi * S3                # Convection sur le Mur 3 (14)
G[15] = (λ_c / (2 * w_c)) * S3    # Conduction à travers le béton (15)
G[16] = (λ_i / w_i) * S3       # Conduction à travers l'isolation (16)
G[17] = hi * S3                # Convection à l'extérieur (17)

# Porte : branche 20
G[20] = (2 * hi + λ_door / w_door) * S_door  # Convection sur la porte (20)

# Sol : branches 21–24
G[21] = hi * S_floor            # Convection sur le sol (21)
G[22] = (λ_f / (2 * w_f)) * S_floor  # Conduction à travers le sol (22)
G[23] = (λ_f / (2 * w_f)) * S_floor  # Conduction à travers le sol (23)
G[24] = hi * S_floor            # Convection à l'extérieur (24)

# Plafond : branches 25–28
G[25] = hi * S_roof             # Convection sur le plafond (25)
G[26] = (λ_r / (2 * w_r)) * S_roof  # Conduction à travers le plafond (26)
G[27] = (λ_r / (2 * w_r)) * S_roof  # Conduction à travers le plafond (27)
G[28] = ho * S_roof             # Convection à l'extérieur (28)

# Résultat : matrice G sous forme diagonale
G = np.diag(G)



# Capacité thermique par nœud 
C = np.zeros(19)
C[1] = concrete['Density'] * concrete['Specific heat'] * concrete['Surface'] * concrete['Width']
C[3] = insulation['Density'] * insulation['Specific heat'] * insulation['Surface'] * insulation['Width']
C[6] = glass['Density'] * glass['Specific heat'] * glass['Surface'] * glass['Width']
C[12] = air['Density'] * air['Specific heat'] * V_air
C = pd.Series(C, index=θ[:19])  

# Sources de température (b) et flux (f)
b = np.zeros(29)
b[3]=b[0]=To
b[8]=b[14]=b[20]=b[21]=b[25]=Ti

# Il y a 19 nœuds → 19 éléments dans f
f=np.zeros(19)
f[0]=E*alphawindow*S1
f[3]=E*alphawout*S1
f[16]=E*alpharoof*S_roof
f[13]=E*alphafloor*S_roof

# Température d’intérêt : air intérieur
y = pd.Series(np.zeros(19), index=θ[:19])
y['θ2'] = 1  # on dit que le nœud d’air intérieur est θ2

# Création du modèle thermique complet
TC = {"A": A, "G": G, "C": C, "b": b, "f": f, "y": y}

θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
q = G @ (-A @ θ + b)

print(f"The temperature in the room is : θ2 = {θ[2]:.2f} °C")