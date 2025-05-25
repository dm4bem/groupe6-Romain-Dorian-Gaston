# -*- coding: utf-8 -*-
"""
Created on Sun May 25 10:56:27 2025

@author: romai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import dm4bem

TC = dm4bem.file2TC('./model/TC.csv', name='', auto_number=False)

[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

controller = False
neglect_air_glass_capacity = True # Always true in this model
imposed_time_step = False
Δt = 4000    # s, imposed time step

# MODEL
# =====

# by default TC['G']['q6'] = 0 and TC['G']['q14'] = 0, i.e. Kp -> 0, no controller (free-floating)

if controller:
    TC['G']['q6'] = 1e3        # Kp -> ∞, almost perfect controller


# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
# print(f'λ = {λ}')

# time step
Δtmax = 2 * min(-1 / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

imposed_time_step = True

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")

dm4bem.print_rounded_time('dt', dt)

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")


To = 10 * np.ones(n)        # outdoor temperature 
Ti = 22 * np.ones(n)     # indoor temperature set point
Φ0 = 0 * np.ones(n)         # solar radiation absorbed by the glass
Φ3 = Φ1 = Φ2 = Φ0           # auxiliary heat sources and solar radiation

data = {
    'To': To,
    'To': To,
    'Ti': Ti,
    'Ti': Ti,
    'Ti': Ti,
    'Ti': Ti,
    'Ti': Ti,
    'Φ0': Φ0,
    'Φ1': Φ1,
    'Φ2': Φ2,
    'Φ3': Φ3
}

input_data_set = pd.DataFrame(data, index=time)


# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 14                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time integration \nTime step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()


print('Steady-state indoor temperature obtained with:')
print(f'- steady-state response to step input: \
{y_exp["θ2"].tail(1).values[0]:.4f} °C')




# Define the start and end dates for the simulation
start_date = '05-01 12:00:00'
end_date = '05-07 12:00:00'

start_date = '2025-' + start_date
end_date = '2025-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')

# Grenoble Alpes Isère Airport (LFLS) weather data
filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2025))
weather = weather.loc[start_date:end_date]

# Temperature sources
To = weather['temp_air']


### Parametres ###


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

τ_gSW = 0.30  # short wave transmitance: reflective blue glass

wall_out = pd.read_csv('./BLDG/walls_out.csv')
w1 = wall_out[wall_out['ID'] == 'w0'] #concrete
surface_orientation = {'slope': w1['β'].values[0],
                        'azimuth': w1['γ'].values[0],
                        'latitude': 45}

rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, w1['albedo'].values[0])

Etot = rad_surf.sum(axis=1)

Φ0 = alphawindow * S1 * Etot  
Φ1 = alphawout * S1 * Etot 
Φ2 = alphafloor * S_roof * Etot    
Φ3 = alpharoof * S_roof *  Etot    

# Indoor air temperature set-point
Ti_day, Ti_night = 20, 16

Ti1 = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)
Ti2 = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# No auxiliary heat sources in this model

# Input data set
input_data_set = pd.DataFrame({'To': To, 'Ti': Ti,
                                'Φ0': Φ0, 'Φ1': Φ1, 'Φ2': Φ2, 'Φ3': Φ3, 'Etot': Etot})

input_data_set.to_csv('./model/input_data_set.csv')
























