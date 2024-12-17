import pandas as pd
import numpy as np

# Constants for the CF formula
a1 = 0.42036059446739293
a2 = 0.8335622813711547
m1 = 0.24940266639537556
m2 = -0.13137915252218796
n1 = -0.31787964985255074
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
H = 8500  # Scale height in meters

# Function to compute CF
def calculate_cf(dhi_m, ghi_m, dni_m, dni_c):
    k_d = dhi_m / ghi_m
    k_t = dni_m / dni_c
    cf = a1 * k_d**a2 + m1 * k_t**m2 + n1
    return cf

# Function to predict epsilon_sky
def predict_epsilon_sky(cf, epsilon_clear_sky, epsilon_clouds):
    epsilon_sky = (1 - cf) * epsilon_clear_sky + cf * epsilon_clouds
    return epsilon_sky

# Function to calculate actual emissivity from temperature and downwelling longwave radiation
def calculate_actual_epsilon(temp, dlw):
    t_kelvin = temp + 273.15  # Convert temperature from Celsius to Kelvin
    epsilon = dlw / (sigma * t_kelvin**4)
    return epsilon

# Function to calculate clear-sky emissivity
def calculate_clear_sky_emissivity(temp, rh, altitude):
    e_s = 6.112 * np.exp(17.625 * temp / (temp - 30.11 + 273.15))  # Saturation vapor pressure
    pw_hpa = e_s * rh / 100  # Partial water vapor pressure
    sqrt_pw = np.sqrt(pw_hpa / 1013.25)
    e_clear_sky = 0.6 + 1.652 * sqrt_pw + 0.15 * (np.exp(-altitude / H) - 1)
    return e_clear_sky

# Error calculation functions
def calculate_errors(actual, predicted):
    errors = predicted - actual
    mbe = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    rMBE = mbe / np.mean(actual) * 100  # Relative MBE (%)
    rRMSE = rmse / np.mean(actual) * 100  # Relative RMSE (%)
    return mbe, rmse, rMBE, rRMSE

# Load your Hawaii dataset
data = pd.read_excel("/Users/akarsh1207/Desktop/Lab/Coimbra Research Group/Longwave-Radiation-Model-Hawaii/Cleaned Station Data/014HI_final.xlsx")
# Assuming your dataset has columns: 'DHI', 'GHI', 'DNI', 'Clearsky DNI', 'Temp', 'DLW', 'RH', 'Altitude',
# 'epsilon_clouds'
dhi_m = data['DHI']
ghi_m = data['GHI']
dni_m = data['DNI']
dni_c = data['Clearsky DNI']
temp = data['temp']
dlw = data['dlw']
rh = data['rh']
altitude = data['site_elev']
epsilon_clouds = 1

# Data filtering conditions
condition_0 = (data['Solar Zenith Angle'] < 72.5)
condition_1 = (data['GHI'] > 0) & (data['DHI'] > 0)
condition_2 = (data['DNI'] / data['Clearsky DNI'] > 0) & (data['DNI'] / data['Clearsky DNI'] <= 1.5)
condition_3 = (data['temp'] <= 90) & (data['temp'] >= -80)
condition_4 = (data['dlw'] > 0)
data = data[condition_0 & condition_1 & condition_2 & condition_3 & condition_4]

# Compute actual epsilon
data['actual_epsilon'] = calculate_actual_epsilon(temp, dlw)

# Compute clear-sky emissivity
data['epsilon_clear_sky'] = calculate_clear_sky_emissivity(temp, rh, altitude)

# Compute CF and predicted epsilon_sky
data['CF'] = calculate_cf(dhi_m, ghi_m, dni_m, dni_c)
data['predicted_epsilon'] = predict_epsilon_sky(data['CF'], data['epsilon_clear_sky'], epsilon_clouds)

# Calculate errors
mbe, rmse, rMBE, rRMSE = calculate_errors(data['actual_epsilon'], data['predicted_epsilon'])

# Print results
print("Mean Bias Error (MBE):", mbe)
print("Root Mean Square Error (RMSE):", rmse)
print("Relative Mean Bias Error (rMBE, %):", rMBE)
print("Relative Root Mean Square Error (rRMSE, %):", rRMSE)

# Optional: Save the results to a new CSV
data.to_csv("predicted_results_with_errors.csv", index=False)
