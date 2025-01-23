import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def brutsaert_clear_sky(Ta, ea, sigma=5.67e-8):
    """
    Brutsaert (1975) clear-sky model for LW↓.
    Ta: Air temperature (K)
    ea: Vapor pressure (hPa)
    """
    epsilon_0 = 1.24 * (ea / Ta) ** (1 / 7)
    return epsilon_0 * sigma * Ta ** 4

def calculate_cloud_fraction(ghi_m, ghi_c, clr_pct=None):
    """
    Calculate cloud fraction (c) based on measured GHI and clear-sky GHI.
    Optionally use clr_pct if provided.
    ghi_m: Measured global horizontal irradiance (W/m^2)
    ghi_c: Clear-sky global horizontal irradiance (W/m^2)
    clr_pct: Clear sky fraction (optional)
    """
    if clr_pct is not None:
        return 1 - clr_pct
    cloud_fraction = 1 - (ghi_m / ghi_c)
    return np.clip(cloud_fraction, 0, 1)

def maykut_church(LW_clear, c):
    """Maykut and Church (1973) cloudy-sky correction."""
    return LW_clear * (1 + 0.22 * c ** 2.75)

def jacobs(LW_clear, c):
    """Jacobs (1978) cloudy-sky correction."""
    return LW_clear * (1 + 0.26 * c)

def suguita_brutsaert(LW_clear, c):
    """Suguita and Brutsaert (1993) cloudy-sky correction."""
    return LW_clear * (1 + 0.0496 * c ** 2.45)

def konzelmann(LW_clear, c, Ta, sigma=5.67e-8):
    """Konzelmann et al. (1994) cloudy-sky correction."""
    return LW_clear * (1 - c ** 4) + 0.952 * c ** 4 * sigma * Ta ** 4

def crawford_duchon(LW_clear, c, Ta, sigma=5.67e-8):
    """Crawford and Duchon (1999) cloudy-sky correction."""
    return LW_clear * (1 - c) + c * sigma * Ta ** 4

def lhomme(LW_clear, c):
    """Lhomme et al. (2007) cloudy-sky correction."""
    return LW_clear * (1.03 + 0.34 * c)

def compute_cloudy_LW(Ta, ea, c, model='crawford_duchon'):
    """
    Compute downward longwave radiation under cloudy conditions.
    Ta: Air temperature (K)
    ea: Vapor pressure (hPa)
    c: Cloud fraction (0 to 1)
    model: Model selection ('maykut_church', 'jacobs', 'suguita_brutsaert', 
            'konzelmann', 'crawford_duchon', 'lhomme')
    """
    LW_clear = brutsaert_clear_sky(Ta, ea)
    if model == 'maykut_church':
        return maykut_church(LW_clear, c)
    elif model == 'jacobs':
        return jacobs(LW_clear, c)
    elif model == 'suguita_brutsaert':
        return suguita_brutsaert(LW_clear, c)
    elif model == 'konzelmann':
        return konzelmann(LW_clear, c, Ta)
    elif model == 'crawford_duchon':
        return crawford_duchon(LW_clear, c, Ta)
    elif model == 'lhomme':
        return lhomme(LW_clear, c)
    else:
        raise ValueError("Invalid model name. Choose from predefined models.")

def compute_emissivity(LW_measured, Ta, sigma=5.67e-8):
    """
    Compute emissivity using Stefan-Boltzmann law.
    LW_measured: Measured longwave radiation (W/m^2)
    Ta: Air temperature (K)
    sigma: Stefan-Boltzmann constant.
    Returns: Emissivity (dimensionless)
    """
    return LW_measured / (sigma * Ta ** 4)

def read_surfrad_h5(file_path, stations):
    """
    Read SURFRAD HDF5 file using pd.read_hdf for given stations.
    file_path: Path to the HDF5 file.
    stations: List of station names to process.
    Returns a combined dataframe.
    """
    combined_data = []
    for station in stations:
        try:
            df = pd.read_hdf(file_path, key=station)
            df['Cloud_Fraction'] = calculate_cloud_fraction(df['ghi_m'], df['ghi_c'], df.get('clr_pct', None))
            df['Emissivity_measured'] = compute_emissivity(df['dlw_m'], df['t_m'])
            df['Ta'] = df['t_m']  # Map t_m to Ta
            df['ea'] = df['pw_hpa']  # Map pw_hpa to ea
            df['Station'] = station
            combined_data.append(df)
        except KeyError as e:
            print(f"Warning: {e}. Skipping station {station}.")
    return pd.concat(combined_data, ignore_index=True)

def validate_emissivity(data, models):
    """
    Validate emissivity models using SURFRAD data.
    Calculate MBE, RMSE, rMBE, and rRMSE.
    """
    results = {}
    for model in models:
        predicted_LW = compute_cloudy_LW(data['Ta'], data['ea'], data['Cloud_Fraction'], model)
        predicted_emissivity = predicted_LW / (5.67e-8 * data['Ta'] ** 4)
        measured_emissivity = data['Emissivity_measured']
        error = predicted_emissivity - measured_emissivity
        mbe = np.mean(error)
        rmse = np.sqrt(np.mean(error ** 2))
        rMBE = (mbe / np.mean(measured_emissivity)) * 100
        rRMSE = (rmse / np.mean(measured_emissivity)) * 100
        results[model] = {
            'MBE': mbe,
            'RMSE': rmse,
            'rMBE': rMBE,
            'rRMSE': rRMSE
        }
    return results

# Example usage with SURFRAD data
if __name__ == "__main__":
    file_path = r'./CS/data.h5'  # SURFRAD HDF5 data file
    stations = ['BON', 'DRA', 'FPK', 'GWC', 'PSU', 'SXF', 'TBL']
    print("Reading SURFRAD data...")
    surfrad_data = read_surfrad_h5(file_path, stations)
    print(f"Data loaded for stations: {stations}")
    models = ['maykut_church', 'jacobs', 'suguita_brutsaert', 'konzelmann', 'crawford_duchon', 'lhomme']
    print("Validating emissivity models with SURFRAD data...")
    model_results = validate_emissivity(surfrad_data, models)
    for model, metrics in model_results.items():
        print(f"{model}: MBE = {metrics['MBE']:.4f}, RMSE = {metrics['RMSE']:.4f}, "
              f"rMBE = {metrics['rMBE']:.2f}%, rRMSE = {metrics['rRMSE']:.2f}%")
    best_model = min(model_results, key=lambda m: model_results[m]['RMSE'])
    print(f"Best model based on RMSE: {best_model}")