# Importa las bibliotecas necesarias
from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import argparse


def interpolate_flux(df, wavelength_intermediate):
    """
    Interpola el flujo en una longitud de onda intermedia utilizando splines cúbicos.

    Parameters:
    - df (pd.DataFrame): DataFrame con columnas 'wavelength' y 'flux'.
    - wavelength_intermediate (float): Longitud de onda intermedia a interpolar.

    Returns:
    - float: Flujo interpolado en la longitud de onda intermedia.
    """
    # Ordena el DataFrame por la columna de longitud de onda para garantizar la interpolación correcta.
    df = df.sort_values(by='wavelength')

    # Extrae las columnas de longitud de onda y flujo como arrays NumPy.
    wavelengths = df['wavelength'].values
    fluxes = df['flux'].values

    # Encuentra los índices de los valores más cercanos a la longitud de onda intermedia.
    closest_indices = np.sort(np.argsort(np.abs(wavelengths - wavelength_intermediate))[:4])

    # Obtiene las longitudes de onda y flujos correspondientes a los valores más cercanos.
    closest_wavelengths = wavelengths[closest_indices]
    closest_fluxes = fluxes[closest_indices]

    # Crea un objeto de spline cúbico con solo estos dos puntos.
    spline = CubicSpline(closest_wavelengths, closest_fluxes, bc_type='natural')

    # Evalúa el spline cúbico en el valor de longitud de onda intermedia.
    interpolated_flux = spline(wavelength_intermediate)

    return interpolated_flux

def sync_wavelength(spec, sync_spec):
    """
    Sincroniza el espaciado de longitud de onda de un espectro de entrada con el de otro espectro de referencia,
    interpolando valores utilizando splines cúbicos.

    Parameters:
    - spec (pd.DataFrame): DataFrame con columnas 'wavelength' y 'flux'.
    - sync_spec (pd.DataFrame): DataFrame de referencia con la columna 'wavelength'.

    Returns:
    - pd.DataFrame: DataFrame sincronizado con columnas 'wavelength' y 'flux'.
    """
    if type(sync_spec)==str:
        sync_spec = pd.read_csv(sync_spec)

    # # Extrae la longitud de onda del espectro de referencia para sincronizar
    # wavelength_sync = sync_spec['wavelength']

    # Se define el espectro de salida
    sync_spec["flux"] = sync_spec.wavelength.apply(lambda wl: interpolate_flux(spec, wl))

    return sync_spec


def main():
    """
    Script principal que utiliza argparse para corregir valores NaN en un espectro.
    """
    # Configura el parser de argumentos
    parser = argparse.ArgumentParser(description="Sincroniza el espaciado de longitud de onda de un espectro de entrada con el de otro espectro de referencia, interpolando valores utilizando splines cúbicos.")
    parser.add_argument("-i", "--input_csv", help="Ruta del archivo CSV de entrada con columnas 'wavelength' y 'flux'", required=True)
    parser.add_argument("-s", "--sync_spec_csv", help="Ruta del archivo CSV de referencia con la columna 'wavelength'", required=True)
    parser.add_argument("-o", "--output_csv", help="Ruta del archivo CSV de salida con el espectro corregido", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensajes detallados")

    # Parsea los argumentos de la línea de comandos
    args = parser.parse_args()

    # Lee los archivos CSV de entrada y referencia
    try:
        spec = pd.read_csv(args.input_csv)
        sync_spec = pd.read_csv(args.sync_spec_csv)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}")
        return

    # Sincroniza los espectros
    spec_synched = sync_wavelength(spec, sync_spec)

    # Guarda el espectro sincronizado en un archivo CSV
    spec_synched.to_csv(args.output_csv, index=False)
    if args.verbose:
        print(f"Guardando espectro sincronizado en {args.output_csv}")

    if args.verbose:
        print("Sincronización exitosa")

if __name__ == "__main__":
    main()
