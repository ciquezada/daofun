import argparse
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np



def fits2ascii(fits_file):
    """
    Convierte un archivo FITS en un DataFrame de Pandas.

    Parameters:
    fits_file (str): Ruta del archivo FITS a ser convertido.

    Returns:
    pd.DataFrame: Un DataFrame con tres columnas: "wavelength," "flux," y "flux_sig."

    Explanation:
    Esta función toma un archivo FITS como entrada y realiza las siguientes operaciones:
    1. Abre el archivo FITS especificado.
    2. Extrae los datos de flujo y los datos de la señal de flujo del archivo FITS.
    3. Realiza la conversión de bytes si es necesario.
    4. Obtiene la información de coordenadas WCS (Sistema de coordenadas mundiales) del encabezado FITS.
    5. Calcula las longitudes de onda a partir de las coordenadas WCS.
    6. Crea un DataFrame de Pandas con tres columnas: "wavelength," "flux," y "flux_sig," que contienen
       las longitudes de onda, los datos de flujo y los datos de la señal de flujo, respectivamente.

    Nota: La columna "wavelength" contiene longitudes de onda multiplicadas por 10^9 para convertirlas a nanómetros.

    Ejemplo de uso:
    >>> fits_file = 'mi_archivo.fits'
    >>> dataframe = fits2ascii(fits_file)
    >>> print(dataframe)
    """

    f = fits.open(fits_file)  # Abre el archivo FITS
    flux = f[0].data  # Extrae los datos de flujo del primer elemento del archivo FITS
    flux = flux.byteswap().newbyteorder()  # Realiza conversión de bytes si es necesario
    flux_sig = f[1].data  # Extrae los datos de la señal de flujo del segundo elemento del archivo FITS
    flux_sig = flux_sig.byteswap().newbyteorder()  # Realiza conversión de bytes si es necesario
    wcs = WCS(f[0].header)  # Obtiene la información de coordenadas WCS del encabezado FITS
    index = np.arange(f[0].header['NAXIS1'])  # Crea un índice basado en el número de píxeles en el eje X
    wavelength = wcs.wcs_pix2world(index[:,np.newaxis], 0)  # Convierte los índices en longitudes de onda
    wave = wavelength.flatten() * 10E9  # Convierte las longitudes de onda a nanómetros
    return pd.DataFrame({"wavelength": wave, "flux": flux, "flux_sig": flux_sig})  # Crea un DataFrame y lo devuelve

def main():
    parser = argparse.ArgumentParser(description="Convierte un archivo FITS en un archivo CSV.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo FITS de entrada")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensajes detallados")
    
    args = parser.parse_args()
    
    input_file_path = args.input_file
    output_file_path = args.output_file
    
    if args.verbose:
        print(f"Convirtiendo {input_file_path} a {output_file_path}")
    
    spectra_df = fits2ascii(input_file_path)
    spectra_df.to_csv(output_file_path, index=False)
    
    if args.verbose:
        print(f"Conversión exitosa. Datos guardados en {output_file_path}")

if __name__ == "__main__":
    main()