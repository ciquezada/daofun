import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

def spec_uspline_corr(spec):
    """
    Corrige valores NaN en el espectro utilizando interpolación cúbica.

    Parameters:
    spec (pd.DataFrame): DataFrame que contiene datos espectrales con columnas "wavelength" y "flux".

    Returns:
    pd.DataFrame: DataFrame con valores NaN en la columna "flux" corregidos mediante interpolación cúbica.

    Explanation:
    Esta función toma un DataFrame de espectros como entrada y realiza las siguientes operaciones:
    1. Copia el DataFrame de entrada para evitar modificar el original.
    2. Elimina filas con valores NaN en las columnas "wavelength" y "flux".
    3. Realiza una interpolación cúbica (splines) en los datos no nulos de longitud de onda y flujo.
    4. Reemplaza los valores NaN en la columna "flux" con los valores interpolados.
    5. Devuelve un nuevo DataFrame con los valores NaN corregidos en la columna "flux".

    Mínimo requerido en el DataFrame de entrada:
    - Columnas: "wavelength" y "flux".

    Ejemplo de uso:
    >>> import pandas as pd
    >>> data = pd.DataFrame({"wavelength": [400, 450, 500, 550, 600], "flux": [0.5, 0.7, NaN, 1.2, 1.5]})
    >>> corrected_data = spec_uspline_corr(data)
    >>> print(corrected_data)
    """

    data_corr = spec.copy()  # Copia el DataFrame de entrada
    data = data_corr.dropna()  # Elimina filas con valores NaN

    x = data["wavelength"]
    y = data["flux"]

    # Realiza una interpolación cúbica (splines) en los datos no nulos de longitud de onda y flujo
    spl = UnivariateSpline(x, y)

    # Reemplaza los valores NaN en la columna "flux" con los valores interpolados
    data_corr["flux"] = data_corr.apply(lambda row: row["flux"] if not np.isnan(row["flux"]) else spl(row["wavelength"]), axis=1)

    return data_corr

def main():
    parser = argparse.ArgumentParser(description="Corrige valores NaN en un espectro utilizando interpolación cúbica.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con columnas 'wavelength' y 'flux'")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida con el espectro corregido")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensajes detallados")
    
    args = parser.parse_args()
    
    if args.input_file:
        try:
            spec = pd.read_csv(args.input_file)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {args.input_file}")
            return
    else:
        print("Error: Se requiere un archivo de entrada.")
        return
    
    corrected_spec = spec_uspline_corr(spec)
    
    if args.output_file:
        corrected_spec.to_csv(args.output_file, index=False)
        if args.verbose:
            print(f"Guardando espectro corregido en {args.output_file}")
    
    if args.verbose:
        print("Corrección exitosa")

if __name__ == "__main__":
    main()
