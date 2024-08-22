import numpy as np
import argparse
import pandas as pd

def rv_corr(spec, vcorr_best):
    """
    Corrige el efecto Doppler de velocidad radial en un espectro.

    Parameters:
    spec (pd.DataFrame): DataFrame que contiene datos espectrales con una columna "wavelength".
    vcorr_best (float): Velocidad radial de corrección en km/s.

    Returns:
    pd.DataFrame: DataFrame con la corrección del efecto Doppler aplicada en la columna "wavelength".

    Explanation:
    Esta función toma un DataFrame de espectros y una velocidad radial de corrección y realiza la corrección del efecto Doppler en las longitudes de onda del espectro. La corrección se realiza mediante la fórmula:
    corrected_wavelength = wavelength / sqrt((1 + vcorr_best/vlight) / (1 - vcorr_best/vlight))

    Donde:
    - corrected_wavelength: Longitud de onda corregida.
    - wavelength: Longitud de onda original del espectro.
    - vcorr_best: Velocidad radial de corrección en km/s.
    - vlight: Velocidad de la luz en km/s.

    El resultado es un nuevo DataFrame que contiene las longitudes de onda corregidas en la columna "wavelength".

    Ejemplo de uso:
    >>> import pandas as pd
    >>> data = pd.DataFrame({"wavelength": [656.3, 486.1, 434.0]})
    >>> corrected_data = rv_corr(data, 30.0)
    >>> print(corrected_data)
    """
    spec = spec.copy()  # Copia el DataFrame de entrada para evitar modificar el original
    vlight = 2.997925e5  # Velocidad de la luz en km/s
    fcor = np.sqrt((1.0 + vcorr_best / vlight) / (1.0 - vcorr_best / vlight))
    spec["wavelength"] = spec["wavelength"] / fcor  # Aplica la corrección a la columna "wavelength"
    return spec

def main():
    parser = argparse.ArgumentParser(description="Corrige el efecto Doppler de velocidad radial en un espectro.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con la columna 'wavelength'")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida con la columna corregida")
    parser.add_argument("-vel", "--vcorr", type=float, help="Velocidad radial de corrección en km/s")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo detallado (con mensajes)")

    args = parser.parse_args()
    
    if args.verbose:
        print(f"Cargando datos desde {args.input_file}")
    
    # Lee los datos del archivo CSV
    try:
        spec = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {args.input_file}")
        return

    # Aplica la corrección del efecto Doppler
    corrected_spec = rv_corr(spec, args.vcorr)

    if args.verbose:
        print(f"Guardando datos corregidos en {args.output_file}")
    
    # Guarda los datos corregidos en un nuevo archivo CSV
    corrected_spec.to_csv(args.output_file, index=False)
    if args.verbose:
        print("Corrección exitosa")

if __name__ == "__main__":
    main()
