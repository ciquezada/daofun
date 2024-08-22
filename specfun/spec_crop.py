import argparse
import pandas as pd

def cut_spectrum(spec, min_wavelength, max_wavelength, verbose=False):
    if min_wavelength < spec['wavelength'].min() or max_wavelength > spec['wavelength'].max():
        print("Error: El rango de longitud de onda especificado está fuera del rango del espectro.")
        return

    if verbose:
        print(f"Cortando espectro en el rango {min_wavelength} - {max_wavelength} Angstrom")
    cut_spec = spec[(spec['wavelength'] >= min_wavelength) & (spec['wavelength'] <= max_wavelength)]
    
    if verbose:
        print("Corte exitoso")

    return cut_spec


def main():
    parser = argparse.ArgumentParser(description="Corta un espectro CSV en un rango de longitud de onda.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con la columna 'wavelength'")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida con el espectro cortado")
    parser.add_argument("-min", "--min_wavelength", type=float, help="Longitud de onda mínima para el corte en nm")
    parser.add_argument("-max", "--max_wavelength", type=float, help="Longitud de onda máxima para el corte en nm")
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

    cut_spec = cut_spectrum(spec, args.min_wavelength, args.max_wavelength, args.verbose)

    if args.verbose:
        print(f"Guardando espectro cortado en {args.output_file}")
    cut_spec.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
