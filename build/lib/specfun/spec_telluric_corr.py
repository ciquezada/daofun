import argparse
import pandas as pd
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import os


cwd_library = os.path.abspath(f"{os.path.dirname(__file__)}")

def telluric_corr(spec, cut_range=None, plot=False, plot_filename=None, plot_title=False, verbose=False):
    """
    Corrige un espectro por líneas telúricas utilizando un archivo telúrico de referencia.

    Parameters:
    spec (pd.DataFrame): DataFrame que contiene los datos espectrales con columnas "wavelength" y "flux."
    cut_range (tuple): Rango de longitud de onda (min, max) para cortar el espectro telúrico. Por ejemplo, (8380, 8870).

    Returns:
    pd.DataFrame: DataFrame con la columna "flux" corregida por líneas telúricas.

    Explanation:
    Esta función toma un DataFrame de espectros y realiza las siguientes operaciones:
    1. Lee un archivo telúrico de referencia ("std.tab") que contiene datos de longitud de onda y flujo.
    2. Aplica un corte al archivo telúrico si se especifica un rango (cut_range).
    3. Interpola el archivo telúrico utilizando un splrep (spline).
    4. Divide la columna "flux" del espectro por la interpolación del archivo telúrico.
    5. La columna "wavelength" del espectro debe coincidir con las longitudes de onda del archivo telúrico.

    Mínimo requerido en el DataFrame de entrada:
    - Columnas: "wavelength" y "flux."

    Ejemplo de uso:
    >>> import pandas as pd
    >>> data_spec = pd.DataFrame({"wavelength": [8381, 8390, 8400, 8410], "flux": [1.2, 1.1, 1.3, 1.0]})
    >>> corrected_spec = telluric_corr(data_spec, cut_range=(8380, 8870))
    >>> print(corrected_spec)
    """

    df_telluric = pd.read_csv(os.path.join(cwd_library, "std.tab"), delim_whitespace=True, names=["wavelength", "flux"])

    if cut_range:
        df_telluric = df_telluric.query(f"wavelength>={cut_range[0]} and wavelength<={cut_range[1]}")

    spline_telluric = splrep(df_telluric.wavelength, df_telluric.flux, k=3)

    if plot:
        if verbose:
            print("Generando gráfico antes y después de la corrección telúrica...")
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].set_ylabel("Flux (Before Correction)")
        axs[0].set_xlabel("Wavelength (Angstrom)")
        axs[0].grid()
        axs[0].set_title(f"Spectrum for {plot_title} before telluric correction")
        axs[0].plot(spec["wavelength"], spec["flux"], c='green')

        axs[1].set_ylabel("Flux (After Correction)")
        axs[1].set_xlabel("Wavelength (Angstrom)")
        axs[1].grid()
        axs[1].set_title("Spectrum after telluric correction")
        axs[1].plot(spec["wavelength"], spec["flux"] / splev(spec["wavelength"], spline_telluric), c='red')

        if plot_filename:
            plt.savefig(plot_filename, dpi=300)
            if verbose:
                print(f"Gráfico guardado en {plot_filename}")
        else:
            plt.show()

    spec["flux"] = spec["flux"] / splev(spec["wavelength"], spline_telluric)
    return spec

def main():
    parser = argparse.ArgumentParser(description="Corrige un espectro por líneas telúricas utilizando un archivo telúrico de referencia.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con columnas 'wavelength' y 'flux.'")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida con el espectro corregido")
    parser.add_argument("-range", "--cut_range", nargs=2, type=float, help="Rango de longitud de onda (min max) para cortar el espectro telúrico. Por ejemplo, 8380 8870.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensajes detallados")
    parser.add_argument("-p", "--plot", action="store_true", help="Generar gráfico antes y después de la normalización")
    parser.add_argument("-f", "--plot_filename", help="Nombre del archivo JPEG para guardar el gráfico")


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

    if args.cut_range:
        cut_range = tuple(args.cut_range)
    else:
        cut_range = None

    telluric_corr(spec, cut_range, args.plot, plot_filename=args.plot_filename, 
                                    plot_title=os.path.splitext(args.input_file)[0], verbose=args.verbose)

    if args.output_file:
        spec.to_csv(args.output_file, index=False)

    if args.verbose:
        print("Corrección telúrica exitosa")

if __name__ == "__main__":
    main()
