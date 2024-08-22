import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def spec_poly_norm(spec, plot=False, save_path=False, plot_title=False):
    """
    Normaliza un espectro utilizando una regresión polinómica de grado 4.

    Parameters:
    spec (pd.DataFrame): DataFrame que contiene datos espectrales con columnas "wavelength" y "flux".

    Returns:
    pd.DataFrame: DataFrame con la columna "flux" normalizada mediante una regresión polinómica.

    Explanation:
    Esta función toma un DataFrame de espectros como entrada y realiza las siguientes operaciones:
    1. Copia el DataFrame de entrada para evitar modificar el original.
    2. Obtiene las columnas "wavelength" y "flux".
    3. Realiza una regresión polinómica de grado 4 en los datos de longitud de onda y flujo.
    4. Calcula el polinomio resultante y normaliza la columna "flux" dividiendo por el valor del polinomio.
    5. Devuelve un nuevo DataFrame con la columna "flux" normalizada.

    Mínimo requerido en el DataFrame de entrada:
    - Columnas: "wavelength" y "flux".

    Ejemplo de uso:
    >>> import pandas as pd
    >>> data = pd.DataFrame({"wavelength": [400, 450, 500, 550, 600], "flux": [0.5, 0.7, 1.2, 1.5, 2.0]})
    >>> normalized_data = spec_poly_norm(data)
    >>> print(normalized_data)
    """

    spec_or = spec.copy()
    data = spec.copy()  # Copia el DataFrame de entrada
    x = data["wavelength"]
    y = data["flux"]

    # Realiza una regresión polinómica de grado 4 en los datos de longitud de onda y flujo
    coefs = np.polyfit(list(x.values), list(y.values), 4)
    poly = np.poly1d(coefs)

    # Calcula el polinomio resultante y normaliza la columna "flux" dividiendo por el valor del polinomio
    y_norm = y / poly(x)
    data["flux"] = y_norm
    if "flux_sig" in data.columns:
        data["flux_sig"] = data["flux_sig"] / poly(x)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].set_ylabel("Flux")
        axs[0].set_xlabel("Wavelength (Angstrom)")
        axs[0].grid()
        axs[0].set_title(f"Spectrum for {plot_title} before normalizing")
        axs[0].plot(spec_or["wavelength"], spec_or["flux"])

        axs[1].set_ylabel("Flux")
        axs[1].set_xlabel("Wavelength (Angstrom)")
        axs[1].grid()
        axs[1].set_title("Spectrum after normalizing")
        axs[1].plot(data["wavelength"], data["flux"], c='green')

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    return data

def main():
    parser = argparse.ArgumentParser(description="Normaliza un espectro mediante regresión polinómica de grado 4.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con columnas 'wavelength' y 'flux'")
    parser.add_argument("-o", "--output_file", help="Ruta del archivo CSV de salida con el espectro normalizado")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar mensajes detallados")
    parser.add_argument("-p", "--plot", action="store_true", help="Generar gráfico antes y después de la normalización")
    parser.add_argument("-f", "--save_path", help="Nombre del archivo JPEG para guardar el gráfico")

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

    normalized_spec = spec_poly_norm(spec, args.plot, save_path=args.save_path, 
                                    plot_title=os.path.splitext(args.input_file)[0])

    if args.output_file:
        normalized_spec.to_csv(args.output_file, index=False)
        if args.verbose:
            print(f"Guardando espectro normalizado en {args.output_file}")

    if args.verbose:
        print("Normalización exitosa")

if __name__ == "__main__":
    main()
