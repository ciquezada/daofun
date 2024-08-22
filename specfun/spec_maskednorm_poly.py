import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def spec_poly_norm_masked(spec, plot=False, save_path=False, plot_title=False):
    """
    Normaliza un espectro utilizando una regresión polinómica de grado 2 en regiones específicas del continuo.

    Parameters:
    spec (pd.DataFrame): DataFrame que contiene datos espectrales con columnas "wavelength," "flux," y "flux_sig."
    plot (bool): Si es True, se crea un gráfico antes y después de la normalización.

    Returns:
    pd.DataFrame: DataFrame con la columna "flux" normalizada mediante una regresión polinómica.

    Explanation:
    Esta función toma un DataFrame de espectros como entrada y realiza las siguientes operaciones:
    1. Define máscaras para seleccionar regiones específicas del continuo alrededor del CaT IR.
    2. Crea una máscara combinada que incluye todas las regiones seleccionadas.
    3. Copia el DataFrame original y aplica la máscara para obtener datos de flujo del continuo.
    4. Realiza una regresión polinómica de grado 2 en los datos de longitud de onda y flujo del continuo.
    5. Normaliza la columna "flux" y "flux_sig" dividiendo por el valor del polinomio en toda la región.
    6. Si `plot` es True, se crea un gráfico antes y después de la normalización.

    Mínimo requerido en el DataFrame de entrada:
    - Columnas: "wavelength," "flux," y "flux_sig."

    Ejemplo de uso:
    >>> import pandas as pd
    >>> data = pd.DataFrame({"wavelength": [8473, 8564, 8626, 8701, 8777], "flux": [0.5, 0.7, 1.2, 1.5, 2.0], "flux_sig": [0.1, 0.2, 0.15, 0.18, 0.22]})
    >>> normalized_data = spec_poly_norm_masked(data, plot=True)
    >>> print(normalized_data)
    """

    spec_or = spec.copy()
    cont_mask_1 = (spec["wavelength"] >= 8474) & (spec["wavelength"] <= 8484)
    cont_mask_2 = (spec["wavelength"] >= 8563) & (spec["wavelength"] <= 8577)
    cont_mask_3 = (spec["wavelength"] >= 8625) & (spec["wavelength"] <= 8642)
    cont_mask_4 = (spec["wavelength"] >= 8700) & (spec["wavelength"] <= 8725)
    cont_mask_5 = (spec["wavelength"] >= 8776) & (spec["wavelength"] <= 8792)
    mask_list = [cont_mask_1, cont_mask_2, cont_mask_3, cont_mask_4, cont_mask_5]
    mask_cont = cont_mask_1 + cont_mask_2 + cont_mask_3 + cont_mask_4 + cont_mask_5
    data = spec[mask_cont].copy()

    x = data["wavelength"]
    y = data["flux"]
    coefs = np.polyfit(list(x.values), list(y.values), 2)
    poly = np.poly1d(coefs)

    spec["flux"] = spec["flux"] / poly(spec["wavelength"])
    spec["flux_sig"] = spec["flux_sig"] / poly(spec["wavelength"])

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
        axs[1].plot(spec["wavelength"], spec["flux"], c='green')
        [axs[1].plot(spec[mask].wavelength, spec[mask].flux, c='red') for mask in mask_list]

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    return spec

def main():
    parser = argparse.ArgumentParser(description="Normaliza un espectro utilizando regresión polinómica en regiones del continuo.")
    parser.add_argument("-i", "--input_file", help="Ruta del archivo CSV de entrada con columnas 'wavelength,' 'flux,' y 'flux_sig.'")
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

    spec_poly_norm_masked(spec, args.plot, save_path=args.save_path, 
                                    plot_title=os.path.splitext(args.input_file)[0])

    if args.output_file:
        spec.to_csv(args.output_file, index=False)

    if args.verbose:
        print("Normalización exitosa")

if __name__ == "__main__":
    main()