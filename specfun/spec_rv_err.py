import numpy as np
import argparse
import pandas as pd

def rv_err_measure(star_type, SNR, FeH, save_files, output_folder, verbose=False):
    """
    Calcula el error segun Alessio
    """
    rverr_path = os.path.join(output_folder, "spec_rverr.csv")
    if star_type == "Dwarf":
        rverr = np.exp(4.209 - 0.997*np.log(SNR) - 0.029*FeH + 0.058*FeH**2)
    elif star_type == "Giant":
        rverr = np.exp(4.624 - 1.023*np.log(SNR) - 0.159*FeH + 0.120*FeH**2)

    if save_files:
        os.makedirs(output_folder, exist_ok=True)
        salida = {"filn":[os.path.basename(output_folder)], "rverr":[rverr]}
        pd.DataFrame(salida).to_csv(rverr_path, index=False)
        if verbose:
            print(f"[Info] Results written in {rverr_path}\n")
    return rverr

def main():
    parser = argparse.ArgumentParser(description="Corrige el efecto Doppler de velocidad radial en un espectro.")
    parser.add_argument("-st", "--star_type", help="Ruta del archivo CSV de entrada con la columna 'wavelength'")
    parser.add_argument("-snr", "--snr", help="Ruta del archivo CSV de salida con la columna corregida")
    parser.add_argument("-feh", "--feh", type=float, help="Velocidad radial de corrección en km/s")
    parser.add_argument("--save_files", help="Save files", action="store_true")
    parser.add_argument("--output_folder", help="Output folder for saved files", type=str, default=".")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo detallado (con mensajes)")

    args = parser.parse_args()
    
    rv = rv_measure(args.star_type, args.snr, args.feh, args.save_files, args.output_folder, args.verbose)

    return rv

    

if __name__ == "__main__":
    main()
