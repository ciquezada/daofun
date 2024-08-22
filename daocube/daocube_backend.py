from daofun.daophot_opt import daophot_dict, phot_dict, allstar_dict
from daofun.fits_handler import FITSFigureV2
from daofun.misc_tools import check_file, read_als, mag2flux, read_ap
from daocube.daophot_pipeline import DaoPipe


import os
import shutil
import numpy as np
import pandas as pd
from spectral_cube import SpectralCube
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings


cwd = os.path.abspath(os.getcwd())

class DaoCube:
    
    cmd_list = ["create_psf",  "allstar",
                            "create_psf", "refine_psf", 
                                "allstar"]

    def __init__(self):
        self.daophot_dict = daophot_dict
        self.phot_dict = phot_dict
        self.allstar_dict = allstar_dict
        self.allstar_dict["re"] = 0
        self.daophot_dict_path = None
        self.phot_dict_path = None
        self.allstar_dict_path = None
        self.cube_fits = None
        self.cube_spectra_folder = None
        self.cube_spectra_folder_path = None
        self.cube_fits_name = None
        self.cube_slices_folder = None
        self.cube_len = None
        self.cube_slices_path = []
        self.target_als = None
        self.target_lst = None
        self.mlist_queue = []
        self.last_als_type = "als"
        self.n_jobs = -1
        
    def init_cube(self, cube_path):
        check_file(cube_path, "Cube fits file: ")
        self.cube_fits = os.path.abspath(cube_path)
        self.cube_fits_folder = os.path.abspath(os.path.dirname(self.cube_fits))
        self.cube_fits_name = os.path.splitext(os.path.basename(cube_path))[0]
        self.cube_slices_folder = "working_on_" + self.cube_fits_name + "_slices"
        self.cube_spectra_folder = self.cube_fits_name + "_spectra"
        self.cube_spectra_folder_path = os.path.abspath(os.path.join(self.cube_fits_folder, self.cube_spectra_folder))
        os.chdir(self.cube_fits_folder)
        print("Loading Cube: (expecting MUSE standard cube)")
        cube = SpectralCube.read(self.cube_fits, hdu=1)
        self.cube_len = cube.shape[0]
        self.first_slice = 0
        self.last_slice = self.cube_len-1

    def slice_cube(self, exist_ok=True, prog_bar=tqdm):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube = SpectralCube.read(self.cube_fits, hdu=1)
            os.makedirs(os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder), 
                                                exist_ok=exist_ok)
            print(f"\tSlices in: {self.cube_slices_folder}")

            for slice_i in prog_bar(range(self.first_slice, self.last_slice+1), 
                                    total=self.last_slice-self.first_slice, desc='Slicing Cube'):
                filepath = os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder, 
                                                    f"{self.cube_fits_name}_{slice_i:04}.fits")
                if not os.path.exists(filepath):
                    cube[slice_i,:,:].write(filepath)
                self.cube_slices_path.append(filepath)

    def run_slices_photometry(self, prog_bar=tqdm):
        self.save_opt_daophot(os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder))
        self.save_opt_phot(os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder))
        self.save_opt_allstar(os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder))
        arg_list = [[slice_path] for slice_path in self.cube_slices_path]

        slices_status = Parallel(n_jobs=self.n_jobs, verbose=False)(
            delayed(self.do_slice_photometry)(*args) for args in prog_bar(arg_list, 
                                                                desc='Photometring Cube', total=len(arg_list)))
        with open("photometry_pipeline.log", "w") as f:
            f.write("\n".join(slices_status))


    def load_opt_file(self, filepath, opt_dict):
        with open(filepath, 'r') as archivo:
            lineas = archivo.readlines()
        # Eliminar espacios en blanco y saltos de línea, dividir por '=' para obtener clave y valor
        valores = {}
        for linea in lineas:
            if '=' in linea:
                clave, valor = linea.strip().split('=')
                valores[clave.strip()] = float(valor.strip())
        self.update_opt_dict(valores, opt_dict)
        
    def update_opt_dict(self, values, opt_dict):
        for k in values.keys():
            if k in opt_dict.keys():
                opt_dict[k] = values[k]

    def load_als(self, als_path):
        check_file(als_path, "Targets als file: ")
        self.target_als = als_path

    def load_lst(self, lst_path):
        check_file(lst_path, "Targets lst file: ")
        self.target_lst = lst_path

    def do_slice_photometry(self, slice_path):
        try:
            os.chdir(self.cube_fits_folder)
            daopipe = DaoPipe(verbose=False)
            daopipe.init_fits(slice_path)
            daopipe.load_opt_daophot(self.daophot_dict_path)
            daopipe.load_opt_phot(self.phot_dict_path)
            daopipe.load_opt_allstar(self.allstar_dict_path)
            daopipe.load_als(self.target_als)
            daopipe.load_lst(self.target_lst)
            daopipe.mlist_queue = [mlist for mlist in self.mlist_queue]
            daopipe.pipeline_cmd += self.cmd_list
            daopipe.run_photometry()
            daopipe.save_photometry()
            daopipe.clean()
            output = "OK"
            os.chdir(self.cube_fits_folder)
        except FileNotFoundError as err:
            output = err
        return(f"[{output}]: {os.path.basename(slice_path)}")
    
    def new_master_list(self, mlist_path):
        self.mlist_queue.append(os.path.abspath(mlist_path))
    
    def extract_spectra(self, prog_bar=tqdm):
        extract_method = read_als if self.last_als_type=="als" else read_ap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_stars =  read_als(self.mlist_queue[-1])[["ID"]].set_index("ID") if len(self.mlist_queue)>0 else read_als(self.target_als)[["ID"]].set_index("ID")
            all_stars_err =  read_als(self.mlist_queue[-1])[["ID"]].set_index("ID") if len(self.mlist_queue)>0 else read_als(self.target_als)[["ID"]].set_index("ID")
            # for slice in range(1200, 3721):
            for slice_i in prog_bar(range(self.first_slice, self.last_slice+1), 
                                         total=self.last_slice-self.first_slice, desc='Extracting spectra'): 
                try:
                    filepath = os.path.join(self.cube_fits_folder,
                                                self.cube_slices_folder, 
                                                    f"{self.cube_fits_name}_{slice_i:04}.als")   
                    # para magnitud
                    new_slice = extract_method(filepath)[["ID", "MAG"]].set_index("ID")
                    new_slice.columns = [f"wl_{slice_i:04}"]
                    all_stars = all_stars.join(new_slice)
                    del(new_slice)
                    # para errores
                    new_slice_err = extract_method(filepath)[["ID", "merr"]].set_index("ID")
                    new_slice_err.columns = [f"wl_{slice_i:04}"]
                    all_stars_err = all_stars_err.join(new_slice_err) 
                    del(new_slice_err)
                except FileNotFoundError as err:
                    all_stars[f"wl_{slice_i:04}"] = np.NaN
                    all_stars_err[f"wl_{slice_i:04}"] = np.NaN
        all_stars = all_stars.copy()
        all_stars_err = all_stars_err.copy()
        os.makedirs(self.cube_spectra_folder_path, exist_ok=True)
        self.save_spectra(all_stars, all_stars_err, prog_bar)


    def save_spectra(self, spectra_df, spectra_df_err, prog_bar=tqdm):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube = SpectralCube.read(self.cube_fits, hdu=1)
        wl = [cube.world[int(wl_slice[3:]),0,0][0].value for wl_slice in spectra_df.columns]
        for (star_id, spectra_mag), (star_id2, spectra_merr) in prog_bar(
                                        zip(spectra_df.iterrows(), spectra_df_err.iterrows()), 
                                             total=spectra_df.shape[0], desc='Saving spectra'):
            spectra_flux = mag2flux(spectra_mag.values)
            spectra_flux_sig = mag2flux(spectra_mag.values - spectra_merr.values) - spectra_flux
            stardata = pd.DataFrame({"wavelength":wl, "flux":spectra_flux, "flux_sig":spectra_flux_sig})
            stardata.to_csv(os.path.join(self.cube_spectra_folder_path,
                                         f"id{star_id}.csv"), index=False)
    
    # SECCION DAOPHOT
    def save_opt_daophot(self, folder_path):
        file_content = f"""re = {self.daophot_dict['re']}
ga = {self.daophot_dict['ga']}
lo = {self.daophot_dict['lo']}
hi = {self.daophot_dict['hi']}
fw = {self.daophot_dict['fw']}
th = {self.daophot_dict['th']}
ls = {self.daophot_dict['ls']}
lr = {self.daophot_dict['lr']}
hs = {self.daophot_dict['hs']}
hr = {self.daophot_dict['hr']}
wa = {self.daophot_dict['wa']}
fi = {self.daophot_dict['fi']}
ps = {self.daophot_dict['ps']}
va = {self.daophot_dict['va']}
an = {self.daophot_dict['an']}
ex = {self.daophot_dict['ex']}
us = {self.daophot_dict['us']}
pr = {self.daophot_dict['pr']}
pe = {self.daophot_dict['pe']}
"""
        with open(os.path.join(folder_path, "daophot.opt"), 'w') as file:
            file.write(file_content)
        self.daophot_dict_path = os.path.join(folder_path, 'daophot.opt')

    def save_opt_phot(self, folder_path):
        file_content = f"""A1 = {self.phot_dict['A1']}
A2 = {self.phot_dict['A2']}
A3 = {self.phot_dict['A3']}
A4 = {self.phot_dict['A4']}
A5 = {self.phot_dict['A5']}
A6 = {self.phot_dict['A6']}
A7 = {self.phot_dict['A7']}
A8 = {self.phot_dict['A8']}
A9 = {self.phot_dict['A9']}
AA = {self.phot_dict['AA']}
AB = {self.phot_dict['AB']}
AC = {self.phot_dict['AC']}
IS = {self.phot_dict['IS']}
OS = {self.phot_dict['OS']}
"""

        with open(os.path.join(folder_path, 'photo.opt'), 'w') as file:
            file.write(file_content)
        self.phot_dict_path = os.path.join(folder_path, 'photo.opt')

    # COLUMNA ALLSTAR
    def save_opt_allstar(self, folder_path):   
        file_content = f"""fi = {self.allstar_dict['fi']}
re = {self.allstar_dict['re']}
wa = {self.allstar_dict['wa']}
pe = {self.allstar_dict['pe']}
ce = {self.allstar_dict['ce']}
cr = {self.allstar_dict['cr']}
ma = {self.allstar_dict['ma']}
pr = {self.allstar_dict['pr']}
is = {self.allstar_dict['is']}
os = {self.allstar_dict['os']}
"""
        with open(os.path.join(folder_path, 'allstar.opt'), 'w') as file:
            file.write(file_content)
        self.allstar_dict_path = os.path.join(folder_path, 'allstar.opt')

    def clean(self):
        os.chdir(cwd)
        cube_slices_folder_path = os.path.join(self.cube_fits_folder,
                                            self.cube_slices_folder)
        if cwd!=cube_slices_folder_path and os.path.exists(cube_slices_folder_path):
            shutil.rmtree(cube_slices_folder_path)

if __name__ == "__main__":
    daocube = DaoCube()
    daocube.init_cube("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/PNT3_21exp_astro_DC.fits")
    daocube.load_als("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/PNT3_21exp_astro_DC_WFC3_F814W.als")
    daocube.load_lst("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/PNT3_21exp_astro_DC_WFC3_F814W.lst")
    daocube.load_opt_file("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/daophot.opt", 
                            daocube.daophot_dict)
    daocube.load_opt_file("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/photo.opt", 
                            daocube.phot_dict)
    daocube.load_opt_file("/data/ciquezada/Projects/muse_cubes/data/test_cube_extraction/allstar.opt", 
                            daocube.allstar_dict)
    
    daocube.slice_cube()
    daocube.run_slices_photometry()
    daocube.clean()
