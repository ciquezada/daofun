from daofun.misc_tools import read_als, check_file, create_working_dir
from daofun.daophot_wraps import find, phot, pick, create_psf, sub_fits, allstar, sync_lst_als
import shutil
import os


cwd = os.getcwd()

class DaoPipe:
    pipeline_list = [
        "create_psf",
        "allstar",
        "phot",
        "refine_psf",
        "change_Mlist",
        "recenter_on",
        "recenter_off",
    ]
    def __init__(self, verbose=True):
        self.verbose = verbose
        ### PARTE EXPERIMENTAL ####
        self.fits_name = None
        self.fits_path = None
        self.fits_folder = None
        self.working_dir = None
        ###########################
        self.last_target_als = None
        self.last_target_lst = None
        self.last_target_psf = None
        self.last_sub_fits = None
        self.last_nei = None
        self.refines = 1
        self.step = 0
        self.mlist_queue = []
        #############################
        self.pipeline_dict = {
                                "create_psf": self.do_create_psf,
                                "allstar": self.do_allstar,
                                "phot": self.do_phot,
                                "refine_psf": self.refine_psf,
                                "change_Mlist": self.change_master_list,
                                "recenter_on": self.recenter_on,
                                "recenter_off": self.recenter_off,
                                }
        self.pipeline_cmd = []

   ################ EXPERIMENTAL ################
    def init_fits(self, fits_path):
        check_file(fits_path)                
        self.fits_name = os.path.splitext(os.path.basename(fits_path))[0]
        self.fits_path = os.path.abspath(fits_path)
        self.fits_folder = os.path.dirname(fits_path)
        self.working_dir = create_working_dir(fits_path) # esto hace un os.chdir

    def load_als(self, als_path):
        self.last_target_als = os.path.basename(als_path)
        shutil.copy(als_path, self.last_target_als)

    def load_lst(self, lst_path):
        self.last_target_lst = os.path.basename(lst_path)
        shutil.copy(lst_path, self.last_target_lst)
    
    def load_psf(self, psf_path):
        self.last_target_psf = os.path.basename(psf_path)
        shutil.copy(psf_path, self.last_target_psf)
        
    def load_opt_daophot(self, daophot_path):
        self.opt_daophot = daophot_path
        shutil.copy(daophot_path, os.path.basename(self.opt_daophot))

    def load_opt_phot(self, phot_path):
        self.opt_phot = phot_path
        shutil.copy(phot_path, os.path.basename(self.opt_phot))
    
    def load_opt_allstar(self, allstar_path):
        self.opt_allstar = allstar_path
        shutil.copy(allstar_path, os.path.basename(self.opt_allstar))

    #######################################################################
    
    def do_create_psf(self):
        out_psf = f"{self.fits_name}.psf_step{self.step}"
        out_nei = f"{self.fits_name}.nei_step{self.step}"
        create_psf(os.path.basename(self.fits_path), self.last_target_als, 
                    self.last_target_lst, out_psf, out_nei, verbose=self.verbose)
        self.last_target_psf = out_psf
        self.last_nei = out_nei
    
    def do_allstar(self):
        out_als = f"{self.fits_name}.als_step{self.step}"
        out_sub_fits = f"{self.fits_name}s{self.step}.fits"
        allstar(os.path.basename(self.fits_path), self.last_target_psf, self.last_target_als, 
                out_als, out_sub_fits, verbose=self.verbose)
        self.last_target_als = out_als

    def do_phot(self):
        out_als = f"{self.fits_name}.als_step{self.step}"
        phot(os.path.basename(self.fits_path), self.last_target_als, out_als, verbose=self.verbose)
        self.last_target_als = out_als

    def refine_psf(self):
        in_fits = os.path.basename(self.fits_path) if self.last_sub_fits==None else self.last_sub_fits
        out_sub_fits = f"{self.fits_name}s{self.step}.fits"
        out_psf = f"{self.fits_name}.psf_step{self.step}"
        out_nei = f"{self.fits_name}.nei_step{self.step}"
        sub_fits(in_fits, self.last_target_psf, 
                 self.last_nei, out_sub_fits, self.last_target_lst, verbose=self.verbose)
        create_psf(out_sub_fits, self.last_target_als, 
                            self.last_target_lst, out_psf, out_nei, verbose=self.verbose)
        self.last_sub_fits = out_sub_fits
        self.last_target_psf = out_psf
        self.last_nei = out_nei
        
    def change_master_list(self):
        new_als_path = self.mlist_queue.pop(0)
        new_als_name = os.path.basename(new_als_path)
        shutil.copy(new_als_path, new_als_name)
        self.last_target_als = new_als_name
        
    def recenter_on(self):
        with open("allstar.opt", 'r') as archivo:
            contenido = archivo.read()
            contenido_modificado = contenido.replace("re = 0", "re = 1")

        with open("allstar.opt", 'w') as archivo:
            archivo.write(contenido_modificado)

    def recenter_off(self):
        with open("allstar.opt", 'r') as archivo:
            contenido = archivo.read()
            contenido_modificado = contenido.replace("re = 1", "re = 0")

        with open("allstar.opt", 'w') as archivo:
            archivo.write(contenido_modificado)      

    def run_photometry(self):
        for cmd in self.pipeline_cmd:
            self.step += 1
            self.pipeline_dict[cmd]()

    def save_photometry(self):
        shutil.copy(self.last_target_als, os.path.join(self.fits_folder, f"{self.fits_name}.als"))

    def clean(self):
        os.chdir(cwd)
        if self.working_dir!=cwd and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

        

if __name__ == "__main__":
    daopipe = DaoPipe(verbose=False)
    daopipe.init_fits("/data/ciquezada/Projects/muse_cubes/data/test_photometry_2/PNT3_21exp_astro_DC_WFC3_F814W.fits")
    daopipe.load_opt_daophot("/data/ciquezada/Projects/muse_cubes/data/photometry_test/photometry/814_psf2_17/daophot.opt")
    daopipe.load_opt_phot("/data/ciquezada/Projects/muse_cubes/data/photometry_test/photometry/814_psf2_17/photo.opt")
    daopipe.load_opt_allstar("/data/ciquezada/Projects/muse_cubes/data/photometry_test/photometry/814_psf2_17/allstar.opt")
    daopipe.load_als("/data/ciquezada/Projects/muse_cubes/data/photometry_test/photometry/814_psf2_17/PNT3_21exp_astro_DC_WFC3_F814W.als")
    daopipe.load_lst("/data/ciquezada/Projects/muse_cubes/data/photometry_test/photometry/814_psf2_17/PNT3_21exp_astro_DC_WFC3_F814W.lst")
    daopipe.pipeline_cmd += ["create_psf",  "allstar",
                             "create_psf", "refine_psf", "allstar"]
    daopipe.run_photometry()
    daopipe.save_photometry()
    daopipe.clean()




