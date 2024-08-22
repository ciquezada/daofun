from specfun.specfun_command import SpecRVcorr, SpecCrop, SpecNaNcorr
from specfun.specfun_command import SpecTellcorr, SpecFits2df, SpecNormPoly
from specfun.specfun_command import SpecMaskedNormPoly, SpecSync, SpecSave
from specfun.specfun_command import SpecGetRV, SpecGetSNR, SpecGetSNRPP, SpecGetRVerr

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import warnings

class SpecFun:
    pipeline_options = {
                        SpecFits2df.step_name: SpecFits2df,
                        SpecRVcorr.step_name: SpecRVcorr,
                        SpecCrop.step_name: SpecCrop,
                        SpecNaNcorr.step_name: SpecNaNcorr,
                        SpecTellcorr.step_name: SpecTellcorr,
                        SpecNormPoly.step_name: SpecNormPoly,
                        SpecMaskedNormPoly.step_name: SpecMaskedNormPoly,
                        SpecSync.step_name: SpecSync,
                        SpecSave.step_name: SpecSave,
                        SpecGetRV.step_name: SpecGetRV,
                        SpecGetRVerr.step_name: SpecGetRVerr,
                        SpecGetSNR.step_name: SpecGetSNR,
                        SpecGetSNRPP.step_name: SpecGetSNRPP,
                        }

    def __init__(self):
        self.pipeline_to_do = []    # Lista de comandos instanciados
        self.pipeline_args = []
        self.indv_spectra = True
        self.spec = None
        self.spec_folder = None
        self.spec_df = None
        self.spec_file_fmt = None
        self.n_jobs = -1
        pass

    def parallel_pipeline(self, row):
        try:
            initial_spectra = pd.read_csv(os.path.join(self.spec_folder, self.spec_file_fmt.format(row["ID"])))
            to_do_list = [step() for step in self.pipeline_to_do]
            for i, item in enumerate(self.pipeline_args):
                if "spec" in item.keys():
                    if item["spec"] == "Initial Spectra":
                        to_do_list[i].spec_in = initial_spectra
                    else:
                        to_do_list[i].spec_in = to_do_list[item["spec"]]
                to_do_list[i].kwargs = item.copy()
                if "save_path" in item.keys():
                    to_do_list[i].kwargs["save_path"] = to_do_list[i].kwargs["save_path"].format(row["ID"])
                if "output_folder" in item.keys():
                    to_do_list[i].kwargs["output_folder"] = to_do_list[i].kwargs["output_folder"].format(row["ID"])
                if "vcorr_best" in item.keys():
                    if type(item["vcorr_best"]) == int:
                        to_do_list[i].RV_in = to_do_list[item["vcorr_best"]]
                    elif type(item["vcorr_best"]) == str:
                        to_do_list[i].RV_in = row[item["vcorr_best"]]
                    else:
                        to_do_list[i].RV_in = float(item["vcorr_best"])
                if "SNR" in item.keys():
                    if type(item["SNR"]) == int:
                        to_do_list[i].SNR_in = to_do_list[item["SNR"]]
                    elif type(item["SNR"]) == str:
                        to_do_list[i].SNR_in = row[item["SNR"]]
                    else:
                        to_do_list[i].SNR_in = float(item["SNR"])
                if "FeH" in item.keys():
                    if type(item["FeH"]) == int:
                        to_do_list[i].FeH_in = to_do_list[item["FeH"]]
                    elif type(item["FeH"]) == str:
                        to_do_list[i].FeH_in = row[item["FeH"]]
                    else:
                        to_do_list[i].FeH_in = float(item["FeH"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for step in to_do_list:
                    step.run_step()
            output = "OK"
        except:
            output = "ERROR"
        return(f"[{output}]: {self.spec_file_fmt.format(row['ID'])}")

    def run_pipeline(self, prog_bar=tqdm):
        if self.indv_spectra:
            initial_spectra = pd.read_csv(self.spec)
            to_do_list = [step() for step in self.pipeline_to_do]
            for i, item in enumerate(self.pipeline_args):
                if "spec" in item.keys():
                    if item["spec"] == "Initial Spectra":
                        to_do_list[i].spec_in = initial_spectra
                    else:
                        to_do_list[i].spec_in = to_do_list[item["spec"]]
                if "vcorr_best" in item.keys():
                    if type(item["vcorr_best"]) == int:
                        to_do_list[i].RV_in = to_do_list[item["vcorr_best"]]
                    else:
                        to_do_list[i].RV_in = float(item["vcorr_best"])
                if "SNR" in item.keys():
                    if type(item["SNR"]) == int:
                        to_do_list[i].RV_in = to_do_list[item["SNR"]]
                    else:
                        to_do_list[i].RV_in = float(item["SNR"])
                if "FeH" in item.keys():
                    if type(item["FeH"]) == int:
                        to_do_list[i].RV_in = to_do_list[item["FeH"]]
                    else:
                        to_do_list[i].RV_in = float(item["FeH"])
                to_do_list[i].kwargs = item
            for step in prog_bar(to_do_list, desc='Running Pipeline'):
                print(f"Iniciando {step.step_name}")
                step.run_step()
        else:
            self.spec_df = pd.read_csv(self.spec_df)
            arg_list = [[row] for i, row in self.spec_df.iterrows()]
            spectra_status = Parallel(n_jobs=self.n_jobs, verbose=False)(
                delayed(self.parallel_pipeline)(*args) for args in prog_bar(arg_list, 
                                                                    desc='Running Pipeline', total=len(arg_list)))
            with open("specfun_pipeline.log", "w") as f:
                f.write("\n".join(spectra_status))


    def init_spectra(self):
        if self.indv_spectra:
            self.spec = pd.read_csv(self.spec)
        else:
            self.spec_df = pd.read_csv(self.spec_df)


# print(specfun.pipeline_options.keys())
# new_func = specfun.pipeline_options["Crop"]()
# print(new_func.kwargs)
# Crear new_func con los parametros, 
# si el parametro es "spec" se elige el spec inicial o un SpecCmd dentro de self.pipeline_to_do con ouput_type==spec
# se guarda respectivamente el path o el key de SpecCmd con el que aparece en la pipeline en .kwargs["spec"] 
# y el spec inicial o SpecCmd en .spec_in

