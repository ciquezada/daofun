import pandas as pd
import os
from specfun.spec_rv_corr import rv_corr
from specfun.spec_crop import cut_spectrum
from specfun.spec_nan_corr import spec_uspline_corr
from specfun.spec_telluric_corr import telluric_corr
from specfun.spec_norm_poly import spec_poly_norm
from specfun.spec_maskednorm_poly import spec_poly_norm_masked
from specfun.spec_wavelength_sync import sync_wavelength
from specfun.fits2df import fits2ascii
from specfun.spec_rv import rv_measure
from specfun.spec_snr import snr_measure
from specfun.spec_snr_pp import snr_measure_pp
from specfun.spec_rv_err import rv_err_measure


class SpecCmd:
    def __init__(self):
        self._spec_in = None
        # self.kwargs = None
        # self.func = None
        # self.output_type = None
        # self.output = None

    def run_step(self, *args, **kwargs):
        self.kwargs["spec"] = self.spec_in
        if "vcorr_best" in self.kwargs.keys():
            self.kwargs["vcorr_best"] = self.RV_in
        if "SNR" in self.kwargs.keys():
            self.kwargs["SNR"] = self.SNR_in
        if "FeH" in self.kwargs.keys():
            self.kwargs["FeH"] = self.FeH_in
        self.output = self.func(*args, **self.kwargs, **kwargs)

    @property
    def spec_in(self):
        if type(self._spec_in)==pd.DataFrame:
            return self._spec_in
        elif issubclass(type(self._spec_in), SpecCmd):
            return self._spec_in.output
        else:
            return None
        
    @spec_in.setter
    def spec_in(self, spec_in):
        self._spec_in = spec_in

        


        
class SpecRVcorr(SpecCmd):
    step_name = "RVcorr"
    output_type = "spec"
    kwargs = {
                "spec": "Initial Spectra", 
                "vcorr_best": "Manual Input"
                }
    def __init__(self):
        super().__init__()
        self._RV_in = None
        self.func = rv_corr

    @property
    def RV_in(self):
        if type(self._RV_in) in [int, float]:
            return self._RV_in
        elif issubclass(type(self._RV_in), SpecCmd):
            return self._RV_in.output
        else:
            return None
        
    @RV_in.setter
    def RV_in(self, RV_in):
        self._RV_in = RV_in

class SpecCrop(SpecCmd):
    step_name = "Crop"
    output_type = "spec"
    kwargs = {
            "spec": "Initial Spectra", 
            "min_wavelength": 0.0,
            "max_wavelength": 0.0,
            }
    def __init__(self):
        super().__init__()
       
        self.func = cut_spectrum

class SpecNaNcorr(SpecCmd):
    step_name = "NaNcorr"
    output_type = "spec"
    kwargs = {
            "spec": "Initial Spectra", 
            }
    def __init__(self):
        super().__init__()
        
        self.func = spec_uspline_corr

class SpecTellcorr(SpecCmd):
    step_name = "Tellcorr"
    output_type = "spec"
    kwargs = {
                "spec": "Initial Spectra", 
                "cut_range": ("", ""), 
                "plot": False, 
                "plot_filename": "", 
                "plot_title": "", 
                }
    def __init__(self):
        super().__init__()
        
        self.func = telluric_corr

class SpecFits2df(SpecCmd):
    step_name = "Fits2df"
    output_type = "spec"
    kwargs = {
                "fits_file": "", 
                }
    
    def __init__(self):
        super().__init__()
        
        self.func = fits2ascii

class SpecNormPoly(SpecCmd):
    step_name = "NormPoly"
    output_type = "spec"
    kwargs = {
                "spec": "Initial Spectra", 
                "plot": False, 
                "save_path": "", 
                "plot_title": "", 
                }
    def __init__(self):
        super().__init__()
        
        self.func = spec_poly_norm

class SpecMaskedNormPoly(SpecCmd):
    step_name = "MaskedNormPoly"
    output_type = "spec"
    kwargs = {
                "spec": "Initial Spectra", 
                "plot": False, 
                "save_path": "", 
                "plot_title": "", 
                }
    def __init__(self):
        super().__init__()
        
        self.func = spec_poly_norm_masked

class SpecSync(SpecCmd):
    step_name = "SyncWavelength"
    output_type = "spec"
    kwargs = {
                "spec": "Initial Spectra", 
                "sync_spec": "", 
                }
    def __init__(self):
        super().__init__()
        
        self.func = sync_wavelength

class SpecSave(SpecCmd):
    step_name = "SpecSave"
    output_type = "file"
    kwargs = {
                "spec": "Initial Spectra", 
                "save_path": "{}.csv", 
                }
    def __init__(self):
        super().__init__()

    def func(self, spec, save_path, *args, **kwargs):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        spec.to_csv(save_path, index=False)

class SpecGetRV(SpecCmd):
    step_name = "GetRV"
    output_type = "RV"
    kwargs = {
                "spec": "Initial Spectra", 
                "grid": "/data/ciquezada/Projects/muse_cubes/downloads/muse_RVgrid",
                "vmin": -450.0,
                "vmax": 450.0,
                "save_plots": False,
                "save_files": False,
                "output_folder": "",
                }
    def __init__(self):
        super().__init__()
        self.func = rv_measure

class SpecGetRVerr(SpecCmd):
    step_name = "GetRV err"
    output_type = "RVerr"
    kwargs = {
                "star_type": "",
                "SNR": "Manual Input",
                "FeH": "Manual Input",
                "save_files": False,
                "output_folder": "",
                }
    def __init__(self):
        super().__init__()
        self._SNR_in = None
        self._FeH_in = None
        self.func = rv_err_measure

    @property
    def SNR_in(self):
        if type(self._SNR_in) in [int, float]:
            return self._SNR_in
        elif issubclass(type(self._SNR_in), SpecCmd):
            return self._SNR_in.output
        else:
            return None
        
    @SNR_in.setter
    def SNR_in(self, SNR_in):
        self._SNR_in = SNR_in

    @property
    def FeH_in(self):
        if type(self._FeH_in) in [int, float]:
            return self._FeH_in
        elif issubclass(type(self._FeH_in), SpecCmd):
            return self._FeH_in.output
        else:
            return None
        
    @FeH_in.setter
    def FeH_in(self, FeH_in):
        self._FeH_in = FeH_in

class SpecGetSNR(SpecCmd):
    step_name = "GetSNR"
    output_type = "SNR"
    kwargs = {
                "spec": "Initial Spectra", 
                "save_files": False,
                "output_folder": "",
                }
    def __init__(self):
        super().__init__()
        self.func = snr_measure

class SpecGetSNRPP(SpecCmd):
    step_name = "GetSNR PP"
    output_type = "SNR"
    kwargs = {
                "spec": "Initial Spectra", 
                "save_files": False,
                "output_folder": "",
                }
    def __init__(self):
        super().__init__()
        self.func = snr_measure_pp


