#!/usr/bin/env python
# -*-encoding:utf-8 -*-

r'''
    RVCORR
      Computes and correct doppler shift of a given spectrum using a
      grid of synthetic spectra
'''

import matplotlib as mpl
mpl.rcParams['font.family']="serif"
mpl.rcParams['axes.linewidth'] = 0.85
from matplotlib.ticker import AutoMinorLocator
import astropy.io.fits as pf
from astropy.wcs import WCS
import numpy as np
import matplotlib.pylab as plt
import sys
import matplotlib.gridspec as gridspec
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import argparse
import os
import scipy.optimize as opt
import pandas as pd
import pymc as pm
import aesara.tensor as at
from scipy.stats import gaussian_kde
import seaborn as sns


def oneGauss(x, A, mu, sig):
    zg = (x - mu)**2 / sig**2
    gg = A * np.exp(-zg / 2.0)
    return gg

def cross_correlate(wobs, fobs, wsin, fsin, vmin, vmax, deltaV=0.5):
    vlight = 2.997925e5
    vels = np.arange(vmin, vmax, deltaV)
    ccf = np.zeros(len(vels))
    for i, vel in enumerate(vels):
        factor = np.sqrt((1.0 + vel / vlight) / (1.0 - vel / vlight))
        f_sin_shift = interp1d(wsin * factor, fsin, bounds_error=True)
        ccf[i] = np.sum(fobs * f_sin_shift(wobs))
    return vels, ccf

def load_templates(grid, verbose):
    # Leer lista de archivos fits de grilla de espectros sinteticos
    files_list = os.listdir(grid)
    templates_list = [elem for elem in files_list if ".fits" in elem]

    if len(templates_list) == 0:
        if verbose:
            print("\n[Error] No templates available in folder " + grid + "\n")
        sys.exit()
    else:
        if verbose:
            print("[Info] Number of templates: ", len(templates_list))

    # Resto del código permanece sin cambios...
    # Store in lists the fluxes and parameters
    # of files in the grid
    fluxes_sint = [None]*len(templates_list)
    params_list = [None]*len(templates_list)
    for i in range(len(templates_list)):
        if i==0:
            flux_sint, header_sint = pf.getdata(grid+"/"+templates_list[i], header=True)
            wcs = WCS(header_sint)
            index = np.arange(header_sint['NAXIS1'])
            wavelength = wcs.wcs_pix2world(index[:,np.newaxis], 0)
            wave_sint = wavelength.flatten()
            fluxes_sint[i]=flux_sint
            
        else:
            flux_sint,header_sint = pf.getdata(grid+"/"+templates_list[i], header=True)
            fluxes_sint[i]=flux_sint
        
        # Ver si hay parametros fisicos anotados en el header
        hkeys = header_sint.keys()
        hparams=[np.nan]*4
        if "TEFF" in hkeys:
            hparams[0]=float(header_sint["TEFF"])
        if "LOGG" in hkeys:
            hparams[1]=float(header_sint["LOGG"])
        if "MH" in hkeys:
            hparams[2]=float(header_sint["MH"])
        if "ALFE" in hkeys:
            hparams[3]=float(header_sint["ALFE"])
        params_list[i]=hparams

    return templates_list, fluxes_sint, wave_sint, params_list

def vmin_vmax(templates_list, wave, wave_sint, vmin, vmax):
	# Calculate the minimum and maximum velocity possible to calculate with the provided grid
	vlight = 2.997925e5
	dif_range = wave_sint[-1]-wave_sint[0]-(wave[-1]-wave[0])
	if dif_range<=0.0:
		print("[Error] Wavelength coverage of template is too short")
		print(f"\tTemplate/range: {templates_list[-1]}/{wave_sint[-1]-wave_sint[0]}")
		print(f"\tObs spectrum: infile")
		print(wave_sint[-1]-wave_sint[0], " < ", wave[-1]-wave[0])
		sys.exit()
	v_min =  vlight*(wave[-1]/wave_sint[-1]-1.0)+4.0
	v_max =  vlight*(wave[0]/wave_sint[0]-1.0)-4.0
	if (vmin!=None):
		v_min=vmin
	if (vmax!=None):
		v_max=vmax
	if (vmin==None) & (vmax==None):
		print(f"[Info] Provided grid allows to compute velocity shifts in({v_min:8.2f},{v_max:8.2f})")
	if (vmin!=None) & (vmax!=None):
		print(f"[Info] Inputs allow to compute velocity shifts in({v_min:8.2f},{v_max:8.2f})")
	return v_min, v_max

def find_vcorr_prelim(templates_list, wave, flux_masked, region, median_flux_obs, 
                                                wave_sint, fluxes_sint, v_min, v_max, verbose):
    #  Make a preliminary cross correlation against a single template (choosen randomly)
    #  to roughly set the observed spectrum into rest frame
    i_prelim_tpl = np.random.randint(len(templates_list))
    ireg = (wave>=region[0]) & (wave<=region[1]) 
    vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,fluxes_sint[i_prelim_tpl],v_min,v_max)
    vcorr_prelim = vv[np.argmax(ccf)]
    if verbose:
        print(f"[Info] Preliminary Vobs:{vcorr_prelim:8.2f}")
    return vcorr_prelim, vv, ccf

def find_best_template(wave, flux, wave_sint, fluxes_sint, median_flux_obs, vcorr_prelim, templates_list, vlight, region, verbose):
    # Calculate chisquare of obseved rv corrected spectra against grid
    #  We consider only user defineded regions, discarding those indicated 
    fcor = np.sqrt( (1.0-vcorr_prelim/vlight) / (1.0+vcorr_prelim/vlight) )
    wave_corr = wave*fcor
    w1 = np.max((np.min(wave_corr),np.min(wave_sint)))
    w2 = np.min((np.max(wave_corr),np.max(wave_sint)))
    i_cm = (wave_sint>=w1) & (wave_sint<=w2)
    spl = InterpolatedUnivariateSpline(wave_corr, flux,k=3) # interpolate obs spec in common range with template
    flux_obs_corr_int = spl(wave_sint[i_cm])/median_flux_obs

    # Define regions to exclude from the analysis considering the fcor 
    imask_in_corr=np.isfinite(wave_sint[i_cm])

    i_chi = np.zeros(len(wave_sint[i_cm]))
    icond = (wave_sint[i_cm]>=region[0]*fcor) & (wave_sint[i_cm]<=region[1]*fcor)
    i_chi = i_chi + icond
    i_chi = np.array(i_chi,dtype=bool) & imask_in_corr

    chi2=[None]*len(templates_list)
    for k in range(len(fluxes_sint)):
        chi2[k] = np.sum( (flux_obs_corr_int[i_chi]-fluxes_sint[k][i_cm][i_chi])**2) / (np.sum(flux_obs_corr_int[i_chi])-3.0)
    chi2=np.array(chi2)
    i_best_hit = np.argmin(chi2)
    if verbose:
        print(f"[Info] Best template chisquare: {chi2[i_best_hit]}")
    return i_best_hit, i_cm, chi2[i_best_hit]

def vcorr_maxccf_template(wave, flux_masked, region, median_flux_obs, wave_sint, flux_sint, vcorr_prelim, verbose):
    #  Make correlation with the best template
    ireg = (wave>=region[0]) & (wave<=region[1])
    vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,flux_sint,vmin=vcorr_prelim-35,vmax=vcorr_prelim+35,deltaV=0.025)
    vcorr_best_discrete = vv[np.argmax(ccf)]
    vv=np.array(vv)
    ccf=np.array(ccf)
    ##
    #   Luego de probar algunos metodos para encontrar el maximo con mas exactitud, concluyo que lo mejor es
    #   minimizar una funcion spline cubica construida a partir del 20% mayor de la ccf. El fit de Gaussiana 
    #   no resulta tan bien porque incluso la parte superior de la ccf puede ser levemente asimetrica
    #
    ccf_norm = ccf/np.max(ccf)
    ii = ccf_norm > np.percentile(ccf_norm,85)
    spl_ccf = InterpolatedUnivariateSpline(vv[ii],ccf_norm[ii],k=3)
    spl_curve = spl_ccf(np.linspace(np.min(vv[ii]),np.max(vv[ii]),1000))
    fm = lambda x: -spl_ccf(x)
    vcorr_minimized = opt.minimize_scalar(fm, bounds=(np.min(vv[ii]),np.max(vv[ii])),method="bounded")
    vcorr_best=vcorr_minimized.x
    if verbose:
        print(f"[Info] ccf Vobs:{vcorr_best_discrete:9.3f}")
        print(f"[Info] Final Vobs:{vcorr_best:9.3f}")

    return vcorr_best, vv, ccf

def rv_measure(spec, grid, vmin, vmax, save_plots, save_files, output_folder, verbose=False):
    vlight = 2.997925e5
    df_obs = spec
    rin = None
    rout = None
    specout_dat_path = os.path.join(output_folder, "spec_vcorr.dat")
    specout_fits_path = os.path.join(output_folder, "spec_vcorr.fits")
    specout_rvdat_path = os.path.join(output_folder, "spec_RV.csv")
    plot_out_path = os.path.join(output_folder, "spec_fit.png")


    def aesara_interpolate(x_new, x, y):
        # Busca los índices en los que se deberían insertar los nuevos valores de x_new en x
        idx = at.extra_ops.searchsorted(x, x_new, side="left") - 1
        idx = at.clip(idx, 0, x.shape[0] - 2)
        
        # Selecciona los valores de x e y correspondientes a estos índices
        x0 = at.take(x, idx)
        x1 = at.take(x, idx + 1)
        y0 = at.take(y, idx)
        y1 = at.take(y, idx + 1)
        
        # Calcula la pendiente y realiza la interpolación lineal
        slope = (y1 - y0) / (x1 - x0)
        y_new = y0 + slope * (x_new - x0)
        return y_new

    # Cargar espectro observado desde archivo CSV
    # df_obs = pd.read_csv(infile)
    wave = df_obs["wavelength"].values
    flux = df_obs["flux"].values
    flux_sig = df_obs["flux_sig"].values

    # print("ejecutando: load_templates")
    templates_list, fluxes_sint, wave_sint, params_list = load_templates(grid, verbose)
    v_min, v_max = vmin_vmax(templates_list, wave, wave_sint, vmin, vmax)

    #  Attempt level-off normalization of input spectrum  --> by-hand recipe, look for smth better!!!
    median_flux_obs=np.percentile(flux[flux>0.7],75)

    # Define regions to compute cross correlation
    regions = np.array([[np.min(wave),np.max(wave)]])
    region = [np.min(wave),np.max(wave)]

    # Define regions to exclude from the analysis
    imask_in=np.isfinite(wave)
    flux_masked = np.where(imask_in,flux,0.0) # a partir de aqui trabajamos con flux_masked

    ireg = (wave>=region[0]) & (wave<=region[1])
    imask_in_wave = np.isfinite(wave)
    wave_ireg, flux_ireg, flux_sig_ireg = (wave[ireg], flux_masked[ireg]/median_flux_obs, flux_sig[ireg])

    #  Make a preliminary cross correlation against a single template (choosen randomly)
    #  to roughly set the observed spectrum into rest frame
    vcorr_prelim, vv, ccf = find_vcorr_prelim(templates_list, wave, flux_masked, region, median_flux_obs, 
                                                wave_sint, fluxes_sint, v_min, v_max, verbose)
    print("vcorr_prelim: ", vcorr_prelim)


    #  Best template is found
    #  Calculate chisquare of obseved rv corrected spectra against grid
    #  We consider only user defineded regions, discarding those indicated 
    i_best_hit, i_cm, chi2 = find_best_template(wave, flux, wave_sint, fluxes_sint, median_flux_obs, vcorr_prelim, templates_list, vlight, region, verbose)
    flux_sint = fluxes_sint[i_best_hit]
    # spl = InterpolatedUnivariateSpline(wave_sint, flux_sint, k=3) # interpolate obs spec in common range with template

    # Use PyMC to infer the best template and velocity correction
    with pm.Model() as model:
        # Velocity correction as a uniform prior 
        # between v_min and v_max
        vcorr = pm.Uniform('vcorr', lower=vcorr_prelim-200, upper=vcorr_prelim+200)
        # vcorr = pm.Normal('vcorr', mu=vcorr_prelim, sigma=200)

        # Model the observed flux as a function of the selected template and vcorr
        fcor = pm.math.sqrt((1.0 - vcorr / vlight) / (1.0 + vcorr / vlight))
        wave_corr = wave_ireg * fcor

        # Interpolating the template to the corrected wavelengths
        # Realizar la interpolación del flujo de la plantilla para las longitudes de onda corregidas
        flux_interp = aesara_interpolate(wave_corr, wave_sint, flux_sint)

        # Likelihood function: observed flux follows a normal distribution around the interpolated template flux
        # likelihood = pm.Normal('obs', mu=flux_interp, sigma=flux_sig_ireg, observed=flux_ireg)
        pm.Poisson('y', mu=flux_interp, observed=flux_ireg)

        # Perform sampling
        trace = pm.sample(5000, tune=3000, return_inferencedata=False, cores=6)


    # def parameter_sigma(trace, hdi_prob=0.95):
    #     lower = np.percentile(trace, (1 - hdi_prob) / 2 * 100, axis=0)
    #     upper = np.percentile(trace, (1 + hdi_prob) / 2 * 100, axis=0)
    #     return np.mean([np.mean(trace)-lower, upper-np.mean(trace)])

    vcorr_best = np.median(trace['vcorr'])
    vcorr_best_err = np.std(trace['vcorr'])# * 1.253

    #  Best velocity correction with best template
    #  Make correlation with the best template
    # vcorr_best, vv_best, ccf_best = vcorr_maxccf_template(wave, flux_masked, region, median_flux_obs, wave_sint, flux_sint, vcorr_prelim, verbose)

    fcor = np.sqrt( (1.0-vcorr_best/vlight) / (1.0+vcorr_best/vlight) )
    wave_best = wave*fcor
    # Compute template autocorrelation
    # vv0,ccf0 = cross_correlate(wave_sint[i_cm],flux_sint[i_cm],wave_sint,flux_sint,vmin=-35,vmax=35, deltaV=0.1)


    # START PLOT
    ##
    #   Make a diagnostic plot of the whole process
    #
    if save_plots:

        # (vv, ccf, vv0, ccf0, vcorr_prelim, vcorr_best, 
        #             v_min, v_max,
        #               wave_best, flux, median_flux_obs, 
        #                 wave_sint, flux_sint, 
        #                     templates_list, i_best_hit, params_list)

        fig = plt.figure(1,figsize=(12,7),facecolor="white")
        fig.subplots_adjust(left=0.065,bottom=0.07,right=0.98,top=0.98,hspace=0.0,wspace=0.0)
        gs1 = gridspec.GridSpec(76, 40)
        # fig.canvas.set_window_title('Radial velocity correction [infile]')
        mpl.rcParams.update({'font.size': 10})

        # Plot first Vcorr
        ax1=fig.add_subplot(gs1[0:30,0:18])
        # ax1.set_xlim(v_min,v_max)
        ax1.set_xlabel("RV (km/s)")
        ax1.set_ylabel("Cross Corr Norm.")
        ax1.plot(vv, ccf/np.max(ccf),color="olivedrab")
        v_label = v_min+0.07*(v_max-v_min) 
        i_label=np.argmin(np.absolute(v_label-vv))
        ccfy_label = ccf[i_label]/np.max(ccf)
        ax1.text(v_label,ccfy_label,str(0),color="black",fontsize=10)
        ax1.axvline(x=vcorr_prelim,ls="--",lw=0.6,color="black")
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        xl=ax1.get_xlim()
        yl=ax1.get_ylim()
        ax1.text(xl[0]+0.7*(xl[1]-xl[0]),yl[0]+0.85*(yl[1]-yl[0]),"Cross-correlation\nguess template",fontsize=11)

        # Plot best Vcorr with all correlation functions
        ax2=fig.add_subplot(gs1[0:30,22:40])

        # Trazar el histograma con KDE en el eje específico
        sns.histplot(trace['vcorr'], kde=True, ax=ax2)
        # Dibujar la distribución de vcorr como un gráfico de pasos
        # counts, bin_edges = np.histogram(trace['vcorr'], bins=30, density=True)
        # ax2.step(bin_edges[:-1], counts, where='mid', color='green')

        # Dibujar la línea vertical en la media de vcorr
        ax2.axvline(vcorr_best, color='tomato', linestyle='-', label=f'Vobs = {vcorr_best:.2f} km/s ± {vcorr_best_err:.0f} km/s')

        # Dibujar las líneas verticales para el rango de error (HDI al 95%)
        ax2.axvline(vcorr_best - vcorr_best_err, color='blue', linestyle=':')
        ax2.axvline(vcorr_best + vcorr_best_err, color='blue', linestyle=':')

        # Etiquetas y leyenda
        ax2.set_xlabel("RV (km/s)")
        ax2.set_ylabel("Probability Density")
        ax2.legend()


        # Plot corrected spectrum vs template
        ax3=fig.add_subplot(gs1[36:75,0:40])
        ax3.set_xlabel("lambda (Angstrom)")
        ax3.set_ylabel("Normalized flux")
        #ax3.set_xlim(np.min(wave_sint),np.max(wave_sint))
        ax3.set_xlim(np.min(wave_best),np.max(wave_best))
        minx = np.min([np.min(flux/median_flux_obs)+0.04,0.7])
        # ax3.set_ylim(minx,1.15)
        ax3.plot(wave_best,flux/median_flux_obs,lw=0.5,color="black",label="Observed")
        ax3.plot(wave_sint,flux_sint,lw=0.5,color="red",label="Template")
        #ax3.plot(wave_best,flux-(flux_shift-1.0),lw=0.5,color="blue",label="Residual")
        ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
        name_best=templates_list[i_best_hit]
        xl=ax3.get_xlim()
        yl=ax3.get_ylim()
        ax3.text(xl[0]+0.5*(xl[1]-xl[0]),yl[0]+0.15*(yl[1]-yl[0]),"Best template: %35s"%name_best,fontsize=11,color="maroon")
        ax3.text(xl[0]+0.5*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"Teff=%6.1f"%params_list[i_best_hit][0],fontsize=11)
        ax3.text(xl[0]+0.6*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"log(g)=%5.2f"%params_list[i_best_hit][1],fontsize=11)
        ax3.text(xl[0]+0.7*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"[M/H]=%5.2f"%params_list[i_best_hit][2],fontsize=11)
        ax3.text(xl[0]+0.8*(xl[1]-xl[0]),yl[0]+0.07*(yl[1]-yl[0]),"[a/Fe]=%5.2f"%params_list[i_best_hit][3],fontsize=11)

        # Highlight excluded regions
        ax3.set_xlim(xl)
        # ax3.set_ylim(yl)


        # Output or interactive display
        if True:
            os.makedirs(output_folder, exist_ok=True)
            fig.savefig(plot_out_path)
            plt.close(fig)
        else:
            plt.show()
    # PLOT END


    # NEW SPEC START
    ##
    #  Create  output spectrum
    #
    if save_files:
        os.makedirs(output_folder, exist_ok=True)
        salida = {"filn":[os.path.basename(output_folder)], "rv":[vcorr_best], "rv_chi":[chi2]}
        pd.DataFrame(salida).to_csv(specout_rvdat_path, index=False)
        if verbose:
            print(f"[Info] Results written in {specout_rvdat_path}\n")

        # END RV_DAT OUT

        return vcorr_best

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("InFile", help="Input spectrum to be corrected")
    parser.add_argument("Grid", help="Name of the folder containing the grid of comparison spectra")
    parser.add_argument("--vmin", help="Minimum velocity to search for", type=float)
    parser.add_argument("--vmax", help="Maximum velocity to search for", type=float)
    parser.add_argument("--save_plots", help="Save plots", action="store_true")
    parser.add_argument("--save_files", help="Save files", action="store_true")
    parser.add_argument("--output_folder", help="Output folder for saved files", type=str, default=".")
    # parser.add_argument("--rin", help="File with regions to include", type=str)
    # parser.add_argument("--rout", help="File with regions to exclude", type=str)
    
    args = parser.parse_args()
    
    infile = args.InFile
    spec = pd.read_csv(infile)

    rv = rv_measure(spec, args.Grid, args.vmin, args.vmax, args.save_plots, args.save_files, args.output_folder)

    return rv



if __name__ == "__main__":
    main()