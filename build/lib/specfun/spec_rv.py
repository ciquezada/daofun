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

def rv_measure(spec, grid, vmin, vmax, save_plots, save_files, output_folder, verbose=False):
    vlight = 2.997925e5
    df_obs = spec
    rin = None
    rout = None
    specout_dat_path = os.path.join(output_folder, "spec_vcorr.dat")
    specout_fits_path = os.path.join(output_folder, "spec_vcorr.fits")
    specout_rvdat_path = os.path.join(output_folder, "spec_RV.csv")
    plot_out_path = os.path.join(output_folder, "spec_fit.png")


    # Cargar espectro observado desde archivo CSV
    # df_obs = pd.read_csv(infile)
    wave = df_obs["wavelength"].values
    flux = df_obs["flux"].values
    flux_sig = df_obs["flux_sig"].values



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


    # Calculate the minimum and maximum velocity possible to calculate with the provided grid
    vlight = 2.997925e5
    dif_range = wave_sint[-1]-wave_sint[0]-(wave[-1]-wave[0])
    if dif_range<=0.0:
        if verbose:
            print("[Error] Wavelength coverage of template is too short")
            print(f"\tTemplate/range: {templates_list[i]}/{wave_sint[-1]-wave_sint[0]}")
            print(f"\tObs spectrum: infile")
            print(wave[-1]-wave[0])
        sys.exit()
    v_min =  vlight*(wave[-1]/wave_sint[-1]-1.0)+4.0
    v_max =  vlight*(wave[0]/wave_sint[0]-1.0)-4.0
    if (vmin!=None):
        v_min=vmin
    if (vmax!=None):
        v_max=vmax
    if (vmin==None) & (vmax==None):
        if verbose:
            print(f"[Info] Provided grid allows to compute velocity shifts in({v_min:8.2f},{v_max:8.2f})")
    if (vmin!=None) & (vmax!=None):
        if verbose:
            print(f"[Info] Inputs allow to compute velocity shifts in({v_min:8.2f},{v_max:8.2f})")


    #  Attempt level-off normalization of input spectrum  --> by-hand recipe, look for smth better!!!
    median_flux_obs=np.percentile(flux[flux>0.7],75)

    # Define regions to compute cross correlation
    if (rin!=None):
        regions = np.genfromtxt(rin)
        if np.shape(regions)==(2,):
            regions=[regions]
    else:
        regions = np.array([[np.min(wave),np.max(wave)]])

    # Define regions to exclude from the analysis
    imask_in=np.isfinite(wave)
    if (rout!=None):
        regions_out = np.genfromtxt(rout)
        if np.shape(regions_out)==(2,):
            regions_out=[regions_out]
        for region in regions_out:
            i_mask =  (wave>=region[0]) & (wave<=region[1]) 
            imask_in = imask_in & ~i_mask
    flux_masked = np.where(imask_in,flux,0.0)

    #  Make a preliminary cross correlation against a single template (choosen randomly)
    #  to roughly set the observed spectrum into rest frame
    i_prelim_tpl = np.random.randint(len(templates_list))
    vels=[]
    vvs=[]
    ccfs=[]
    for region in regions:
        ireg = (wave>=region[0]) & (wave<=region[1]) 
        vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,fluxes_sint[i_prelim_tpl],v_min,v_max)
        vcorr_prelim = vv[np.argmax(ccf)]
        vels.append(vcorr_prelim)
        vvs.append(vv)
        ccfs.append(ccf)
    vcorr_prelim = np.median(vels)
    if verbose:
        print(f"[Info] Preliminary Vobs:{vcorr_prelim:8.2f}")


    # START PLOT
    ##
    #   Make a diagnostic plot of the whole process
    #
    if save_plots:
        fig = plt.figure(1,figsize=(12,7),facecolor="white")
        fig.subplots_adjust(left=0.065,bottom=0.07,right=0.98,top=0.98,hspace=0.0,wspace=0.0)
        gs1 = gridspec.GridSpec(76, 40)
        fig.canvas.set_window_title('Radial velocity correction [infile]')
        mpl.rcParams.update({'font.size': 10})

        # Plot first Vcorr
        ax1=fig.add_subplot(gs1[0:30,0:18])
        ax1.set_xlim(v_min,v_max)
        ax1.set_xlabel("RV (km/s)")
        ax1.set_ylabel("Cross Corr Norm.")
        for i in range(len(vvs)):
            ax1.plot(vvs[i], ccfs[i]/np.max(ccfs[i]),color="olivedrab")
            v_label = v_min+0.07*(v_max-v_min) 
            i_label=np.argmin(np.absolute(v_label-vvs[i]))
            ccfy_label = ccfs[i][i_label]/np.max(ccfs[i])
            ax1.text(v_label,ccfy_label,str(i),color="black",fontsize=10)
            ax1.axvline(x=vels[i],ls="--",lw=0.6,color="black")
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        xl=ax1.get_xlim()
        yl=ax1.get_ylim()
        ax1.text(xl[0]+0.7*(xl[1]-xl[0]),yl[0]+0.85*(yl[1]-yl[0]),"Cross-correlation\nguess template",fontsize=11)

    # END PLOT


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
    if (rout!=None):
        regions_out = np.genfromtxt(rout)
        if np.shape(regions_out)==(2,):
            regions_out=[regions_out]
        for region in regions_out:
            i_mask =  (wave_sint[i_cm]>=region[0]*fcor) & (wave_sint[i_cm]<=region[1]*fcor) 
            imask_in_corr = imask_in_corr & ~i_mask

    i_chi = np.zeros(len(wave_sint[i_cm]))
    for region in regions:
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



    #  Make correlation with the best template
    vels=[]
    vvs=[]
    ccfs=[]
    for region in regions:
        ireg = (wave>=region[0]) & (wave<=region[1])
        vv,ccf = cross_correlate(wave[ireg],flux_masked[ireg]/median_flux_obs,wave_sint,fluxes_sint[i_best_hit],vmin=vcorr_prelim-35,vmax=vcorr_prelim+35,deltaV=0.025)
        vcorr_best_discrete = vv[np.argmax(ccf)]
        vels.append(vcorr_best_discrete)
        vvs.append(vv)
        ccfs.append(ccf)
    vcorr_best_discrete = np.median(vels)
    vvs=np.array(vvs)
    ccfs=np.array(ccfs)

    ##
    #   Luego de probar algunos metodos para encontrar el maximo con mas exactitud, concluyo que lo mejor es
    #   minimizar una funcion spline cubica construida a partir del 20% mayor de la ccf. El fit de Gaussiana 
    #   no resulta tan bien porque incluso la parte superior de la ccf puede ser levemente asimetrica
    #
    vels_best=np.empty(len(vels))
    for i in range(len(vels)):
        ccf_norm = ccfs[i]/np.max(ccfs[i])
        ii = ccf_norm > np.percentile(ccf_norm,85)
        spl_ccf = InterpolatedUnivariateSpline(vvs[i][ii],ccf_norm[ii],k=3)
        spl_curve = spl_ccf(np.linspace(np.min(vvs[i][ii]),np.max(vvs[i][ii]),1000))
        fm = lambda x: -spl_ccf(x)
        vcorr_minimized = opt.minimize_scalar(fm, bounds=(np.min(vvs[i][ii]),np.max(vvs[i][ii])),method="bounded")
        vcorr_best=vcorr_minimized.x
        vels_best[i]=vcorr_best
    vcorr_best = np.median(vels_best)
    err_vcorr_best = np.std(vels_best)
    if verbose:
        print(f"[Info] ccf Vobs:{vcorr_best_discrete:9.3f}")
        print(f"[Info] Final Vobs:{vcorr_best:9.3f}")
    if len(regions)>1:
        if verbose:
            print(f"[Info] Vobs Err:{err_vcorr_best:9.3f}")

    fcor = np.sqrt( (1.0-vcorr_best/vlight) / (1.0+vcorr_best/vlight) )
    wave_best = wave*fcor


    # START PLOT
    # Compute template autocorrelation
    if save_plots:
        vv0,ccf0 = cross_correlate(wave_sint[i_cm],fluxes_sint[i_best_hit][i_cm],wave_sint,fluxes_sint[i_best_hit],vmin=-35,vmax=35, deltaV=0.1)

        # Plot best Vcorr with all correlation functions
        ax2=fig.add_subplot(gs1[0:30,22:40])
        ax2.set_xlabel("RV (km/s)")
        ax2.set_ylabel("Cross Corr Norm.")
        for i in range(len(ccfs)):
            ax2.plot(vvs[i], ccfs[i]/np.max(ccfs[i]),color="olivedrab",label="Cross-correlation")
        handles, labels = ax2.get_legend_handles_labels()
        handles_gral=[]
        labels_gral=[]
        handles_gral.append(handles[0])
        labels_gral.append(labels[0])
        ax2.plot(vv0+vcorr_best, ccf0/np.max(ccf0),color="black",ls="--",label="Auto-correlation\ntemplate")
        handles, labels = ax2.get_legend_handles_labels()
        handles_gral.append(handles[len(ccfs)])
        labels_gral.append(labels[len(ccfs)])
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ccf_mins=[]
        for i in range(len(ccfs)):
            ccf_mins.append(np.min((np.min(ccfs[i]/np.max(ccfs[i])),np.min(ccf0/np.max(ccf0)))))
        yplotmin=np.min(ccf_mins)
        yl_i = yplotmin-0.05*(1.0-yplotmin)
        yl_s = 1.0+0.1*(1.0-yplotmin)
        ax2.set_ylim(yl_i,yl_s)
        xl=ax2.get_xlim()
        yl=ax2.get_ylim()
        ax2.legend(handles_gral,labels_gral,loc=2,frameon=False,fontsize=11)
        #ax2.text(xl[0]+0.68*(xl[1]-xl[0]),yl[0]+0.85*(yl[1]-yl[0]),"Cross-correlation\nbest template")
        ax2.text(xl[0]+0.60*(xl[1]-xl[0]),yl[0]+0.7*(yl[1]-yl[0]),"Vobs =%9.3f km/s"%vcorr_best,color="tomato",fontsize=11)
        if len(regions)>1:
            ax2.text(xl[0]+0.60*(xl[1]-xl[0]),yl[0]+0.63*(yl[1]-yl[0]),"Err Vobs =%9.3f km/s"%err_vcorr_best,color="tomato",fontsize=11)



        # Plot corrected spectrum vs template
        ax3=fig.add_subplot(gs1[36:75,0:40])
        ax3.set_xlabel("lambda (Angstrom)")
        ax3.set_ylabel("Normalized flux")
        #ax3.set_xlim(np.min(wave_sint),np.max(wave_sint))
        ax3.set_xlim(np.min(wave_best),np.max(wave_best))
        minx = np.min([np.min(flux/median_flux_obs)+0.04,0.7])
        ax3.set_ylim(minx,1.15)
        ax3.plot(wave_best,flux/median_flux_obs,lw=0.5,color="black",label="Observed")
        ax3.plot(wave_sint,fluxes_sint[i_best_hit],lw=0.5,color="red",label="Template")
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

        # Highlight used regions except if whole spectrum is used
        if len(regions)>1:
            for i,region in enumerate(regions):
                ax3.fill_between(np.linspace(region[0],region[1],1000),yl[0],yl[1],lw=0,color="cornflowerblue",alpha=0.4)
                ax3.text(region[0],yl[0]+0.90*(yl[1]-yl[0]),str(i),color="black",fontsize=10)

        # Highlight excluded regions
        if (rout!=None):
            regions_out = np.genfromtxt(rout)
            if np.shape(regions_out)==(2,):
                regions_out=[regions_out]
            for region in regions_out:
                ax3.fill_between(np.linspace(region[0]*fcor,region[1]*fcor,500),yl[0],yl[1],lw=0,color="tomato",alpha=0.4)
        ax3.set_xlim(xl)
        ax3.set_ylim(yl)


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
        #crval1_new = hdulist_obs[0].header["crval1"]*fcor
        #cdelt1_new = hdulist_obs[0].header["cdelt1"]*fcor
        #if "cd1_1" in hdulist_obs[0].header.keys():
            #cd1_1_new = hdulist_obs[0].header["cd1_1"]*fcor
        #hdulist_obs.close()

        # salida=open(specout_dat_path,"w")
        # for x,y in zip(wave_best,flux):
        #     salida.write(str(x)+" "+str(y)+"\n")
        # salida.close()


        # if verbose:# #
        #   print("=========")
        #   print(wave_best[0])
        #   print(wave_best[-1]-wave_best[-2])
        #   print(wave_best[1]-wave_best[0])
        #   print(wave_best[600]-wave_best[599])
        # #print{"========="}()

        # crval1_new = wave_best[0]
        # cdelt1_new = hdulist_obs[0].header["cdelt1"]*fcor
        # crpix1_new = 1
        # cd1_1_new = cdelt1_new
        # #if "cd1_1" in hdulist_obs[0].header.keys():
        #     #cd1_1_new = hdulist_obs[0].header["cd1_1"]*fcor


        # hdulist_obs.close()



        # data, header = pf.getdata(infile, header=True)
        # header['crval1']=crval1_new
        # header['cdelt1']=cdelt1_new
        # header['crpix1']=crpix1_new
        # if "LTV1" in hdulist_obs[0].header.keys():
        #     header['LTV1']=crpix1_new
        # if "cd1_1" in hdulist_obs[0].header.keys():
        #     header['cd1_1']=cd1_1_new
        # header['DopCor']=(vcorr_best,'Vrad correction applied')
        # spec_out = specout_fits_path
        # pf.writeto(spec_out, data, header, overwrite=True)
        # if verbose:# 
        #   print(f"[Info] Corrected spectrum written in: {spec_out}")

        # NEW SPEC END

        # START RV_DAT OUT


        # salida=open(specout_rvdat_path,"w")
        # if len(regions)==1:
        #     linsal = "%25s  %9.3f %9.3f %5.5f  %6.1f  %5.2f  %5.2f  %5.2f \n" % (os.path.basename(output_folder),vcorr_best,np.nan, chi2[i_best_hit], params_list[i_best_hit][0],params_list[i_best_hit][1],params_list[i_best_hit][2],params_list[i_best_hit][3])
        #     salida.write(linsal)
        # else:
        #     linsal = "%25s  %9.3f %9.3f %5.5f  %6.1f  %5.2f  %5.2f  %5.2f \n" % (os.path.basename(output_folder),vcorr_best,err_vcorr_best, chi2[i_best_hit], params_list[i_best_hit][0],params_list[i_best_hit][1],params_list[i_best_hit][2],params_list[i_best_hit][3])
        #     salida.write(linsal)
        # salida.close()


        salida = {"filn":[os.path.basename(output_folder)], "rv":[vcorr_best], "rv_chi":[chi2[i_best_hit]]}
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