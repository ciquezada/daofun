from daofun.misc_tools import read_lst, read_coo, read_als
from daofun.misc_tools import create_working_dir, clean_psf_lst, get_parameters_from_fits
from daofun.daophot_wraps import find, phot, pick, create_psf, sub_fits, allstar, sync_lst_als
from daofun.daofun_gui_selection import  refine_psf
from daofun.daofun_gui_inspect import  inspect_targets
from daofun.daofun_backend import DaoFun
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import argparse
import os
import PySimpleGUI as sg
import shutil
import matplotlib.pyplot as plt
plt.style.use('default')

# COLUMNA CON EL FITS
def link_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def init_main_layout(daofun):
    # LAYOUT COLUMNAS
    mainfit_viewer_column = [
                [sg.Button('Load FOV fits', key='-LOAD_FITS-'), 
                        sg.Text(text="", expand_x=True, key="-placeholder-"),
                            sg.Button('Linear', key='-SCALE_MAIN_FITS-', disabled=True, size=(5,1))],
                [sg.Text(text="", expand_x=True, key="-TEXT_MAIN_FITS-"), 
                        sg.Button('MAGerr', key='-SHOW_MAGerr-', disabled=True), 
                            sg.Button('χ²', key='-SHOW_chi2-', disabled=True), 
                                sg.Button('Sharp', key='-SHOW_sharp-', disabled=True)],
                [sg.Canvas(key="-MAIN_CANVAS-", expand_x=True, expand_y=True)],
                ]

    daophot_phot_opt_column = [ 
                [sg.Text('daophot.opt')],
                [sg.Text('Readout Noise '), sg.InputText(default_text=daofun.daophot_dict['re'], key='re', size=(12,1)), sg.Button('?', key=f'INFO_re', size=(1,1))],
                [sg.Text('Gain          '), sg.InputText(default_text=daofun.daophot_dict['ga'], key='ga', size=(12,1)), sg.Button('?', key=f'INFO_ga', size=(1,1))],
                # [sg.Text('Low Good      '), sg.InputText(default_text=daofun.daophot_dict['lo'], key='lo', size=(12,1)), sg.Button('?', key=f'INFO_lo', size=(1,1))],
                # [sg.Text('High Good     '), sg.InputText(default_text=daofun.daophot_dict['hi'], key='hi', size=(12,1)), sg.Button('?', key=f'INFO_hi', size=(1,1))],
                [sg.Text('FWHM of star  '), sg.InputText(default_text=daofun.daophot_dict['fw'], key='fw', size=(12,1)), sg.Button('?', key=f'INFO_fw', size=(1,1))],
                [sg.Text('Threshold     '), sg.InputText(default_text=daofun.daophot_dict['th'], key='th', size=(12,1)), sg.Button('?', key=f'INFO_th', size=(1,1))],
                # [sg.Text('Low Sharpness '), sg.InputText(default_text=daofun.daophot_dict['ls'], key='ls', size=(12,1)), sg.Button('?', key=f'INFO_ls', size=(1,1))],
                # [sg.Text('Low Roundness '), sg.InputText(default_text=daofun.daophot_dict['lr'], key='lr', size=(12,1)), sg.Button('?', key=f'INFO_lr', size=(1,1))],
                # [sg.Text('High Sharpness'), sg.InputText(default_text=daofun.daophot_dict['hs'], key='hs', size=(12,1)), sg.Button('?', key=f'INFO_hs', size=(1,1))],
                # [sg.Text('High Roundness'), sg.InputText(default_text=daofun.daophot_dict['hr'], key='hr', size=(12,1)), sg.Button('?', key=f'INFO_hr', size=(1,1))],
                # [sg.Text('Watch Progress'), sg.InputText(default_text=daofun.daophot_dict['wa'], key='wa', size=(12,1)), sg.Button('?', key=f'INFO_wa', size=(1,1))],
                [sg.Text('Fitting Radius'), sg.InputText(default_text=daofun.daophot_dict['fi'], key='fi', size=(12,1)), sg.Button('?', key=f'INFO_fi', size=(1,1))],
                [sg.Text('PSF Radius    '), sg.InputText(default_text=daofun.daophot_dict['ps'], key='ps', size=(12,1)), sg.Button('?', key=f'INFO_ps', size=(1,1))],
                # [sg.Text('Variable PSF  '), sg.InputText(default_text=daofun.daophot_dict['va'], key='va', size=(12,1)), sg.Button('?', key=f'INFO_va', size=(1,1))],
                # [sg.Text('Analytic PSF  '), sg.InputText(default_text=daofun.daophot_dict['an'], key='an', size=(12,1)), sg.Button('?', key=f'INFO_an', size=(1,1))],
                # [sg.Text('PSF Cleaning  '), sg.InputText(default_text=daofun.daophot_dict['ex'], key='ex', size=(12,1)), sg.Button('?', key=f'INFO_ex', size=(1,1))],
                # [sg.Text('us(DEPRECATED)'), sg.InputText(default_text=daofun.daophot_dict['us'], key='us', size=(12,1)), sg.Button('?', key=f'INFO_us', size=(1,1))],
                # [sg.Text('Profile Error '), sg.InputText(default_text=daofun.daophot_dict['pr'], key='pr', size=(12,1)), sg.Button('?', key=f'INFO_pr', size=(1,1))],
                # [sg.Text('Percent Error '), sg.InputText(default_text=daofun.daophot_dict['pe'], key='pe', size=(12,1)), sg.Button('?', key=f'INFO_pe', size=(1,1))],
                [sg.Button('Advanced OPT', key='-OPEN_DAOPHOT_ADVANCED_OPTIONS-')],
                [sg.Button('From MUSE cube', key='-LOAD_FITS_OPT-')], 
                [sg.Button('Find', key='-FIND-')],
                [sg.HSeparator()],
                [sg.HSeparator()],
                [sg.Text('photo.opt')],
                [sg.Text('Apert Radii 1 '), sg.InputText(default_text=daofun.phot_dict['A1'], key='A1', size=(12,1)), sg.Button('?', key=f'INFO_A1', size=(1,1))],
                [sg.Text('Apert Radii 2 '), sg.InputText(default_text=daofun.phot_dict['A2'], key='A2', size=(12,1)), sg.Button('?', key=f'INFO_A2', size=(1,1))],
                [sg.Text('Apert Radii 3 '), sg.InputText(default_text=daofun.phot_dict['A3'], key='A3', size=(12,1)), sg.Button('?', key=f'INFO_A3', size=(1,1))],
                # [sg.Text('Apert Radii 4 '), sg.InputText(default_text=daofun.phot_dict['A4'], key='A4', size=(12,1)), sg.Button('?', key=f'INFO_A4', size=(1,1))],
                # [sg.Text('Apert Radii 5 '), sg.InputText(default_text=daofun.phot_dict['A5'], key='A5', size=(12,1)), sg.Button('?', key=f'INFO_A5', size=(1,1))],
                # [sg.Text('Apert Radii 6 '), sg.InputText(default_text=daofun.phot_dict['A6'], key='A6', size=(12,1)), sg.Button('?', key=f'INFO_A6', size=(1,1))],
                # [sg.Text('Apert Radii 7 '), sg.InputText(default_text=daofun.phot_dict['A7'], key='A7', size=(12,1)), sg.Button('?', key=f'INFO_A7', size=(1,1))],
                # [sg.Text('Apert Radii 8 '), sg.InputText(default_text=daofun.phot_dict['A8'], key='A8', size=(12,1)), sg.Button('?', key=f'INFO_A8', size=(1,1))],
                # [sg.Text('Apert Radii 9 '), sg.InputText(default_text=daofun.phot_dict['A9'], key='A9', size=(12,1)), sg.Button('?', key=f'INFO_A9', size=(1,1))],
                # [sg.Text('Apert Radii 10'), sg.InputText(default_text=daofun.phot_dict['AA'], key='AA', size=(12,1)), sg.Button('?', key=f'INFO_AA', size=(1,1))],
                # [sg.Text('Apert Radii 11'), sg.InputText(default_text=daofun.phot_dict['AB'], key='AB', size=(12,1)), sg.Button('?', key=f'INFO_AB', size=(1,1))],
                # [sg.Text('Apert Radii 12'), sg.InputText(default_text=daofun.phot_dict['AC'], key='AC', size=(12,1)), sg.Button('?', key=f'INFO_AC', size=(1,1))],
                [sg.Text('InnerSky Radii'), sg.InputText(default_text=daofun.phot_dict['IS'], key='IS', size=(12,1)), sg.Button('?', key=f'INFO_IS', size=(1,1))],
                [sg.Text('OuterSky Radii'), sg.InputText(default_text=daofun.phot_dict['OS'], key='OS', size=(12,1)), sg.Button('?', key=f'INFO_OS', size=(1,1))],
                [sg.Button('Advanced OPT', key='-OPEN_PHOT_ADVANCED_OPTIONS-')],
                [sg.Button('PHOT', key='-PHOT-'), sg.Button('PICK', key='-PICK-')], 
                [sg.Button('NEW PSF', key='-NEW_PSF-'), sg.Button('REFINE PSF', key='-REFINE_PSF-')],
                [sg.Button('Export opt files', key='-EXPORT_OPT_FILES-')],
                [sg.HSeparator()],
                [sg.HSeparator()],
                [sg.Text('FIND sum,averaged'), sg.InputText(default_text=daofun.find_sumaver, key='find_sumaver', size=(9,1)), sg.Button('?', key=f'INFO_sumaver', size=(1,1))],
                [sg.Text('PICK min(stars,mag)'), sg.InputText(default_text=daofun.pick_minmag, key='pick_minmag', size=(7,1)), sg.Button('?', key=f'INFO_minmag', size=(1,1))],
                ]

    allstar_opt_column = [ 
                [sg.Text('allstar.opt')],
                [sg.Button('Load targets (ap file)', key='-LOAD_ALLSTAR_ALS-'), 
                                sg.Text(text="Elegir targets", expand_x=True, key="-TEXT_ALLSTAR_ALS-")],
                [sg.Button('Load PSF', key='-LOAD_ALLSTAR_PSF-'), 
                                sg.Text(text="Elegir modelo PSF", expand_x=True, key="-TEXT_ALLSTAR_PSF-")],
                [sg.Text('Fitting Radius'), sg.InputText(default_text=daofun.allstar_dict["fi"], key='a_fi', size=(12,1)), sg.Button('?', key=f'INFO_a_fi', size=(1,1))],
                # [sg.Text('Recenter Stars'), sg.InputText(default_text=daofun.allstar_dict["re"], key='a_re', size=(12,1)), sg.Button('?', key=f'INFO_a_re', size=(1,1))],
                # [sg.Text('Watch Progress'), sg.InputText(default_text=daofun.allstar_dict["wa"], key='a_wa', size=(12,1)), sg.Button('?', key=f'INFO_a_wa', size=(1,1))],
                # [sg.Text('Percent Error '), sg.InputText(default_text=daofun.allstar_dict["pe"], key='a_pe', size=(12,1)), sg.Button('?', key=f'INFO_a_pe', size=(1,1))],
                # [sg.Text('Clip Exponent '), sg.InputText(default_text=daofun.allstar_dict["ce"], key='a_ce', size=(12,1)), sg.Button('?', key=f'INFO_a_ce', size=(1,1))],
                # [sg.Text('Clip Range    '), sg.InputText(default_text=daofun.allstar_dict["cr"], key='a_cr', size=(12,1)), sg.Button('?', key=f'INFO_a_cr', size=(1,1))],
                # [sg.Text('Max Group Size'), sg.InputText(default_text=daofun.allstar_dict["ma"], key='a_ma', size=(12,1)), sg.Button('?', key=f'INFO_a_ma', size=(1,1))],
                # [sg.Text('Profile Error '), sg.InputText(default_text=daofun.allstar_dict["pr"], key='a_pr', size=(12,1)), sg.Button('?', key=f'INFO_a_pr', size=(1,1))],
                [sg.Text('InnerSky Radii'), sg.InputText(default_text=daofun.allstar_dict["is"], key='a_is', size=(12,1)), sg.Button('?', key=f'INFO_a_is', size=(1,1))],
                # [sg.Text('OuterSky Radii'), sg.InputText(default_text=daofun.allstar_dict["os"], key='a_os', size=(12,1)), sg.Button('?', key=f'INFO_a_os', size=(1,1))],
                [sg.Button('Advanced OPT', key='-OPEN_ALLSTAR_ADVANCED_OPTIONS-'), 
                    sg.Button('ALLSTAR', key='-ALLSTAR-'),
                        sg.Button('Export photometry', key='-EXPORT_PHOTOMETRY-'),
                            sg.Button('Inspect targets', key='-INSPECT_PHOTOMETRY-')],
                # [sg.Button('PHOT', key='-PHOT-'), sg.Button('PICK', key='-PICK-')], 
                # [sg.Button('NEW PSF', key='-NEW_PSF-')],
                # [sg.Button('REFINE PSF', key='-REFINE_PSF-')],
                [sg.Text(text="", expand_x=True, key="-TEXT_OUT_ALS-")],
                [sg.Canvas(key="-ALLSTAR_CANVAS-", expand_x=True, expand_y=True)],

                ]

    layout = [
            [
                sg.Column(mainfit_viewer_column, expand_y=True, expand_x=True),  # Columna izquierda
                sg.VSeparator(),
                sg.Column(daophot_phot_opt_column, size=(310, 850)),  # Columna central
                sg.VSeparator(),
                sg.Column(allstar_opt_column, expand_y=True, expand_x=True),  # Columna derecha
            ]
        ]
    
    return layout

# ADVANCED OPTIONS WINDOW
def open_daophot_advanced_options_window(daofun):
    layout = [
        [sg.Text('daophot.opt (advanced)')],
        # [sg.Text('Readout Noise '), sg.InputText(default_text=daofun.daophot_dict['re'], key='re'), sg.Button('?', key=f'INFO_re', size=(1,1))],
        # [sg.Text('Gain          '), sg.InputText(default_text=daofun.daophot_dict['ga'], key='ga'), sg.Button('?', key=f'INFO_ga', size=(1,1))],
        [sg.Text('Low Good      '), sg.InputText(default_text=daofun.daophot_dict['lo'], key='lo'), sg.Button('?', key=f'INFO_lo', size=(1,1))],
        [sg.Text('High Good     '), sg.InputText(default_text=daofun.daophot_dict['hi'], key='hi'), sg.Button('?', key=f'INFO_hi', size=(1,1))],
        # [sg.Text('FWHM of star  '), sg.InputText(default_text=daofun.daophot_dict['fw'], key='fw'), sg.Button('?', key=f'INFO_fw', size=(1,1))],
        # [sg.Text('Threshold     '), sg.InputText(default_text=daofun.daophot_dict['th'], key='th'), sg.Button('?', key=f'INFO_th', size=(1,1))],
        [sg.Text('Low Sharpness '), sg.InputText(default_text=daofun.daophot_dict['ls'], key='ls'), sg.Button('?', key=f'INFO_ls', size=(1,1))],
        [sg.Text('Low Roundness '), sg.InputText(default_text=daofun.daophot_dict['lr'], key='lr'), sg.Button('?', key=f'INFO_lr', size=(1,1))],
        [sg.Text('High Sharpness'), sg.InputText(default_text=daofun.daophot_dict['hs'], key='hs'), sg.Button('?', key=f'INFO_hs', size=(1,1))],
        [sg.Text('High Roundness'), sg.InputText(default_text=daofun.daophot_dict['hr'], key='hr'), sg.Button('?', key=f'INFO_hr', size=(1,1))],
        [sg.Text('Watch Progress'), sg.InputText(default_text=daofun.daophot_dict['wa'], key='wa'), sg.Button('?', key=f'INFO_wa', size=(1,1))],
        # [sg.Text('Fitting Radius'), sg.InputText(default_text=daofun.daophot_dict['fi'], key='fi'), sg.Button('?', key=f'INFO_fi', size=(1,1))],
        # [sg.Text('PSF Radius    '), sg.InputText(default_text=daofun.daophot_dict['ps'], key='ps'), sg.Button('?', key=f'INFO_ps', size=(1,1))],
        [sg.Text('Variable PSF  '), sg.InputText(default_text=daofun.daophot_dict['va'], key='va'), sg.Button('?', key=f'INFO_va', size=(1,1))],
        [sg.Text('Analytic PSF  '), sg.InputText(default_text=daofun.daophot_dict['an'], key='an'), sg.Button('?', key=f'INFO_an', size=(1,1))],
        [sg.Text('PSF Cleaning  '), sg.InputText(default_text=daofun.daophot_dict['ex'], key='ex'), sg.Button('?', key=f'INFO_ex', size=(1,1))],
        [sg.Text('us(DEPRECATED)'), sg.InputText(default_text=daofun.daophot_dict['us'], key='us'), sg.Button('?', key=f'INFO_us', size=(1,1))],
        [sg.Text('Profile Error '), sg.InputText(default_text=daofun.daophot_dict['pr'], key='pr'), sg.Button('?', key=f'INFO_pr', size=(1,1))],
        [sg.Text('Percent Error '), sg.InputText(default_text=daofun.daophot_dict['pe'], key='pe'), sg.Button('?', key=f'INFO_pe', size=(1,1))],
        [sg.Button('Save Advanced Options', key='-SAVE_ADVANCED_OPTIONS-')],
]

    window = sg.Window('Advanced Options', layout, finalize=True)
    return window 

def open_phot_advanced_options_window(daofun):
    layout = [
        [sg.Text('photo.opt (advanced)')],
        # [sg.Text('Apert Radii 1 '), sg.InputText(default_text=daofun.phot_dict['A1'], key='A1'), sg.Button('?', key=f'INFO_A1', size=(1,1))],
        # [sg.Text('Apert Radii 2 '), sg.InputText(default_text=daofun.phot_dict['A2'], key='A2'), sg.Button('?', key=f'INFO_A2', size=(1,1))],
        # [sg.Text('Apert Radii 3 '), sg.InputText(default_text=daofun.phot_dict['A3'], key='A3'), sg.Button('?', key=f'INFO_A3', size=(1,1))],
        [sg.Text('Apert Radii 4 '), sg.InputText(default_text=daofun.phot_dict['A4'], key='A4'), sg.Button('?', key=f'INFO_A4', size=(1,1))],
        [sg.Text('Apert Radii 5 '), sg.InputText(default_text=daofun.phot_dict['A5'], key='A5'), sg.Button('?', key=f'INFO_A5', size=(1,1))],
        [sg.Text('Apert Radii 6 '), sg.InputText(default_text=daofun.phot_dict['A6'], key='A6'), sg.Button('?', key=f'INFO_A6', size=(1,1))],
        [sg.Text('Apert Radii 7 '), sg.InputText(default_text=daofun.phot_dict['A7'], key='A7'), sg.Button('?', key=f'INFO_A7', size=(1,1))],
        [sg.Text('Apert Radii 8 '), sg.InputText(default_text=daofun.phot_dict['A8'], key='A8'), sg.Button('?', key=f'INFO_A8', size=(1,1))],
        [sg.Text('Apert Radii 9 '), sg.InputText(default_text=daofun.phot_dict['A9'], key='A9'), sg.Button('?', key=f'INFO_A9', size=(1,1))],
        [sg.Text('Apert Radii 10'), sg.InputText(default_text=daofun.phot_dict['AA'], key='AA'), sg.Button('?', key=f'INFO_AA', size=(1,1))],
        [sg.Text('Apert Radii 11'), sg.InputText(default_text=daofun.phot_dict['AB'], key='AB'), sg.Button('?', key=f'INFO_AB', size=(1,1))],
        [sg.Text('Apert Radii 12'), sg.InputText(default_text=daofun.phot_dict['AC'], key='AC'), sg.Button('?', key=f'INFO_AC', size=(1,1))],
        # [sg.Text('InnerSky Radii'), sg.InputText(default_text=daofun.phot_dict['IS'], key='IS'), sg.Button('?', key=f'INFO_IS', size=(1,1))],
        # [sg.Text('OuterSky Radii'), sg.InputText(default_text=daofun.phot_dict['OS'], key='OS'), sg.Button('?', key=f'INFO_OS', size=(1,1))],
        [sg.Button('Save Advanced Options', key='-SAVE_ADVANCED_OPTIONS-')],
        ]

    window = sg.Window('Advanced Options', layout, finalize=True)
    return window 

def open_allstar_advanced_options_window(daofun):
    layout = [
        [sg.Text('allstar.opt (advanced)')],
        # [sg.Text('Fitting Radius'), sg.InputText(default_text=daofun.allstar_dict['fi'], key='a_fi'), sg.Button('?', key=f'INFO_a_fi', size=(1,1))],
        [sg.Text('Recenter Stars'), sg.InputText(default_text=daofun.allstar_dict['re'], key='a_re'), sg.Button('?', key=f'INFO_a_re', size=(1,1))],
        [sg.Text('Watch Progress'), sg.InputText(default_text=daofun.allstar_dict['wa'], key='a_wa'), sg.Button('?', key=f'INFO_a_wa', size=(1,1))],
        [sg.Text('Percent Error '), sg.InputText(default_text=daofun.allstar_dict['pe'], key='a_pe'), sg.Button('?', key=f'INFO_a_pe', size=(1,1))],
        [sg.Text('Clip Exponent '), sg.InputText(default_text=daofun.allstar_dict['ce'], key='a_ce'), sg.Button('?', key=f'INFO_a_ce', size=(1,1))],
        [sg.Text('Clip Range    '), sg.InputText(default_text=daofun.allstar_dict['cr'], key='a_cr'), sg.Button('?', key=f'INFO_a_cr', size=(1,1))],
        [sg.Text('Max Group Size'), sg.InputText(default_text=daofun.allstar_dict['ma'], key='a_ma'), sg.Button('?', key=f'INFO_a_ma', size=(1,1))],
        [sg.Text('Profile Error '), sg.InputText(default_text=daofun.allstar_dict['pr'], key='a_pr'), sg.Button('?', key=f'INFO_a_pr', size=(1,1))],
        # [sg.Text('InnerSky Radii'), sg.InputText(default_text=daofun.allstar_dict['is'], key='a_is'), sg.Button('?', key=f'INFO_a_is', size=(1,1))],
        [sg.Text('OuterSky Radii'), sg.InputText(default_text=daofun.allstar_dict['os'], key='a_os'), sg.Button('?', key=f'INFO_a_os', size=(1,1))],
        [sg.Button('Save Advanced Options', key='-SAVE_ADVANCED_OPTIONS-')],
        ]

    window = sg.Window('Advanced Options', layout, finalize=True)
    return window 

def parse_arguments():
    parser = argparse.ArgumentParser(description="""
DAOFUN is an interactive adaptation of the DAOPHOT 
astronomical image processing software. It provides 
a user-friendly interface through button-based interactions, 
enabling astronomers to perform various tasks seamlessly. 
The software operates by executing DAOPHOT commands in the 
background, facilitating astronomical data analysis, such 
as finding, photometry, and refining point spread functions (PSFs) 
in images. DAOFUN simplifies the utilization of DAOPHOT's capabilities 
by integrating them into an accessible and intuitive graphical user interface.


Credits: by Carlos Quezada
            inspired in the work of Alvaro Valenzuela
            thanks to DAOPHOT by Peter Stetson""")
    return parser.parse_args()

def run_daofun():
    args = parse_arguments()
    # MAIN WINDOW
    # os.environ['DISPLAY']=':11.0'

    initial_cwd = os.getcwd()
    os.chdir(initial_cwd)
    old_dir = ""

    # INICIAMOS DaoFUN
    daofun = DaoFun()
    layout = init_main_layout(daofun)

    # Create the window
    window = sg.Window("DAOfun: DAOPHOT fun with buttons!!!",
        layout,
        finalize=True,
        element_justification="center",
        resizable=True,
        font="Helvetica 14"
        )

    m_canvas = link_figure(window["-MAIN_CANVAS-"].TKCanvas, daofun.fig_main)
    allstar_canvas = link_figure(window["-ALLSTAR_CANVAS-"].TKCanvas, daofun.fig_allstar)

    fits_selected = False
    daophot_advopt_window = None
    phot_advopt_window = None
    allstar_advopt_window = None

    button_enable = lambda button_x: button_x.update(disabled=False)
    button_disable = lambda button_x: button_x.update(disabled=True)
    button_update = lambda button_x: button_x.update(button_color=('white', 'green'))
    button_outdate = lambda button_x: button_x.update(button_color=('white', 'red'))
    button_normal = lambda button_x: button_x.update(button_color=('#000000', '#fdcb52'))

    while True:
        if daophot_advopt_window is None and phot_advopt_window is None and allstar_advopt_window is None:
            event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == '-FIND-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            daofun.update_opt_daophot(values)
            daofun.save_opt_daophot()
            daofun.find_sumaver = values["find_sumaver"]
            try:
                find(in_fits, f"{filename}.coox", daofun.find_sumaver)
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup(f'FIND terminado, {read_coo(f"{filename}.coox").shape[0]} encontradas.')
            daofun.update_main_fits_find(read_coo(f"{filename}.coox"))
            m_canvas.draw()
            button_update(window['-FIND-'])
            button_enable(window['-PHOT-']), button_outdate(window['-PHOT-'])
            button_disable(window['-PICK-']), button_outdate(window['-PICK-'])
            button_disable(window['-NEW_PSF-'])

        elif event == '-PHOT-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            daofun.update_opt_phot(values)
            daofun.save_opt_phot()
            try:
                phot(in_fits, f"{filename}.coox", f"{filename}.apx")
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup('PHOT terminado (.apx).')
            button_update(window['-PHOT-'])
            button_outdate(window['-PICK-'])
            button_enable(window['-PICK-'])
            button_disable(window['-NEW_PSF-'])

        elif event == '-PICK-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif window['-PHOT-'].ButtonColor == ('white', 'red'):
                sg.popup('No se ha ejecutado PHOT')
                continue
            daofun.pick_minmag = values["pick_minmag"]
            try:
                pick(in_fits, f"{filename}.apx", f"{filename}.lstx", daofun.pick_minmag)
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup(f'PICK terminado, {read_lst(f"{filename}.lstx").shape[0]} elegidas.')
            daofun.update_main_fits_pick(read_lst(f"{filename}.lstx"), 
                                    read_coo(f"{filename}.coox") if os.path.isfile(f"{filename}.coox") else None)
            m_canvas.draw()
            button_update(window['-PICK-'])
            button_enable(window['-NEW_PSF-'])

        elif event == '-ALLSTAR-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif window['-LOAD_ALLSTAR_ALS-'].ButtonColor != ('white', 'green') or window['-LOAD_ALLSTAR_PSF-'].ButtonColor != ('white', 'green'):
                sg.popup('Falta cargar fotometria y modelo PSF')
                continue
            if os.getcwd()!=os.path.dirname(in_psf):
                if os.path.exists(in_psf_name):
                    overwrite_confirmation = sg.popup_yes_no('Accion peligrosa: Estas importando un PSF model y reemplazara un archivo. ¿Desea sobrescribirlos?')
                    if overwrite_confirmation == 'No':
                        continue
                    os.remove(in_psf_name)
                shutil.copy(in_psf, in_psf_name)
            if os.getcwd()!=os.path.dirname(in_als):
                if os.path.exists(in_als_name):
                    overwrite_confirmation = sg.popup_yes_no('Accion peligrosa: Estas importando targets y reemplazara un archivo. ¿Desea sobrescribirlos?')
                    if overwrite_confirmation == 'No':
                        continue
                    os.remove(in_als_name)
                shutil.copy(in_als, in_als_name)
            daofun.update_opt_allstar(values)
            daofun.save_opt_allstar()
            try:
                allstar(in_fits, in_psf_name, in_als_name, f"{filename}.alsx", f"{filename}sx.fits")
                als_out = f"{filename}.alsx"
                fits_out = f"{filename}sx.fits"
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup(f'ALLSTAR terminado.')
            window['-TEXT_OUT_ALS-'].update(value=als_out)
            daofun.update_allstar_canvas(als_out, fits_out)
            allstar_canvas.draw()
            button_update(window['-ALLSTAR-'])
            daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'merr', window['-SCALE_MAIN_FITS-'].get_text())
            m_canvas.draw()
            button_enable(window['-EXPORT_PHOTOMETRY-']), button_enable(window['-INSPECT_PHOTOMETRY-'])
            button_enable(window['-SHOW_MAGerr-']), button_update(window['-SHOW_MAGerr-'])
            button_enable(window['-SHOW_chi2-']), button_enable(window['-SHOW_sharp-'])
            button_normal(window['-SHOW_chi2-']), button_normal(window['-SHOW_sharp-'])
            button_enable(window['-SCALE_MAIN_FITS-'])

        elif event == '-LOAD_FITS_OPT-':
            file_path_sel = sg.popup_get_file('Selecciona un archivo .fits', file_types=(("FITS Files", "*.fits"),))
            if file_path_sel:
                file_path = file_path_sel
                new_re, new_ga = get_parameters_from_fits(file_path)
                window['re'].update(value=new_re)
                window['ga'].update(value=new_ga)

        elif event == '-LOAD_FITS-':
            in_fits_sel = sg.popup_get_file('Selecciona un archivo .fits', file_types=(("FITS Files", "*.fits"),))
            if in_fits_sel:
                in_fits = in_fits_sel
                filename = os.path.splitext(os.path.basename(in_fits))[0]
                daofun.update_main_fits(in_fits)
                if fits_selected:
                    os.chdir(initial_cwd)
                    if old_dir!=initial_cwd and os.path.exists(old_dir):
                        shutil.rmtree(old_dir)
                window['-TEXT_MAIN_FITS-'].update(value=os.path.basename(in_fits))
                m_canvas.draw()
                old_dir = create_working_dir(in_fits)
                fits_selected = True
                button_enable(window['-FIND-']), button_outdate(window['-FIND-'])
                button_disable(window['-PHOT-']), button_disable(window['-PICK-'])
                button_normal(window['-PHOT-']), button_normal(window['-PICK-'])
                button_disable(window['-NEW_PSF-']), button_disable(window['-REFINE_PSF-'])
                button_disable(window['-SHOW_MAGerr-']), button_normal(window['-SHOW_MAGerr-'])
                button_disable(window['-SHOW_chi2-']), button_normal(window['-SHOW_chi2-'])
                button_disable(window['-SHOW_sharp-']), button_normal(window['-SHOW_sharp-'])
                button_disable(window['-SCALE_MAIN_FITS-']), button_normal(window['-SCALE_MAIN_FITS-'])
                button_normal(window['-LOAD_ALLSTAR_ALS-']), button_normal(window['-LOAD_ALLSTAR_PSF-'])
                button_disable(window['-EXPORT_PHOTOMETRY-']), button_disable(window['-INSPECT_PHOTOMETRY-'])
        
        elif event == '-OPEN_DAOPHOT_ADVANCED_OPTIONS-':
            if daophot_advopt_window is None:
                daophot_advopt_window = open_daophot_advanced_options_window(daofun)
            else:
                dao_advopt_event, dao_advopt_values = daophot_advopt_window.read()
                if dao_advopt_event == sg.WINDOW_CLOSED or dao_advopt_event == 'Close':
                    daophot_advopt_window.close()
                    daophot_advopt_window = None
                elif dao_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                    daofun.update_opt_daophot(dao_advopt_values)
                    daophot_advopt_window.close()
                    daophot_advopt_window = None
                elif dao_advopt_event.startswith('INFO_'):
                    info_key = dao_advopt_event  # La clave del botón de información
                    info_text = daofun.info_docstrings.get(info_key, "No hay información disponible para esta opción.")
                    sg.popup_scrolled(info_text, title="Help", size=(80, 30))
        
        elif event == '-OPEN_PHOT_ADVANCED_OPTIONS-':
            if phot_advopt_window is None:
                phot_advopt_window = open_phot_advanced_options_window(daofun)
            else:
                phot_advopt_event, phot_advopt_values = phot_advopt_window.read()
                if phot_advopt_event == sg.WINDOW_CLOSED or phot_advopt_event == 'Close':
                    phot_advopt_window.close()
                    phot_advopt_window = None
                elif phot_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                    daofun.update_opt_phot(phot_advopt_values)
                    phot_advopt_window.close()
                    phot_advopt_window = None
                elif phot_advopt_event.startswith('INFO_'):
                    info_key = phot_advopt_event  # La clave del botón de información
                    info_text = daofun.info_docstrings.get(info_key, "No hay información disponible para esta opción.")
                    sg.popup_scrolled(info_text, title="Help", size=(80, 30))
        
        elif event == '-OPEN_ALLSTAR_ADVANCED_OPTIONS-':
            if allstar_advopt_window is None:
                allstar_advopt_window = open_allstar_advanced_options_window(daofun)
            else:
                allsar_advopt_event, allstar_advopt_values = allstar_advopt_window.read()
                if allsar_advopt_event == sg.WINDOW_CLOSED or allsar_advopt_event == 'Close':
                    allstar_advopt_window.close()
                    allstar_advopt_window = None
                elif allsar_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                    daofun.update_opt_allstar(allstar_advopt_values)
                    allstar_advopt_window.close()
                    allstar_advopt_window = None
                elif allsar_advopt_event.startswith('INFO_'):
                    info_key = allsar_advopt_event  # La clave del botón de información
                    info_text = daofun.info_docstrings.get(info_key, "No hay información disponible para esta opción.")
                    sg.popup_scrolled(info_text, title="Help", size=(80, 30))

        elif event == '-NEW_PSF-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif window['-PHOT-'].ButtonColor == ('white', 'red') or window['-PICK-'].ButtonColor == ('white', 'red'):
                sg.popup('Falta actualizar PHOT y PICK')
                continue
            try:
                clean_psf_lst()
                shutil.copy(f"{filename}.lstx", f"{filename}.lst0")
                shutil.copy(f"{filename}.apx", f"{filename}.ap_psf")
                refine_psf(in_fits, f"{filename}.lst0", f"{filename}.lst1", 80)
                create_psf(in_fits, f"{filename}.ap_psf", f"{filename}.lst1", f"{filename}.psf1", f"{filename}.nei1")
                last_lst = f"{filename}.lst1"
                last_psf = f"{filename}.psf1"
                refines = 1
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup(f'PSF terminado.')
            daofun.update_main_fits_sel(read_lst(f"{filename}.lst1"), 
                                    read_lst(f"{filename}.lst0"),
                                    read_coo(f"{filename}.coox") if os.path.isfile(f"{filename}.coox") else None)
            m_canvas.draw()
            button_enable(window['-REFINE_PSF-'])
        
        elif event == '-REFINE_PSF-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            try:
                last_fits = in_fits if refines==1 else f"{filename}s{refines-1}.fits"
                new_fits = f"{filename}s{refines}.fits"
                last_lst = f"{filename}.lst{refines}"
                new_lst = f"{filename}.lst{refines+1}"
                last_psf = f"{filename}.psf{refines}"
                new_psf = f"{filename}.psf{refines+1}"
                last_nei = f"{filename}.nei{refines}"
                new_nei = f"{filename}.nei{refines+1}"

                sub_fits(last_fits, last_psf, last_nei, new_fits, last_lst)
                refine_psf(new_fits, last_lst, new_lst, 80)
                create_psf(new_fits, f"{filename}.ap_psf", new_lst, new_psf, new_nei)
                refines+=1
                last_lst, last_psf = new_lst, new_psf
            except FileNotFoundError as err:
                sg.popup(f"[FILE NOT FOUND]:\n{err}")
                continue
            sg.popup(f'PSF refinada.')
            daofun.update_main_fits_sel(read_lst(new_lst), 
                                    read_lst(f"{filename}.lst0"),
                                    read_coo(f"{filename}.coox") if os.path.isfile(f"{filename}.coox") else None)
            m_canvas.draw()
        
        elif event == '-LOAD_ALLSTAR_ALS-':
            in_als_sel = sg.popup_get_file('Selecciona un archivo de fotometria', 
                                                                    file_types=(("AP Files", "*.ap*"),
                                                                                    ("ALS Files", "*.als*"),))
            if in_als_sel:
                in_als = in_als_sel
                in_als_name = os.path.basename(in_als)
                window['-TEXT_ALLSTAR_ALS-'].update(value=in_als_name)
                button_update(window['-LOAD_ALLSTAR_ALS-'])
                button_outdate(window['-ALLSTAR-'])

        elif event == '-LOAD_ALLSTAR_PSF-':
            in_psf_sel = sg.popup_get_file('Selecciona un archivo de modelo PSF', file_types=(("PSF Files", "*.psf*"),))
            if in_psf_sel:
                in_psf = in_psf_sel
                in_psf_name = os.path.basename(in_psf)
                window['-TEXT_ALLSTAR_PSF-'].update(value=in_psf_name)
                button_update(window['-LOAD_ALLSTAR_PSF-'])
                button_outdate(window['-ALLSTAR-'])
        
        elif event == '-EXPORT_PHOTOMETRY-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif window['-ALLSTAR-'].ButtonColor != ('white', 'green'):
                sg.popup('Falta correr ALLSTAR')
                continue
            out_dir = sg.popup_get_folder('Selecciona donde exportar la fotometria:')
            if out_dir:
                export_als = os.path.join(out_dir, os.path.splitext(als_out)[0])+".als"
                export_psf = os.path.join(out_dir, os.path.splitext(in_psf_name)[0])+".psf"
                export_lst_in = os.path.join(out_dir, os.path.splitext(in_psf_name)[0]) + ".lst" + os.path.splitext(in_psf_name)[1].strip(".psf")
                export_lst_out = os.path.join(out_dir, os.path.splitext(in_psf_name)[0]) + ".lst"
                try:
                    shutil.copy(als_out, export_als)
                    shutil.copy(in_psf_name, export_psf)
                    sync_lst_als(os.path.basename(export_lst_in), als_out, "out.lst")
                    shutil.copy("out.lst", export_lst_out)
                    os.remove("out.lst")
                except FileNotFoundError as err:
                    sg.popup(f"[FILE NOT FOUND]:\n{err}")
                    continue
                sg.popup('Fotometria exportada.')

        elif event == '-INSPECT_PHOTOMETRY-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif window['-ALLSTAR-'].ButtonColor != ('white', 'green'):
                sg.popup('Falta correr ALLSTAR')
                continue
            inspect_targets(in_fits, als_out)
            

        elif event == '-EXPORT_OPT_FILES-':
            if not fits_selected:
                sg.popup('No se ha seleccionado un fits.')
                continue
            out_dir = sg.popup_get_folder('Selecciona donde exportar los opt files:')
            if out_dir:
                export_dao_opt = os.path.join(out_dir,"daophot.opt")
                export_phot_opt = os.path.join(out_dir, "photo.opt")
                export_allstar_opt = os.path.join(out_dir, "allstar.opt")
                try:
                    shutil.copy("daophot.opt", export_dao_opt)
                except FileNotFoundError as err:
                    sg.popup(f"[FILE NOT FOUND]:\n{err}")
                try:
                    shutil.copy("photo.opt", export_phot_opt)
                except FileNotFoundError as err:
                    sg.popup(f"[FILE NOT FOUND]:\n{err}")
                try:
                    shutil.copy("allstar.opt", export_allstar_opt)
                except FileNotFoundError as err:
                    sg.popup(f"[FILE NOT FOUND]:\n{err}")
                sg.popup('OPT files exportados.')

        elif event == '-SHOW_MAGerr-':
            if not window['-SHOW_MAGerr-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'merr', window['-SCALE_MAIN_FITS-'].get_text())
                m_canvas.draw()
                button_update(window['-SHOW_MAGerr-'])
                button_normal(window['-SHOW_chi2-']), button_normal(window['-SHOW_sharp-'])

        elif event == '-SHOW_chi2-':
            if not window['-SHOW_chi2-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'chi', window['-SCALE_MAIN_FITS-'].get_text())
                m_canvas.draw()
                button_update(window['-SHOW_chi2-'])
                button_normal(window['-SHOW_MAGerr-']), button_normal(window['-SHOW_sharp-'])
        
        elif event == '-SHOW_sharp-':
            if not window['-SHOW_sharp-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'sharpness', window['-SCALE_MAIN_FITS-'].get_text())
                m_canvas.draw()
                button_update(window['-SHOW_sharp-'])
                button_normal(window['-SHOW_chi2-']), button_normal(window['-SHOW_MAGerr-'])
        elif event == '-SCALE_MAIN_FITS-':
            if window['-SCALE_MAIN_FITS-'].get_text()=="Linear":
                window['-SCALE_MAIN_FITS-'].update(text="Log")
            else:
                window['-SCALE_MAIN_FITS-'].update(text="Linear")
            if window['-SHOW_MAGerr-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'merr', window['-SCALE_MAIN_FITS-'].get_text())
            elif window['-SHOW_chi2-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'chi', window['-SCALE_MAIN_FITS-'].get_text())
            elif window['-SHOW_sharp-'].ButtonColor == ('white', 'green'):
                daofun.update_main_fits_allstar(read_als(f"{filename}.alsx"), 'sharpness', window['-SCALE_MAIN_FITS-'].get_text())    
            m_canvas.draw()
        elif event.startswith('INFO_'):
            info_key = event  # La clave del botón de información
            info_text = daofun.info_docstrings.get(info_key, "No hay información disponible para esta opción.")
            sg.popup_scrolled(info_text, title="Help", size=(80, 30))



    window.close()
    os.chdir(initial_cwd)
    if old_dir!=initial_cwd and os.path.exists(old_dir):
        shutil.rmtree(old_dir)
    daofun.close_figs()

if __name__ == "__main__":
    run_daofun()