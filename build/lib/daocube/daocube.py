from daocube.daocube_backend import DaoCube
from daocube.daophot_pipeline import DaoPipe
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import argparse
import os
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from tqdm import tqdm

sg.theme('DarkAmber')  
plt.style.use('default')


def format_progress(progress_dict):
    n = progress_dict['n']
    total = progress_dict['total']
    elapsed = progress_dict['elapsed']
    remaining = progress_dict['total'] - progress_dict['n']
    rate = progress_dict['rate'] if progress_dict['rate'] is not None else 0
    cols = progress_dict['ncols']
    
    # Formatear el tiempo transcurrido
    elapsed_hours = int(elapsed // 3600)
    elapsed_minutes = int((elapsed % 3600) // 60)
    elapsed_seconds = int(elapsed % 60)
    
    # Calcular el tiempo restante
    remaining_seconds = int(remaining / rate) if rate and remaining else 0
    remaining_hours = remaining_seconds // 3600
    remaining_minutes = (remaining_seconds % 3600) // 60
    remaining_seconds = remaining_seconds % 60
    
    # Construir la representación del tiempo transcurrido
    elapsed_time = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}" if elapsed_hours > 0 else f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"
    
    # Construir la representación del tiempo restante
    remaining_time = f"{remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}" if remaining_hours > 0 else f"{remaining_minutes:02d}:{remaining_seconds:02d}"
    
    return f"{n}/{total} [{elapsed_time}<{remaining_time}, {rate:.2f}it/s]"


def meter_bar(iterable, desc, *args, title='', total=None, **kwargs):
    key = desc
    max_value = len(iterable) if total==None else total
    sg.OneLineProgressMeter(title, 0, max_value, key, *args, **kwargs)
    for i, val in enumerate(iterable):
        yield val
        if not sg.OneLineProgressMeter(title, i+1, max_value, key, *args, **kwargs):
            break

def link_pbar(window_pbar, window_pbar_title, window_pbar_prog):
    def progress_bar(iterable, desc, total=None):
        max_value = len(iterable) if total==None else total
        i_prog = lambda i: i/max_value*100
        window_pbar_title.update(desc)
        with tqdm(iterable, total=max_value, desc=desc) as tqdm_iterable:
            for i, val in enumerate(tqdm_iterable):
                yield val
                window_pbar.update(i_prog(i+1))
                window_pbar_prog.update(format_progress(tqdm_iterable.format_dict))
    return progress_bar

# COLUMNA CON EL FITS
def link_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def init_main_layout():
    # LAYOUT COLUMNAS
    mainfit_viewer_column = [
                [sg.Text('Load cube fits:')],
                [sg.Input(key='-CUBE_FILE-', enable_events=True, expand_x=True), sg.FileBrowse(file_types=(("FITS cube", "*.fits*"), ))],
                [sg.Text('Load targets file:')],
                [sg.Input(key='-ALS_FILE-', enable_events=True, expand_x=True), sg.FileBrowse(file_types=(("als file", "*.als*"),
                                                                                                            ("ap file", "*.ap*")))],
                [sg.Text('Load targets psf list:')],
                [sg.Input(key='-LST_FILE-', enable_events=True, expand_x=True), sg.FileBrowse(file_types=(("lst file", "*.lst*"), ))],
                [sg.Button('Daophot options', key='-OPEN_DAOPHOT_ADVANCED_OPTIONS-', size=(20,1)),
                    sg.Text(text="", expand_x=True, key="-placeholder-"),
                        sg.Button('Import daophot.opt', key='-LOAD_DAOPHOT_OPT-', size=(20,1))],
                [sg.Button('Photometry options', key='-OPEN_PHOT_ADVANCED_OPTIONS-', size=(20,1)),
                    sg.Text(text="", expand_x=True, key="-placeholder-"),
                        sg.Button('Import photo.opt', key='-LOAD_PHOT_OPT-', size=(20,1))],
                [sg.Button('Allstar options', key='-OPEN_ALLSTAR_ADVANCED_OPTIONS-', size=(20,1)),
                    sg.Text(text="", expand_x=True, key="-placeholder-"),
                        sg.Button('Import allstar.opt', key='-LOAD_ALLSTAR_OPT-', size=(20,1))],
                [sg.Button('Extract spectra', key='-RUN_DAOCUBE-')],
                [sg.Text('', key='-PBAR_TITLE-', expand_x=True), 
                    sg.ProgressBar(100, orientation='h', expand_x=True, size=(20,20), key='-PBAR-'),
                        sg.Text('', key='-PBAR_PROGRESS-', expand_x=True)],
                ]

    selected_options = [f"[{i+1:02d}] {comnd}" for i, comnd in enumerate(DaoCube.cmd_list)]
    available_options = [*DaoPipe.pipeline_list]

    pipeline_viewer_column = [
            [
                sg.Column([
                    [sg.Text(text="Available\nactions", justification="center", 
                            auto_size_text=True)],  # Añadir auto_size_text=True aquí
                    [sg.Listbox(values=available_options, expand_y=True, expand_x=True, key='-AVAILABLE_LIST-')]
                ], expand_y=True, expand_x=True),
                sg.Column([
                    [sg.Button('Add >>', size=(10, 2), key='-ADD_BUTTON-')],
                    [sg.Button('<< Remove', size=(10, 2), key='-REMOVE_BUTTON-')],
                ]),
                sg.Column([
                    [sg.Text(text="Photometry\npipeline", justification="center", 
                            auto_size_text=True)],  # También aquí
                    [sg.Listbox(values=selected_options, expand_y=True, expand_x=True, key='-SELECTED_LIST-')]
                ], expand_y=True, expand_x=True),
            ]
        ]

    layout = [
            [
                sg.Column(mainfit_viewer_column, expand_y=True, expand_x=True),  # Columna izquierda
                sg.VSeparator(),
                sg.Column(pipeline_viewer_column, expand_y=True, expand_x=True),
            ]
        ]
    
    return layout

# ADVANCED OPTIONS WINDOW
def open_daophot_advanced_options_window(daocube):
    layout = [
        [sg.Text('daophot.opt (advanced)')],
        [sg.Text('re'), sg.InputText(default_text=daocube.daophot_dict['re'], key='re')],
        [sg.Text('ga'), sg.InputText(default_text=daocube.daophot_dict['ga'], key='ga')],
        [sg.Text('lo'), sg.InputText(default_text=daocube.daophot_dict['lo'], key='lo')],
        [sg.Text('hi'), sg.InputText(default_text=daocube.daophot_dict['hi'], key='hi')],
        [sg.Text('fw'), sg.InputText(default_text=daocube.daophot_dict['fw'], key='fw')],
        [sg.Text('th'), sg.InputText(default_text=daocube.daophot_dict['th'], key='th')],
        [sg.Text('ls'), sg.InputText(default_text=daocube.daophot_dict['ls'], key='ls')],
        [sg.Text('lr'), sg.InputText(default_text=daocube.daophot_dict['lr'], key='lr')],
        [sg.Text('hs'), sg.InputText(default_text=daocube.daophot_dict['hs'], key='hs')],
        [sg.Text('hr'), sg.InputText(default_text=daocube.daophot_dict['hr'], key='hr')],
        [sg.Text('wa'), sg.InputText(default_text=daocube.daophot_dict['wa'], key='wa')],
        [sg.Text('fi'), sg.InputText(default_text=daocube.daophot_dict['fi'], key='fi')],
        [sg.Text('ps'), sg.InputText(default_text=daocube.daophot_dict['ps'], key='ps')],
        [sg.Text('va'), sg.InputText(default_text=daocube.daophot_dict['va'], key='va')],
        [sg.Text('an'), sg.InputText(default_text=daocube.daophot_dict['an'], key='an')],
        [sg.Text('ex'), sg.InputText(default_text=daocube.daophot_dict['ex'], key='ex')],
        [sg.Text('us'), sg.InputText(default_text=daocube.daophot_dict['us'], key='us')],
        [sg.Text('pr'), sg.InputText(default_text=daocube.daophot_dict['pr'], key='pr')],
        [sg.Text('pe'), sg.InputText(default_text=daocube.daophot_dict['pe'], key='pe')],
        [sg.Button('Save Advanced Options', key='-SAVE_ADVANCED_OPTIONS-')],
]

    window = sg.Window('Advanced Options', layout, finalize=True)
    return window 

def open_phot_advanced_options_window(daocube):
    layout = [
        [sg.Text('photo.opt (advanced)')],
        [sg.Text('A1'), sg.InputText(default_text=daocube.phot_dict['A1'], key='A1')],
        [sg.Text('A2'), sg.InputText(default_text=daocube.phot_dict['A2'], key='A2')],
        [sg.Text('A3'), sg.InputText(default_text=daocube.phot_dict['A3'], key='A3')],
        [sg.Text('A4'), sg.InputText(default_text=daocube.phot_dict['A4'], key='A4')],
        [sg.Text('A5'), sg.InputText(default_text=daocube.phot_dict['A5'], key='A5')],
        [sg.Text('A6'), sg.InputText(default_text=daocube.phot_dict['A6'], key='A6')],
        [sg.Text('A7'), sg.InputText(default_text=daocube.phot_dict['A7'], key='A7')],
        [sg.Text('A8'), sg.InputText(default_text=daocube.phot_dict['A8'], key='A8')],
        [sg.Text('A9'), sg.InputText(default_text=daocube.phot_dict['A9'], key='A9')],
        [sg.Text('AA'), sg.InputText(default_text=daocube.phot_dict['AA'], key='AA')],
        [sg.Text('AB'), sg.InputText(default_text=daocube.phot_dict['AB'], key='AB')],
        [sg.Text('AC'), sg.InputText(default_text=daocube.phot_dict['AC'], key='AC')],
        [sg.Text('IS'), sg.InputText(default_text=daocube.phot_dict['IS'], key='IS')],
        [sg.Text('OS'), sg.InputText(default_text=daocube.phot_dict['OS'], key='OS')],
        [sg.Button('Save Advanced Options', key='-SAVE_ADVANCED_OPTIONS-')],
        ]

    window = sg.Window('Advanced Options', layout, finalize=True)
    return window 

def open_allstar_advanced_options_window(daocube):
    layout = [
        [sg.Text('allstar.opt (advanced)')],
        [sg.Text('fi'), sg.InputText(default_text=daocube.allstar_dict['fi'], key='fi')],
        [sg.Text('re'), sg.InputText(default_text=daocube.allstar_dict['re'], key='re')],
        [sg.Text('wa'), sg.InputText(default_text=daocube.allstar_dict['wa'], key='wa')],
        [sg.Text('pe'), sg.InputText(default_text=daocube.allstar_dict['pe'], key='pe')],
        [sg.Text('ce'), sg.InputText(default_text=daocube.allstar_dict['ce'], key='ce')],
        [sg.Text('cr'), sg.InputText(default_text=daocube.allstar_dict['cr'], key='cr')],
        [sg.Text('ma'), sg.InputText(default_text=daocube.allstar_dict['ma'], key='ma')],
        [sg.Text('pr'), sg.InputText(default_text=daocube.allstar_dict['pr'], key='pr')],
        [sg.Text('is'), sg.InputText(default_text=daocube.allstar_dict['is'], key='is')],
        [sg.Text('os'), sg.InputText(default_text=daocube.allstar_dict['os'], key='os')],
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
            inspired in the work of Manuela Zoccali
            powered by DAOPHOT by Peter Stetson""")
    return parser.parse_args()

def run_daocube():
    args = parse_arguments()
    # MAIN WINDOW
    # os.environ['DISPLAY']=':11.0'

    # initial_cwd = os.getcwd()
    # os.chdir(initial_cwd)
    # old_dir = ""

    # INICIAMOS DaoFUN
    daocube = DaoCube()
    layout = init_main_layout()

    selected_options = [f"[{i+1:02d}] {comnd}" for i, comnd in enumerate(DaoCube.cmd_list)]

    # Create the window
    window = sg.Window("DAOcube: DAOPHOT on a Cube...",
        layout,
        size=(1100,450),
        finalize=True,
        element_justification="center",
        resizable=True,
        font="Helvetica 14"
        )

    # m_canvas = link_figure(window["-MAIN_CANVAS-"].TKCanvas, daofun.fig_main)
    # allstar_canvas = link_figure(window["-ALLSTAR_CANVAS-"].TKCanvas, daofun.fig_allstar)

    # fits_selected = False
    daophot_advopt_window = None
    phot_advopt_window = None
    allstar_advopt_window = None

    button_enable = lambda button_x: button_x.update(disabled=False)
    button_disable = lambda button_x: button_x.update(disabled=True)
    button_update = lambda button_x: button_x.update(button_color=('white', 'green'))
    button_outdate = lambda button_x: button_x.update(button_color=('white', 'red'))
    button_normal = lambda button_x: button_x.update(button_color=('#000000', '#fdcb52'))

    window['-SELECTED_LIST-'].bind("<Double-Button-1>", "DOUBLECLICK")
    window['-AVAILABLE_LIST-'].bind("<Double-Button-1>", "DOUBLECLICK")

    while True:
        if daophot_advopt_window is None and phot_advopt_window is None and allstar_advopt_window is None:
            event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break

        elif event == '-OPEN_DAOPHOT_ADVANCED_OPTIONS-':
            print("OPEN_DAOPHOT_ADVANCED_OPTIONS")
            if daophot_advopt_window is None:
                daophot_advopt_window = open_daophot_advanced_options_window(daocube)
            dao_advopt_event, dao_advopt_values = daophot_advopt_window.read()
            if dao_advopt_event == sg.WINDOW_CLOSED or dao_advopt_event == 'Close':
                daophot_advopt_window.close()
                daophot_advopt_window = None
            elif dao_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                print("SAVE_ADVANCED_OPTIONS")
                daocube.update_opt_dict(dao_advopt_values, daocube.daophot_dict)
                daophot_advopt_window.close()
                daophot_advopt_window = None
        
        elif event == '-OPEN_PHOT_ADVANCED_OPTIONS-':
            if phot_advopt_window is None:
                phot_advopt_window = open_phot_advanced_options_window(daocube)
            phot_advopt_event, phot_advopt_values = phot_advopt_window.read()
            if phot_advopt_event == sg.WINDOW_CLOSED or phot_advopt_event == 'Close':
                phot_advopt_window.close()
                phot_advopt_window = None
            elif phot_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                daocube.update_opt_dict(phot_advopt_values, daocube.phot_dict)
                phot_advopt_window.close()
                phot_advopt_window = None
        
        elif event == '-OPEN_ALLSTAR_ADVANCED_OPTIONS-':
            if allstar_advopt_window is None:
                allstar_advopt_window = open_allstar_advanced_options_window(daocube)
            allsar_advopt_event, allstar_advopt_values = allstar_advopt_window.read()
            if allsar_advopt_event == sg.WINDOW_CLOSED or allsar_advopt_event == 'Close':
                allstar_advopt_window.close()
                allstar_advopt_window = None
            elif allsar_advopt_event=="-SAVE_ADVANCED_OPTIONS-":
                daocube.update_opt_dict(allstar_advopt_values, daocube.allstar_dict)
                allstar_advopt_window.close()
                allstar_advopt_window = None

        elif event == "-LOAD_DAOPHOT_OPT-":
            file_path = sg.popup_get_file('Selecciona un archivo daophot.opt', file_types=(("daophot.opt File", "*.opt"),))
            if file_path:
                daocube.load_opt_file(file_path, daocube.daophot_dict)


        elif event == "-LOAD_PHOT_OPT-":
            file_path = sg.popup_get_file('Selecciona un archivo phot.opt', file_types=(("phot.opt File", "*.opt"),))
            if file_path:
                daocube.load_opt_file(file_path, daocube.phot_dict)


        elif event == "-LOAD_ALLSTAR_OPT-":
            file_path = sg.popup_get_file('Selecciona un archivo allstar.opt', file_types=(("allstar.opt File", "*.opt"),))
            if file_path:
                daocube.load_opt_file(file_path, daocube.allstar_dict)


        elif event == '-RUN_DAOCUBE-':
            if not values["-CUBE_FILE-"]:
                sg.popup('No se ha seleccionado un fits.')
                continue
            elif not values["-ALS_FILE-"]:
                sg.popup('No se ha seleccionado un file de targets.')
                continue
            elif not values["-LST_FILE-"]:
                sg.popup('No se ha seleccionado un file de seleccion psf.')
                continue

            daocube.init_cube(values["-CUBE_FILE-"])
            daocube.load_als(values["-ALS_FILE-"])
            daocube.load_lst(values["-LST_FILE-"])
            daocube.slice_cube(prog_bar=link_pbar(window["-PBAR-"], 
                                                  window["-PBAR_TITLE-"], 
                                                  window["-PBAR_PROGRESS-"]))
            window["-PBAR_TITLE-"].update("Photometring Cube: ")
            window["-PBAR-"].update(0)
            window["-PBAR_PROGRESS-"].update("ETA in console")
            window.Refresh()
            daocube.cmd_list = [sel[5:] for sel in selected_options]
            daocube.run_slices_photometry()
            daocube.extract_spectra(prog_bar=link_pbar(window["-PBAR-"], 
                                                  window["-PBAR_TITLE-"], 
                                                  window["-PBAR_PROGRESS-"]))
            window["-PBAR_TITLE-"].update("Completed")
            window["-PBAR-"].update(0)
            window["-PBAR_PROGRESS-"].update(f"Spectra in {os.path.basename(daocube.cube_spectra_folder)}")
            window.Refresh()

            daocube.clean()
        
        elif event == '-ADD_BUTTON-' or event == '-AVAILABLE_LIST-DOUBLECLICK':
            selected = values['-AVAILABLE_LIST-']
            if selected:
                selected = selected[0]
                if selected == "change_Mlist":
                    new_mlist = sg.popup_get_file('Selecciona un archivo de fotometria', 
                                                                    file_types=(("AP Files", "*.ap*"),
                                                                                    ("ALS Files", "*.als*"),))
                    if not new_mlist:
                        continue
                    daocube.new_master_list(os.path.abspath(new_mlist))
                elif selected == "phot":
                    daocube.last_als_type = "ap"
                elif selected == "allstar":
                    daocube.last_als_type = "als"
                selected_options.append(f"[{len(selected_options)+1:02d}] {selected}")
                window['-SELECTED_LIST-'].update(values=selected_options)

        # Remover elementos de la lista de elementos seleccionados
        elif event == '-REMOVE_BUTTON-' or event == '-SELECTED_LIST-DOUBLECLICK':
            selected = values['-SELECTED_LIST-']
            if selected:
                selected = selected[0]
                if selected[5:]=="change_Mlist":
                    i_selected = [i for i, x in enumerate(filter(lambda x: x[5:]==selected[5:], selected_options)) if x==selected][0]
                    daocube.mlist_queue.pop(i_selected)
                selected_options.remove(f"{selected}")
                sel_list = [sel[5:] for sel in selected_options]
                selected_options = [f"[{i+1:02d}] {comnd}" for i, comnd in enumerate(sel_list)]
                window['-SELECTED_LIST-'].update(values=selected_options)

 





    window.close()


if __name__ == "__main__":
    run_daocube()