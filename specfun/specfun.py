from specfun.specfun_backend import SpecFun
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import argparse
import os
import json
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

def init_main_layout():
    # LAYOUT COLUMNAS
    mainfit_viewer_column = [
                [sg.Text('Load spec file:')],
                [sg.Input(key='-SPEC_FILE-', enable_events=True, expand_x=True), sg.FileBrowse(file_types=(("Spectra csv", "*.csv*"), ))],
                [sg.Checkbox(key='-NOT_USE_SINGLE_SPEC-', text="Use a directory of spectra")],
                [sg.Text('Load spectra database .csv:')],
                [sg.Input(key='-SPEC_DB_FILE-', enable_events=True, expand_x=True), sg.FileBrowse(file_types=(("Spectra Database", "*.csv*"), ))],
                [sg.Text('Load spectra database .csv directory:')],
                [sg.Input(key='-SPEC_DB_DIR-', enable_events=True, expand_x=True), sg.FolderBrowse()],
                [sg.Text('File fmt (ex: "id{:04d}")'), sg.Input(key='-SPEC_FILE_FMT-', enable_events=True, expand_x=True)],
                # [sg.Button('Daophot options', key='-OPEN_DAOPHOT_ADVANCED_OPTIONS-', size=(20,1)),
                #     sg.Text(text="", expand_x=True, key="-placeholder-"),
                #         sg.Button('Import daophot.opt', key='-LOAD_DAOPHOT_OPT-', size=(20,1))],
                # [sg.Button('Photometry options', key='-OPEN_PHOT_ADVANCED_OPTIONS-', size=(20,1)),
                #     sg.Text(text="", expand_x=True, key="-placeholder-"),
                #         sg.Button('Import photo.opt', key='-LOAD_PHOT_OPT-', size=(20,1))],
                # [sg.Button('Allstar options', key='-OPEN_ALLSTAR_ADVANCED_OPTIONS-', size=(20,1)),
                #     sg.Text(text="", expand_x=True, key="-placeholder-"),
                #         sg.Button('Import allstar.opt', key='-LOAD_ALLSTAR_OPT-', size=(20,1))],
                [sg.Button('Run Pipeline', key='-RUN_specfun-')],
                [sg.Text('', key='-PBAR_TITLE-', expand_x=True), 
                    sg.ProgressBar(100, orientation='h', expand_x=True, size=(20,20), key='-PBAR-'),
                        sg.Text('', key='-PBAR_PROGRESS-', expand_x=True)],
                ]

    selected_options = []
    available_options = [*SpecFun.pipeline_options.keys()]

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
                    [sg.Button('Import', size=(10, 2), key='-IMPORT_PIPELINE-')],
                    [sg.Button('Export', size=(10, 2), key='-EXPORT_PIPELINE-')],
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

def open_kwargs_window(kwargs, title, pipeline_to_do):
    layout = [
        [sg.Text(f'Configuration for {title}', font=('Helvetica', 14))],
    ]

    for key, value_type in kwargs.items():
        print(key)
        if key == 'spec':
            # Opciones disponibles para el tipo "spec"
            spec_options_all = ['Initial Spectra'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do)]
            spec_options = ['Initial Spectra'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do) if getattr(step, 'output_type', None) == 'spec']

            layout.append([sg.Text(f'{key}:'), sg.Combo(spec_options, key=f'-{key}-', 
                                                            default_value=value_type if value_type=='Initial Spectra' else spec_options_all[value_type+1], size=(20, 1))])
        elif key == 'vcorr_best':
            # Opciones disponibles para el tipo "spec"
            vcorr_options_all = ['Manual Input'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do)]
            vcorr_options = ['Manual Input'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do) if getattr(step, 'output_type', None) == 'RV']

            layout.append([sg.Text(f'Step Output RV:'), sg.Combo(vcorr_options, key=f'-{key}-', 
                                                            default_value="Manual Input" if type(value_type) in [float, str] else vcorr_options_all[value_type+1], size=(20, 1))])
            layout.append([sg.Text(f'Manual Input:'), sg.Input(key=f'-{key}_MI-', default_text=value_type if type(value_type) in [float, str] else "", size=(10, 1))])
        elif key in ['SNR', 'FeH']:
            # Opciones disponibles para el tipo "spec"
            input_options_all = ['Manual Input'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do)]
            input_options = ['Manual Input'] + [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(pipeline_to_do) if getattr(step, 'output_type', None) == key]

            layout.append([sg.Text(f'Step Output {key}:'), sg.Combo(input_options, key=f'-{key}-', 
                                                            default_value="Manual Input" if type(value_type) in [float, str] else input_options_all[value_type+1], size=(20, 1))])
            layout.append([sg.Text(f'Manual Input:'), sg.Input(key=f'-{key}_MI-', default_text=value_type if type(value_type) in [float, str] else "", size=(10, 1))])
        elif type(value_type) == float:
            layout.append([sg.Text(f'{key}:'), sg.Input(key=f'-{key}-', default_text=value_type, size=(10, 1))])
        elif type(value_type) == bool:
            layout.append([sg.Checkbox(key=f'-{key}-', text=key)])
        elif type(value_type) == str and key != "save_path" and key != "output_folder" and key != "star_type":
            layout.append([sg.Text(f'{key}:'), sg.Input(key=f'-{key}-', default_text=value_type, size=(20, 1))])
        elif type(value_type) == tuple:
            layout.append([sg.Text(f'{key}:'), sg.Input(key=f'-{key}_1-', default_text=value_type[0], size=(10, 1)), 
                                sg.Text('to'), sg.Input(key=f'-{key}_2-', default_text=value_type[1], size=(10, 1))])
        elif key == 'save_path' or key == "output_folder":
            layout.append([sg.Text(f'{key}:'), sg.Input(key=f'-{key}-', default_text=value_type, size=(15, 1)), sg.SaveAs(target=f'-{key}-')])
        elif key == "star_type":
            layout.append([sg.Text(f'Star type:'), sg.Combo(["Dwarf", "Giant"], key=f'-{key}-', 
                                                            default_value="Dwarf", size=(20, 1))])

    layout.append([sg.Button('OK', key='-OK-')])

    return sg.Window(f'Configure {title}', layout, finalize=True)

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


def run_specfun():
    args = parse_arguments()
    # MAIN WINDOW
    # os.environ['DISPLAY']=':11.0'

    # initial_cwd = os.getcwd()
    # os.chdir(initial_cwd)
    # old_dir = ""

    # INICIAMOS DaoFUN
    specfun = SpecFun()
    layout = init_main_layout()


    # Create the window
    window = sg.Window("SpecFun: spectra in pipeline",
        layout,
        size=(1250,450),
        finalize=True,
        element_justification="center",
        resizable=True,
        font="Helvetica 14"
        )

    # m_canvas = link_figure(window["-MAIN_CANVAS-"].TKCanvas, daofun.fig_main)
    # allstar_canvas = link_figure(window["-ALLSTAR_CANVAS-"].TKCanvas, daofun.fig_allstar)
    selected_options = []

    # fits_selected = False
    step_kwargs_window = None

    button_enable = lambda button_x: button_x.update(disabled=False)
    button_disable = lambda button_x: button_x.update(disabled=True)
    button_update = lambda button_x: button_x.update(button_color=('white', 'green'))
    button_outdate = lambda button_x: button_x.update(button_color=('white', 'red'))
    button_normal = lambda button_x: button_x.update(button_color=('#000000', '#fdcb52'))

    window['-SELECTED_LIST-'].bind("<Double-Button-1>", "DOUBLECLICK")
    window['-AVAILABLE_LIST-'].bind("<Double-Button-1>", "DOUBLECLICK")

    while True:
        if step_kwargs_window is None:
            event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break

        elif event == '-RUN_specfun-':
            specfun.indv_spectra = not bool(values['-NOT_USE_SINGLE_SPEC-'])
            if specfun.indv_spectra:
                specfun.spec = values["-SPEC_FILE-"]
                specfun.run_pipeline(link_pbar(window["-PBAR-"], 
                                                  window["-PBAR_TITLE-"], 
                                                  window["-PBAR_PROGRESS-"]))
            else:
                specfun.spec_df = values["-SPEC_DB_FILE-"]
                specfun.spec_folder = values["-SPEC_DB_DIR-"]
                specfun.spec_file_fmt = values["-SPEC_FILE_FMT-"]
                window["-PBAR_TITLE-"].update("Photometring Cube: ")
                window["-PBAR-"].update(0)
                window["-PBAR_PROGRESS-"].update("ETA in console")
                window.Refresh()
                specfun.run_pipeline()

        elif event == '-ADD_BUTTON-' or event == '-AVAILABLE_LIST-DOUBLECLICK':
            selected = values['-AVAILABLE_LIST-']
            if selected:
                selected = selected[0]
                new_step_kwargs = specfun.pipeline_options[selected].kwargs.copy()
                if step_kwargs_window is None:
                    step_kwargs_window = open_kwargs_window(new_step_kwargs, selected, specfun.pipeline_to_do)
                step_wind_event, step_wind_values = step_kwargs_window.read()
                if step_wind_event == sg.WINDOW_CLOSED or step_wind_event == 'Close':
                    step_kwargs_window.close()
                    step_kwargs_window = None
                elif step_wind_event=="-OK-":
                    print("SAVE_ADVANCED_OPTIONS")
                    step_kwargs_window.close()
                    step_kwargs_window = None

                    # new_step_kwargs = {key: step_wind_values[f'-{key}-'] for key in new_step_kwargs.keys()}
                    for key, value_type in new_step_kwargs.items():
                        if key == 'spec':
                            selected_option = step_wind_values[f'-{key}-']
                            if selected_option == 'Initial Spectra':
                                new_step_kwargs[key] = 'Initial Spectra'
                            else:
                                ind = int(selected_option[1:3])-1
                                new_step_kwargs[key] = ind
                        elif key in ['vcorr_best', 'SNR', 'FeH']:
                            selected_option = step_wind_values[f'-{key}-']
                            if selected_option != 'Manual Input':
                                ind = int(selected_option[1:3])-1
                                new_step_kwargs[key] = ind
                            elif step_wind_values[f'-{key}_MI-'].isnumeric():
                                new_step_kwargs[key] = float(step_wind_values[f'-{key}_MI-'])
                            else:
                                new_step_kwargs[key] = step_wind_values[f'-{key}_MI-']
                        elif type(value_type) == float:
                            new_step_kwargs[key] = float(step_wind_values[f'-{key}-'])
                        elif type(value_type) == bool:
                            new_step_kwargs[key] = bool(step_wind_values[f'-{key}-'])
                        elif type(value_type) == str:
                            new_step_kwargs[key] = step_wind_values[f'-{key}-']
                        elif type(value_type) == tuple:
                            value1 = float(step_wind_values[f'-{key}_1-'])
                            value2 = float(step_wind_values[f'-{key}_2-'])
                            new_step_kwargs[key] = (value1, value2)
                        elif key == 'save_path':
                            new_step_kwargs[key] = step_wind_values[f'-{key}-']
                    new_step = specfun.pipeline_options[selected]

                    specfun.pipeline_args.append(new_step_kwargs)
                    specfun.pipeline_to_do.append(new_step)
                    print(specfun.pipeline_args)
                    selected_options.append(f"[{len(selected_options)+1:02d}] {selected}")
                    window['-SELECTED_LIST-'].update(values=selected_options)

        # Remover elementos de la lista de elementos seleccionados
        elif event == '-REMOVE_BUTTON-':
            selected = values['-SELECTED_LIST-']
            if selected:
                selected = selected[0]
                # if selected[5:]=="change_Mlist":
                #     i_selected = [i for i, x in enumerate(filter(lambda x: x[5:]==selected[5:], selected_options)) if x==selected][0]
                #     daocube.mlist_queue.pop(i_selected)
                # selected_options.remove(f"{selected}")
                # sel_list = [sel[5:] for sel in selected_options]
                # selected_options = [f"[{i+1:02d}] {comnd}" for i, comnd in enumerate(sel_list)]
                ind_remv = int(selected[1:3])-1
                ind_remv_list = [ind_remv]
                for i, item in enumerate(specfun.pipeline_args):
                    if "spec" in item.keys() and item["spec"] in ind_remv_list:
                        ind_remv_list.append(i)
                ch = sg.popup_ok_cancel(f"This steps will be removed:", 
                                            "\n".join([selected_options[i] for i in ind_remv_list]),
                                            title="Remove")
                if ch=="OK":
                    specfun.pipeline_args = [value for i, value in enumerate(specfun.pipeline_args) if i not in ind_remv_list]
                    specfun.pipeline_to_do = [value for i, value in enumerate(specfun.pipeline_to_do) if i not in ind_remv_list]
                    selected_options = [value for i, value in enumerate(selected_options) if i not in ind_remv_list]
                # AQUI HAY QUE ACTUALIZAR LOS INDICES EN LOS ARGUMENTOS SPEC
                selected_options = [f"[{i+1:02d}] {comnd[5:]}" for i, comnd in enumerate(selected_options)]
                window['-SELECTED_LIST-'].update(values=selected_options)
        
        elif event == '-SELECTED_LIST-DOUBLECLICK':
            selected = values['-SELECTED_LIST-']
            if selected:
                selected = selected[0]
                ind = int(selected[1:3])-1
                new_step_kwargs = specfun.pipeline_args[ind]
                if step_kwargs_window is None:
                    step_kwargs_window = open_kwargs_window(new_step_kwargs, selected, specfun.pipeline_to_do[:ind])
                step_wind_event, step_wind_values = step_kwargs_window.read()
                if step_wind_event == sg.WINDOW_CLOSED or step_wind_event == 'Close':
                    step_kwargs_window.close()
                    step_kwargs_window = None
                elif step_wind_event=="-OK-":
                    print("SAVE_ADVANCED_OPTIONS")
                    step_kwargs_window.close()
                    step_kwargs_window = None

                    # new_step_kwargs = {key: step_wind_values[f'-{key}-'] for key in new_step_kwargs.keys()}
                    for key, value_type in new_step_kwargs.items():
                        if key == 'spec':
                            selected_option = step_wind_values[f'-{key}-']
                            if selected_option == 'Initial Spectra':
                                new_step_kwargs[key] = 'Initial Spectra'
                            else:
                                ind = int(selected_option[1:3])-1
                                new_step_kwargs[key] = ind
                        elif key in ['vcorr_best', 'SNR', 'FeH']:
                            selected_option = step_wind_values[f'-{key}-']
                            if selected_option != 'Manual Input':
                                ind = int(selected_option[1:3])-1
                                new_step_kwargs[key] = ind
                            elif step_wind_values[f'-{key}_MI-'].isnumeric():
                                new_step_kwargs[key] = float(step_wind_values[f'-{key}_MI-'])
                            else:
                                new_step_kwargs[key] = step_wind_values[f'-{key}_MI-']
                        elif type(value_type) == float:
                            new_step_kwargs[key] = float(step_wind_values[f'-{key}-'])
                        elif type(value_type) == bool:
                            new_step_kwargs[key] = bool(step_wind_values[f'-{key}-'])
                        elif type(value_type) == str:
                            new_step_kwargs[key] = step_wind_values[f'-{key}-']
                        elif type(value_type) == tuple:
                            value1 = float(step_wind_values[f'-{key}_1-'])
                            value2 = float(step_wind_values[f'-{key}_2-'])
                            new_step_kwargs[key] = (value1, value2)
                        elif key == 'save_path':
                            new_step_kwargs[key] = step_wind_values[f'-{key}-']

        elif event == '-IMPORT_PIPELINE-':
            file_path = sg.popup_get_file('Importar pipeline', file_types=(("JSON Pipeline File", "*.json"),))
            if file_path:
                with open(file_path, 'r') as archivo:
                    pipeline_in = json.load(archivo)

                # Obtener las listas originales a partir del diccionario
                specfun.pipeline_args = list(args[1] for args in pipeline_in)
                specfun.pipeline_to_do = [specfun.pipeline_options[step] for step in list(args[0] for args in pipeline_in)]
                selected_options = [f"[{i+1:02d}] {step.step_name}" for i, step in enumerate(specfun.pipeline_to_do)]
                window['-SELECTED_LIST-'].update(values=selected_options)


        elif event == '-EXPORT_PIPELINE-':
            file_path = sg.popup_get_file('Exportar pipeline', save_as=True, file_types=(("JSON Pipeline File", "*.json"),))
            if file_path:
                # Crear un diccionario a partir de las listas
                pipeline_out = list(zip([step.step_name for step in specfun.pipeline_to_do], specfun.pipeline_args))

                # Guardar el diccionario en un archivo JSON
                with open(file_path, 'w') as archivo:
                    json.dump(pipeline_out, archivo, indent=4)


 

    window.close()


if __name__ == "__main__":
    run_specfun()