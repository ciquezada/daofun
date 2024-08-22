from daofun.fits_handler import FITSFigureV2
from daofun.misc_tools import check_file, read_lst, read_als, read_coo
import PySimpleGUI as sg
import matplotlib
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import aplpy as apl
import astropy.io.fits as fits
# matplotlib.use("TkAgg")
sg.theme('DarkAmber')  


def inspect_targets(in_fits, in_targets, fits_suffix=""):
    in_targets_ext = os.path.splitext(in_targets)[1]
    if "als" in in_targets_ext:
        read_cmd = read_als
    elif "lst" in in_targets_ext:
        read_cmd = read_lst
    elif "coo" in in_targets_ext:
        read_cmd = read_coo
    with plt.style.context('dark_background'):

        color_sel = [(1, 0, 0, 0.3)]
        color_bck = [(0, 0, 1, 0.3)]


        filename = os.path.splitext(os.path.basename(in_fits))[0]
        # LEEMOS EL HEADER (LAS DOS PRIMERAS FILAS)
        check_file(f"{in_targets}", "lst file input ")
        check_file(f"{in_fits}", "fits file input ")


        lst = read_cmd(f"{in_targets}")

        dropped_stars_ids  = []

        saving_text = lambda n: f"Targets  ({n}/{lst.shape[0]})"

        star_counter = 0


        # ARMAMOS FIGURA CON EL FITS
        fig = plt.figure(figsize=(4, 4))
        fig.tight_layout()
        im = FITSFigureV2(f"{in_fits}",
                                                            subplot=(1,1,1), 
                                                            figure=fig)
        im.tick_labels.set_font(size='small')
        im.show_circles_colors(lst.X, lst.Y, 
                                    radius=4, coords_frame="pixel", 
                                    colors=color_bck*len(lst.Y),
                                    facecolor=None)
        # im.set_title(i+k, loc="left")  
        if 1: #SACAR                                  
            im.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)   

        # ARMAMOS FIT CORTADO
        cropfig = plt.figure(figsize=(4, 4))
        cropfig.tight_layout()
        cropim = apl.FITSFigure(f"{in_fits}",
                                                            subplot=(1,1,1), 
                                                            figure=cropfig)
        cropim.tick_labels.set_font(size='small')
        if 1: #SACAR                                  
            cropim.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)    
        # CARGAMOS LA IMAGEN
        hdulist = fits.open(f"{in_fits}")
        complete_image   = hdulist[0].data
        graphfig = plt.figure(figsize=(4,4))
        graphfig.tight_layout()
        graph_ax = graphfig.add_subplot(1, 1, 1, projection='3d')




        # LAYOUT COLUMNAS
        mainfit_viewer_column = [
            [sg.Text(text=f"{in_fits}", size=(40, 1))],
            [sg.Canvas(key="-MAIN CANVAS-", expand_x=True, expand_y=True)],
        ]

        star_list_column = [
            [
                sg.Listbox(
                    values=[f"{int(row.ID):5d} - mag: {row.MAG:2.1f}" for i, row in lst.sort_values("ID").iterrows()],
                    enable_events=True, size=(18, 20), key="-STAR LIST-", auto_size_text=True
                )
            ],
        ]

        graph_viewer_column = [
            [sg.Text(text="", size=(35, 1), justification="center", text_color="black", background_color = "black", key="-GTOP-")],
            [sg.Canvas(key="-CROP CANVAS GRAPH-")],
        ]

        bottom_column = [
            [
                sg.Text(text=saving_text(lst.shape[0]-len(dropped_stars_ids)), size=(80, 1), key="-TBOT-"),
                sg.Button("OK")
                ],
        ]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(mainfit_viewer_column),
                sg.VSeperator(),
                sg.Column(star_list_column),
                sg.VSeperator(), 
                sg.Canvas(key="-CROP CANVAS-", expand_x=True),
                sg.VSeperator(), 
                sg.Column(graph_viewer_column, justification="center", element_justification="center"),
                ],
            [
                sg.Column(bottom_column, justification="right", element_justification="right")
                ]
        ]

        # Create the window
        window = sg.Window("Source Inspector",
            layout,
            finalize=True,
            element_justification="center",
            font="Helvetica 18",
            )
        window.bind("<Up>", "up_arrow")
        window.bind("<Down>", "down_arrow")
        window.bind("<Left>", "left_arrow")
        window.bind("<Right>", "right_arrow")

        def link_figure(canvas, figure):
            figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
            figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
            return figure_canvas_agg
        # Linking plots
        m_canvas = link_figure(window["-MAIN CANVAS-"].TKCanvas, fig)
        crop_canvas = link_figure(window["-CROP CANVAS-"].TKCanvas, cropfig)
        graph_canvas = link_figure(window["-CROP CANVAS GRAPH-"].TKCanvas, graphfig)

        def canvas_update_star(star_id):
            lst_no_sel = lst[~(lst.ID==star_id)]
            star = lst[(lst.ID==star_id)].iloc[0]
            im.show_circles_colors(lst_no_sel.X, lst_no_sel.Y, 
                            radius=4, coords_frame="pixel", 
                            colors=color_bck*(len(lst.Y)-1),
                            facecolor=None)
            im.show_circles_colors(star.X, star.Y, 
                            radius=4, coords_frame="pixel", 
                            colors=color_sel,
                            facecolor=None)
            m_canvas.draw()

            ra, dec = cropim.pixel2world(star['X'], star['Y'])
            cropim.recenter(ra, dec, radius=5/3600.)
            crop_canvas.draw()


            if np.isnan(complete_image[max(int(star['Y'])-20, 0):int(star['Y'])+20, max(int(star['X'])-20, 0):int(star['X'])+20]).any():
                window["-GTOP-"].update("[EDGE WARNING]", background_color="orange")
            else:
                window["-GTOP-"].update("", background_color = "black")

            # graphfig = plt.figure(figsize=(4,4))
            # graphfig.tight_layout()
            # graph_ax = graphfig.add_subplot(1, 1, 1, projection='3d')

            xlims = (max(int(star['X'])-10, 0), min(int(star['X'])+10, complete_image.shape[1]))
            ylims = (max(int(star['Y'])-10, 0), min(int(star['Y'])+10, complete_image.shape[0]))

            Z_data = complete_image[ylims[0]:ylims[1], xlims[0]:xlims[1]]
            Z_data_log = np.copy(Z_data)
            Z_data_log[Z_data_log<=0.001] = 10**0.5
            Z_data_log = np.log10(Z_data_log)
            
            X_data = [*range(*xlims)]
            Y_data = [*range(*ylims)]
            X_data, Y_data = np.meshgrid(X_data, Y_data)
            graph_ax.clear()
            surf = graph_ax.plot_surface(X_data, Y_data, Z_data_log, rstride=1, cstride=1, cmap=cm.inferno,
                            linewidth=0, antialiased=False)
            graph_ax.set_ylabel("DEC (PIX)")
            graph_ax.set_xlabel("RA (PIX)")

            theta = np.linspace(0, 2 * np.pi, 100)
            y_circ = 4*np.cos(theta)+int(star['Y'])
            x_circ = 4*np.sin(theta)+int(star['X'])
            graph_ax.plot(x_circ, y_circ, Z_data_log.max()*0.7, color="black", zorder=5)

            # def animate(angle):
            #     graph_ax.view_init(elev=20., azim=angle*3)
            #     return [surf]

            # # Animate
            # anim = animation.FuncAnimation(graphfig, animate, frames=120, interval=50, blit=True)

            # # Save
            # writer = animation.PillowWriter(fps=20,
            #                     metadata=dict(artist='Me'),
            #                     bitrate=1800)
            # anim.save('temp_sel.gif', writer=writer)
            # plt.close(graphfig)

            # window.FindElement("-CROP CANVAS GRAPH-").UpdateAnimation('temp_sel.gif',  time_between_frames=50)

            graph_canvas.draw()

        anim = None
        while True:
            event, values = window.read()
            # End program if user closes window or
            # presses the OK button
            if event=="OK":
                break

            elif event in ["-STAR LIST-", "up_arrow", "down_arrow"]:
                if not len(window["-STAR LIST-"].get_indexes()):
                        continue
                if event.strip("-STAR LIST-") in ["up_arrow", "down_arrow"]:
                    ite = -1 if "up_arrow" in event else 1
                    opt_list = window["-STAR LIST-"].get_list_values()
                    actual_index = window["-STAR LIST-"].get_indexes()[0]
                    next_index = (actual_index+ite)%len(opt_list)
                    window["-STAR LIST-"].update(set_to_index=[next_index], scroll_to_index=next_index)
                    sel_star_id = int(opt_list[next_index].split()[0])
                else:
                    sel_star_id = int(values["-STAR LIST-"][0].split()[0])
                [fig.axes[0].collections.pop() for i in range(len(fig.axes[0].collections))]
                canvas_update_star(sel_star_id)

            if event == sg.WIN_CLOSED:
                break

        window.close()
        plt.close(fig)
        plt.close(cropfig)
        # plt.close(graphfig)