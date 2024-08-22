from daofun.daophot_opt import daophot_dict, phot_dict, allstar_dict, info_docstrings
from daofun.fits_handler import FITSFigureV2
from daofun.misc_tools import read_als, check_file, create_working_dir
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


cwd = os.getcwd()

class DaoFun:

    def __init__(self):
        self.daophot_dict = daophot_dict
        self.phot_dict = phot_dict
        self.allstar_dict = allstar_dict

        self.info_docstrings = info_docstrings

        self.find_sumaver = '1,1'
        self.pick_minmag = '200,20'

        self.fig_main = plt.figure(figsize=(7.5,7.5))
        self.fig_main.tight_layout()

        self.im_main = None

        self.fig_allstar = plt.figure(figsize=(6,6))
        self.fig_allstar.tight_layout()


    # SECCION DAOPHOT
    def save_opt_daophot(self, folder_path=""):
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

    def update_opt_daophot(self, values):
        for k in values.keys():
            if k in self.daophot_dict.keys():
                self.daophot_dict[k] = values[k]

    def save_opt_phot(self, folder_path=""):
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

    def update_opt_phot(self, values):
        for k in values.keys():
            if k in self.phot_dict.keys():
                self.phot_dict[k] = values[k]

    # COLUMNA ALLSTAR
    def save_opt_allstar(self, folder_path=""):   
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

    def update_opt_allstar(self, values):
        for k in values.keys():
            if k[2:] in self.allstar_dict.keys():
                self.allstar_dict[k[2:]] = values[k]


    def update_main_fits(self, file_path):
        if self.im_main != None:
            self.fig_main.clf()
        self.im_main = FITSFigureV2(file_path,
                            subplot=(1,1,1), 
                            figure=self.fig_main)
        self.im_main.tick_labels.set_font(size='small')                               
        self.im_main.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)
        self.fig_main.tight_layout()

    def update_main_fits_find(self, coo):
        color_sel = [(1, 0, 0, 0.3)]
        color_bck = [(0, 0, 1, 0.3)]
        [self.fig_main.axes[0].collections.pop() for i in range(len(self.fig_main.axes[0].collections))]
        self.im_main.show_circles_colors(coo.X, coo.Y, 
                        radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                        colors=color_bck*(len(coo.Y)-1),
                        facecolor=None)
        
    def update_main_fits_pick(self, lst, coo=None):
        color_sel = [(1, 0, 0, 0.3)]
        color_bck = [(0, 0, 1, 0.3)]
        [self.fig_main.axes[0].collections.pop() for i in range(len(self.fig_main.axes[0].collections))]
        if type(coo)!=None:
            self.im_main.show_circles_colors(coo.X, coo.Y, 
                            radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                            colors=color_bck*(len(coo.Y)-1),
                            facecolor=None)
        self.im_main.show_circles_colors(lst.X, lst.Y, 
                        radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                        colors=color_sel,
                        facecolor=None)
        
    def update_main_fits_sel(self, new_lst, lst, coo=None):
        color_sel = [(1, 0, 0, 0.3)]
        color_bck = [(0, 0, 1, 0.3)]
        color_new_sel = [(0, 1, 0, 0.3)]
        [self.fig_main.axes[0].collections.pop() for i in range(len(self.fig_main.axes[0].collections))]
        if type(coo)!=None:
            self.im_main.show_circles_colors(coo.X, coo.Y, 
                            radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                            colors=color_bck*(len(coo.Y)-1),
                            facecolor=None)
        self.im_main.show_circles_colors(lst.X, lst.Y, 
                        radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                        colors=color_sel,
                        facecolor=None)
        self.im_main.show_circles_colors(new_lst.X, new_lst.Y, 
                    radius=float(self.daophot_dict["fi"]), coords_frame="pixel", 
                    colors=color_new_sel,
                    facecolor=None)
        
    def update_main_fits_allstar(self, new_als, col, scale="Linear"):
        vals = np.abs(new_als[col]) if col=="sharpness" else new_als[col]
        vals = np.log10(vals) if scale=="Log" else vals
        min_val = np.percentile(vals, 5)
        max_val = np.percentile(vals, 95)
        vals_inv = 1 - (vals - min_val) / (max_val - min_val)
        color_sel = cm.RdYlGn(vals_inv)
        color_sel = [(*color[:3], 0.3) for color in color_sel]
        [self.fig_main.axes[0].collections.pop() for i in range(len(self.fig_main.axes[0].collections))]
        # if type(coo)!=None:
        #     self.im_main.show_circles_colors(coo.X, coo.Y, 
        #                     radius=float(self.allstar_dict["fi"]), coords_frame="pixel", 
        #                     colors=color_bck*(len(coo.Y)-1),
        #                     facecolor=None)
        self.im_main.show_circles_colors(new_als.X, new_als.Y, 
                        radius=float(self.allstar_dict["fi"]), coords_frame="pixel", 
                        colors=color_sel,
                        facecolor=None)

    def update_allstar_canvas(self, als_out, fits_out):
        cmd = read_als(als_out)
        cmd_f = (cmd.merr<0.13) & (cmd.chi<2) & (cmd.sharpness<1) & (cmd.sharpness>-1)
        cmd_sel = cmd[cmd_f]
        cmd_no_sel = cmd[~cmd_f]

        self.fig_allstar.clf()

        ax1 = self.fig_allstar.add_subplot(2,2,1)
        ax1.plot(cmd_no_sel.MAG, cmd_no_sel.chi, "k+", alpha=0.4)
        ax1.plot(cmd_sel.MAG, cmd_sel.chi, "r+")
        ax1.set_ylim(0, 7)
        ax1.set_xlim(11, 22)
        ax1.set_ylabel("$\chi^2$", size=10)
        ax1.set_xlabel("MAG", size=10)
        ax1.grid()
        # ax1.set_title("$\chi^2$", fontsize=10)


        ax2 = self.fig_allstar.add_subplot(2,2,2)
        ax2.plot(cmd_no_sel.MAG, cmd_no_sel.sharpness, "k+", alpha=0.4)
        ax2.plot(cmd_sel.MAG, cmd_sel.sharpness, "r+")
        ax2.set_ylim(-4, 4)
        ax2.set_xlim(11, 22)
        ax2.set_ylabel("Sharp", size=10)
        ax2.set_xlabel("MAG", size=10)
        ax2.grid()
        # ax2.set_title("Sharp", fontsize=10)

        ax3 = self.fig_allstar.add_subplot(2,2,3)
        ax3.plot(cmd_no_sel.MAG, cmd_no_sel.merr, "k+", alpha=0.4)
        ax3.plot(cmd_sel.MAG, cmd_sel.merr, "r+")
        ax3.set_ylim(0, 0.5)
        ax3.set_xlim(11, 22)
        ax3.set_ylabel("$MAG_{err}$", size=10)
        ax3.set_xlabel("MAG", size=10)
        ax3.grid()
        # ax3.set_title("$I_{err}$", fontsize=20)

        im = FITSFigureV2(fits_out,
                            subplot=(2,2,4), 
                            figure=self.fig_allstar)
        im.tick_labels.set_font(size='small')                               
        im.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)

        self.fig_allstar.tight_layout()

    def close_figs(self):
        plt.close(self.fig_main)
        plt.close(self.fig_allstar)

