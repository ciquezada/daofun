from daofun.fits_handler import FITSFigureV2
import os
import re
from time import sleep  # Otra opción sería "from time import sleep" para usar solo sleep
import pandas as pd
import matplotlib.pyplot as plt
# from astropy.visualization import astropy_mpl_style
# plt.style.use(astropy_mpl_style)
# from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import aplpy
from IPython import display
import numpy as np


def os_mkdir(folder_name):
    # making needed directories
    gen = (x for x in range(1,99999))
    aux_folder_name = folder_name[:-1]
    while True:
        try:
            os.mkdir(folder_name)
            break
        except OSError as error:
            folder_name = f"{aux_folder_name}{next(gen)}"
    return folder_name

def del_files(admitted_files):
	for_del = [*filter(lambda x: x not in admitted_files, os.listdir())]
	if len(for_del):
		print("remove? (y)")
		print('\n'.join(for_del))
		sleep(1)
		yes = input()
		if yes in ["y", ""]:
			cmddel = '\n'.join([f"rm {x}" for x in for_del])
			os.system(cmddel)

def max_ext(ext):
	reg_ex = rf"\.{ext}\d+|{ext}\d+\.fits"
	ext_list = " ".join(re.findall(reg_ex, " ".join(os.listdir())))
	indx_list = [int(x) for x in re.findall(r"\d+", ext_list)]
	if len(indx_list):
		return max(indx_list)
	return ""

def check_file(file, msg=""):
	if not os.path.isfile(file):
		raise FileNotFoundError(f"{msg}{file}")

def read_lst(file_path, file_path_out=None):
	file_path_out = file_path if not file_path_out else file_path_out
	with open(file_path, 'r') as f:
		header = []
		header.append(f.readline())
		header.append(f.readline())
	header_str = ''.join([line for line in header])
	
	lst = pd.read_csv(file_path,
						skiprows = 3,
						names = ['ID', 'X', 'Y', 'MAG', 'SHARP', 'ROUND'],
						delim_whitespace=True
						)
	
	def custom_save(path_out=file_path_out, lst_out=lst):
		lst_out.to_csv(path_out, index=False, sep="\t", header=0)
		with open(path_out, 'r+') as f:
			content = f.read()
			f.seek(0, 0)
			f.write(f"{header_str}\n{content}")
	
	lst.custom_save = custom_save
	return lst

def read_coo(file_path, file_path_out=None):
	file_path_out = file_path if not file_path_out else file_path_out
	with open(file_path, 'r') as f:
		header = []
		header.append(f.readline())
		header.append(f.readline())
	header_str = ''.join([line for line in header])
	
	coo = pd.read_csv(file_path,
						skiprows = 3,
						names = ['ID', 'X', 'Y', 'COL1', 'COL2', 'COL3', "COL4"],
						delim_whitespace=True
						)
	
	def custom_save(path_out=file_path_out, coo_out=coo):
		coo_out.to_csv(path_out, index=False, sep="\t", header=0)
		with open(path_out, 'r+') as f:
			content = f.read()
			f.seek(0, 0)
			f.write(f"{header_str}\n{content}")
	
	coo.custom_save = custom_save
	return coo

def read_als(file_path, file_path_out=None):
	file_path_out = file_path if not file_path_out else file_path_out
	with open(file_path, 'r') as f:
		header = []
		header.append(f.readline())
		header.append(f.readline())
	header_str = ''.join([line for line in header])
	
	als = pd.read_csv(file_path,
						skiprows = 3,
						names = ['ID', 'X', 'Y', "MAG",  "merr",  "msky",  "niter",  "chi",  "sharpness"],
						delim_whitespace=True
						)
	
	def custom_save(path_out=file_path_out, als=als):
		als.to_csv(path_out, index=False, sep="\t", header=0)
		with open(path_out, 'r+') as f:
			content = f.read()
			f.seek(0, 0)
			f.write(f"{header_str}\n{content}")
	
	als.custom_save = custom_save
	return als

def read_ap(file_path, file_path_out=None):
	# Lee el archivo con los datos
	with open(file_path, 'r') as file:
		header = []
		header.append(file.readline())
		header.append(file.readline())
		header_str = ''.join([line for line in header])
		file.readline()
		file.readline()
		lines = file.readlines()
		

	# Inicializa listas para cada columna
	star_ids = []
	x_coordinates = []
	y_coordinates = []
	aperture1_magnitudes = []
	# aperture2_magnitudes = []
	# aperture3_magnitudes = []
	# aperture4_magnitudes = []
	modal_sky_values = []
	std_dev_sky_values = []
	# skewness_sky_values = []
	std_error_aperture1 = []
	# std_error_aperture2 = []
	# std_error_aperture3 = []
	# std_error_aperture4 = []

	# Procesa los bloques de texto
	for i in range(0, len(lines), 3):
		
		# Divide las dos líneas de cada bloque
		header_line = [line for line in lines[i].strip().split(' ') if line != ""]
		data_line = [line for line in lines[i+1].strip().split(' ') if line != ""]
		
		# Extrae los datos de la primera línea
		star_id = int(header_line[0])
		x_coord = float(header_line[1])
		y_coord = float(header_line[2])
		aperture1_mag = float(header_line[3])
		# aperture2_mag = float(header_line[4])
		# aperture3_mag = float(header_line[5])
		
		# Extrae los datos de la segunda línea
		modal_sky = float(data_line[0])
		std_dev_sky = float(data_line[1])
		# skewness_sky = float(data_line[2])
		std_error_a1 = float(data_line[3])
		# std_error_a2 = float(data_line[4])
		# std_error_a3 = float(data_line[5])
		
		# Agrega los datos a las listas correspondientes
		star_ids.append(star_id)
		x_coordinates.append(x_coord)
		y_coordinates.append(y_coord)
		aperture1_magnitudes.append(aperture1_mag)
		# aperture2_magnitudes.append(aperture2_mag)
		# aperture3_magnitudes.append(aperture3_mag)
		modal_sky_values.append(modal_sky)
		std_dev_sky_values.append(std_dev_sky)
		# skewness_sky_values.append(skewness_sky)
		std_error_aperture1.append(std_error_a1)
		# std_error_aperture2.append(std_error_a2)
		# std_error_aperture3.append(std_error_a3)
	del(lines)
	# Crea un DataFrame con los datos
	data = {
		'ID': star_ids,
		'X': x_coordinates,
		'Y': y_coordinates,
		'MAG': aperture1_magnitudes,
		# 'MAG2': aperture2_magnitudes,
		# 'MAG3': aperture3_magnitudes,
		'msky': modal_sky_values,
		'mskyerr': std_dev_sky_values,
		# 'SKYskew': skewness_sky_values,
		'merr': std_error_aperture1,
		# 'merr2': std_error_aperture2,
		# 'merr3': std_error_aperture3,
	}

	ap = pd.DataFrame(data)
	def custom_save(path_out=file_path_out, ap=ap):
		ap.to_csv(path_out, index=False, sep="\t", header=0)
		with open(path_out, 'r+') as f:
			content = f.read()
			f.seek(0, 0)
			f.write(f"{header_str}\n{content}")
	
	ap.custom_save = custom_save
	return ap

def find_ID(als, row):
	als_copy = als.copy()
	als_copy["dist"] = als_copy.apply(lambda x: (x.X-row.X)**2 + (x.Y-row.Y)**2, axis=1)
	dist = als_copy.sort_values("dist").iloc[0].dist
	if dist>1:
		raise LookupError(f"Distance for {row.ID} is too large ({dist}px)")
	return int(als_copy.sort_values("dist").iloc[0].ID)

def mag2flux(mag):
	return 10**(-2/5 * (mag-25))

def show_fits(filename, pos_files=[], fits_suffix=""):
	pos_files = pos_files if isinstance(pos_files, list) or isinstance(pos_files, tuple) else [pos_files]
	fig = plt.figure(figsize=(16,16))
	im = FITSFigureV2(f"{filename}{fits_suffix}.fits", subplot=(1,1,1), figure=fig)
	im.tick_labels.set_font(size='small')
	im.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)
	for i, ext in enumerate(pos_files):
		colors = [(0,0,1), (1,0,0), (0,1,0)]
		if "lst" in ext or "nei" in ext:
			pos_lst = read_lst(f"{filename}.{ext}")
		elif "als" in ext:
			pos_lst = read_als(f"{filename}.{ext}")
		im.show_circles_colors(pos_lst.X, pos_lst.Y, radius=4, coords_frame="pixel", colors=np.array([(*colors[i%3],0.3)]*len(pos_lst.Y)))
	plt.subplots_adjust(wspace=0.3)
	plt.ion()
	plt.show(block=False)
	plt.clf()
	display.clear_output(wait = True)
	im.close()
	return 1

def show_fits_simple(filename, pos_files=[]):
	pos_files = pos_files if isinstance(pos_files, list) or isinstance(pos_files, tuple) else [pos_files]
	fig = plt.figure(figsize=(16,16))
	im = FITSFigureV2(filename, subplot=(1,1,1), figure=fig)
	im.tick_labels.set_font(size='small')
	for i, pos_file in enumerate(pos_files):
		colors = [(0,0,1), (1,0,0), (0,1,0)]
		if "lst" in pos_file or "nei" in pos_file:
			pos_lst = read_lst(pos_file)
		elif "als" in pos_file:
			pos_lst = read_als(pos_file)
		im.show_circles_colors(pos_lst.X, pos_lst.Y, radius=4, coords_frame="pixel", colors=np.array([(*colors[i%3],0.3)]*len(pos_lst.Y)))
	plt.subplots_adjust(wspace=0.3)
	im.show_grayscale(stretch='log', vmin=1, vmax=6.85e4)
	plt.show(block=False)
	im.close()
	return 1

def cartesian2radec(fits_file, x_array, y_array):
	# Load the FITS file
	f = aplpy.FITSFigure(fits_file)

	# Convert pixel coordinates to RA and Dec
	ra_array, dec_array = f.pixel2world(x_array, y_array)

	f.close()
	return ra_array, dec_array

def clean_psf_lst():
	archivos = os.listdir()
	for archivo in archivos:
		if re.match(r'.*(psf|lst)\d+$', archivo.lower()):
			os.remove(archivo)

def create_working_dir(path_to_fits):
    path_to_dir = os.path.abspath(os.path.dirname(path_to_fits))
    filename = os.path.splitext(os.path.basename(path_to_fits))[0]
    os.chdir(path_to_dir)
    working_dir = os_mkdir(f"{filename}_working_dir_0")
    os.system(f"cp {filename}.fits {working_dir}/.")
    os.chdir(working_dir)
    return os.path.abspath(f"../{working_dir}")

def get_parameters_from_fits(file_path):
    with fits.open(file_path) as hdul_param:
        param_header = hdul_param[0].header
        ron = np.mean([param_header[f'HIERARCH ESO DET OUT{i} RON'] for i in range(1, 5)])
        gain = np.mean([param_header[f'HIERARCH ESO DET OUT{i} GAIN'] for i in range(1, 5)])
    
    new_re = np.round(ron, 2)
    new_ga = np.round(gain, 2)
    return new_re, new_ga