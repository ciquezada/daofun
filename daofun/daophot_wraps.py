from daofun.misc_tools import check_file, read_lst, read_als, find_ID 
import os


def find(in_fits, out_coo, sum_aver="1,1", verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	if verbose:
		print(f"daophot: find({filename})")
	check_file("daophot.opt", "opt file input: ")
	check_file(f"{in_fits}", "fits file input: ")
	overwrite = [""] if os.path.isfile(f"{out_coo}") else []
	cmd_list = ['daophot << EOF >> pipe.log', f'at {filename}', 
					'find', f'{sum_aver}', f"{out_coo}",
					*overwrite,
					'y', 
					'exit', 'EOF']
	cmd = '\n'.join(cmd_list)
	os.system(cmd)
	check_file(f"{out_coo}", "coo not created: ")
	if verbose:
		print(f"  -> {out_coo}")
	return(f"{out_coo}")

def phot(in_fits, in_coo, out_ap, opt_file="photo.opt", verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	if verbose:
		print(f"daophot: phot({filename})")
	check_file("daophot.opt", "opt file input: ")
	check_file(f"{in_fits}", "fits file input: ")
	check_file(opt_file, "opt file input: ")
	check_file(f"{in_coo}", "positions file input: ")
	if os.path.isfile(f"{filename}.psf"):
		if verbose:
			print("  PSF file found")
		raise NotImplementedError(f"PSF phot: remove {filename}.psf")
	overwrite = [""] if os.path.isfile(f"{out_ap}") else []
	cmd_list = ['daophot << EOF >> pipe.log', f'at {filename}', 
					'phot', f'{opt_file}\n', 
					f"{in_coo}", f"{out_ap}",
					*overwrite,
					'exit', 'EOF']
	cmd = '\n'.join(cmd_list)
	os.system(cmd)
	check_file(f"{out_ap}", "ap not created: ")
	if verbose:
		print(f"  -> {out_ap}")
	return(f"{out_ap}")

def pick(in_fits, in_ap, out_lst, stars_minmag="200,20", verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	if verbose:
		print(f"daophot: pick({filename})")
	check_file("daophot.opt", "opt file input: ")
	check_file(f"{in_fits}", "fits file input: ")
	check_file(f"{in_ap}", "aperture file input: ")
	overwrite = [""] if os.path.isfile(f"{out_lst}") else []
	cmd_list = ['daophot << EOF >> pipe.log', f'at {filename}', 
					'pick', f"{in_ap}", f'{stars_minmag}', f"{out_lst}",
					*overwrite,
					'exit', 'EOF']
	cmd = '\n'.join(cmd_list)
	os.system(cmd)
	check_file(f"{out_lst}", "lst not created: ")
	if verbose:
		print(f"  -> {out_lst}")
	return(f"{out_lst}")

def create_psf(in_fits, in_ap, in_lst, out_psf, out_nei, suffix="", verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	if verbose:
		print(f"daophot: psf({filename})")
	check_file("daophot.opt", "opt file input: ")
	check_file(f"{in_fits}", "fits file input: ")
	check_file(f"{in_ap}", "ap file input")
	check_file(f"{in_lst}", "lst file input")
	check_file(f"{in_fits}", "fits file input")
	overwrite_psf = [""] if os.path.isfile(f"{out_psf}") else []
	overwrite_nei = [""] if os.path.isfile(f"{os.path.basename(out_psf)[0]}.nei") else []

	cmd_list = ['daophot << EOF >> pipe.log', f'at {filename}', 
				'psf', f"{in_ap}", f"{in_lst}", f"{out_psf}",
				*overwrite_psf, *overwrite_nei,
				'exit', 'EOF',
				f"mv {os.path.splitext(out_psf)[0]}.nei {out_nei} >>pipe.log 2>>pipe.log"]
	cmd = '\n'.join(cmd_list)
	os.system(cmd)
	if out_nei!="":
		check_file(f"{out_nei}", "nei not created: ")
		if verbose:
			print(f"  -> {out_nei}")
	check_file(f"{out_psf}", "psf not created: ")
	if verbose:
		print(f"  -> {out_psf}")
	return(f"{out_psf}")

def sub_fits(in_fits, in_psf, in_sub, out_fits, in_lst=False, verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	out_filename = os.path.splitext(os.path.basename(out_fits))[0]
	if verbose:
		print(f"daophot: sub({filename})")
	check_file("daophot.opt", "opt file input: ")
	check_file(f"{in_psf}", "psf file input")
	check_file(f"{in_sub}", "substraction list file input")
	check_file(f"{in_fits}", "fits file input")
	add_exceptions = []
	if in_lst:
		check_file(f"{in_lst}", "exception list file input")
		add_exceptions = ["y", f"{in_lst}"]

	cmd_list = ['daophot << EOF >> pipe.log', 
				f'at {filename}', 
				'sub', f"{in_psf}", f"{in_sub}", 
				*add_exceptions,
				f"{out_filename}", 
				'exit', 'EOF']
	cmd = '\n'.join(cmd_list)
	os.system(cmd)
	check_file(f"{out_fits}", "fits not created: ")
	if verbose:
		print(f"  -> {out_fits}")
	return(f"{out_fits}")

def allstar(in_fits, in_psf, in_ap, out_als, out_fits, RE=True, verbose=True):
	filename = os.path.splitext(os.path.basename(in_fits))[0]
	out_filename = os.path.splitext(os.path.basename(out_fits))[0]
	if verbose:
		print(f"allstar: {filename}")
	check_file("allstar.opt", "opt file input: ")
	check_file(f"{in_fits}", "fits file input: ")
	check_file(f"{in_psf}", "psf file input")
	check_file(f"{in_ap}", "positions list file input")
	overwrite_als = [""] if os.path.isfile(f"{out_als}") else []
	do_RE = [""] if RE else ["RE=0\n"]
	cmd_list = ['allstar << EOF >> pipe.log',
				*do_RE,
				filename, f"{in_psf}", f"{in_ap}", 
				f"{out_als}", *overwrite_als, f"{out_filename}", 
				'exit', "EOF"]
	cmdallstar = '\n'.join(cmd_list)
	os.system(cmdallstar)
	check_file(f"{out_als}", "als not created: ")
	if verbose:
		print(f"  -> {out_als}")
	return(f"{out_als}")

def sync_lst_als(lst_in, als_in, lst_out, verbose=True):
	lst = read_lst(f"{lst_in}", f"{lst_out}")
	als = read_als(f"{als_in}")
	def find_ID_wrap(als, row):
		try:
			return find_ID(als, row)
		except LookupError as err:
			if verbose:
				print(err)
			return "NotFound"			
	lst["ID"] = lst.apply(lambda row: find_ID_wrap(als, row), axis=1)
	lst_new = lst.query("ID != 'NotFound'")
	lst.custom_save(f"{lst_out}", lst_new)