""" Classes and methods for handling astronomical data """

import math
import os
import pickle

from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.nddata import Cutout2D

from pathlib import Path
import numpy as np

import galprime


class CutoutList:
    """Class to contain both the data and associated metadata for a set of cutouts.
    """
    
    def __init__(self, cutouts=None, cutout_data=[], metadata={}):
        if cutouts is None:
            cutouts = []
        if metadata == None:
            metadata = {}
        
        self.cutouts = cutouts              # List to hold cutouts
        self.cutout_data = cutout_data      # This is the container for associated data to each cutout
        self.metadata = metadata            # Any extra needed metadata
    
        metadata["N_CUTOUTS"] = len(cutouts)
    
    def update_medatada(self):
        self.metadata["N_CUTOUTS"] = len(self.cutouts)
        
    def cutouts_to_file(self, filename="cutouts.fits", overwrite=False):
        
        if len(self.cutouts) == 0:
            return

        out_hdulist = fits.HDUList()
        head = fits.Header()
        
        # Add the metadata (if any) to the header
        for n in self.metadata:
            head[n] = self.metadata[n]

        # Now add the individual cutouts to the HDUList with associated data in the individual headers
        for i, cutout in enumerate(self.cutouts):
            this_head = head.copy()
            try:
                this_head.update(self.cutout_data[i])
            except ValueError:
                pass
            out_hdulist.append(fits.ImageHDU(data=cutout, header=this_head))
        
        out_hdulist.writeto(filename, overwrite=overwrite)



def get_image_filenames(images_directory, modifier='*.fits'):
    """
    Retrieves a list of all available filenames for a given directory, and a given band.

    :param images_directory: The top-level directory from which to get image filenames from.
    :type images_directory: str
    :param image_band: The image band, defaults to "i"
    :type image_band: str, optional
    :param check_band: Check to make sure the images are of a certain band, defaults to False.
        WARNING: Optimized for HSC filename format (ex: HSC-I_9813_4c3.fits).
    :type check_band: bool, optional
    """
    image_filenames = []
    images = Path(images_directory).rglob(modifier)
    for image in images:
        image_filenames.append(str(image))
    return image_filenames




def get_closest_psf(psfs, ra, dec, ra_key="RA", dec_key="DEC"):
    ras = np.array([psfs.cutout_data[i]["RA"] for i in range(len(psfs.cutout_data))])
    decs = np.array([psfs.cutout_data[i]["DEC"] for i in range(len(psfs.cutout_data))])
    
    closest = np.argmin(((ras - ra) ** 2 + (decs - dec) ** 2))
        
    return psfs.cutouts[closest]


def get_wcs(fits_filename):
    """
        Finds and returns the WCS for an image. If Primary Header WCS no good, searches each index until a good one
        is found. If none found, raises a ValueError

        :param fits_filename: The filename for the FITS file.
        :type fits_filename: str
        :return: The WCS object.
        :raises: ValueError
    """
    # Try just opening the initial header
    wcs_init = wcs.WCS(fits_filename)
    ra, dec = wcs_init.axis_type_names
    if ra.upper() == "RA" and dec.upper() == "DEC":
        return wcs_init

    else:
        hdu_list = fits.open(fits_filename)
        for n in hdu_list:
            try:
                wcs_slice = wcs.WCS(n.header)
                ra, dec = wcs_slice.axis_type_names
                if ra.upper() == "RA" and dec.upper() == "DEC":
                    return wcs_slice
            except:
                continue
        hdu_list.close()

    raise ValueError


def get_angular_size_dist(z, H0=71, WM=0.27):
    """
    Return the angular size distance in Megaparsecs.
    (Stripped down version of Cosmocalc by Ned Wright and Tom Aldcroft (aldcroft@head.cfa.harvard.edu))

    :param z: The redshift.
    :type z: float
    :param H0: The Hubble constant, defaults to 71 km/s/Mpc
    :type H0: float, optional
    :param WM: matter density parameter, defaults to 0.27
    :type WM: float, optional
    """
    try:
        c = 299792.458  # velocity of light in km/sec

        if z > 100:
            z /= 299792.458  # Values over 100 are in km/s

        WV = 1.0 - WM - 0.4165 / (H0 * H0)  # Omega(vacuum) or lambda
        age = 0.0  # age of Universe in units of 1/H0

        h = H0 / 100.
        WR = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
        WK = 1 - WM - WR - WV
        az = 1.0 / (1 + 1.0 * z)
        n = 1000  # number of points in integrals
        for i in range(n):
            a = az * (i + 0.5) / n
            adot = math.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
            age += 1. / adot

        DCMR = 0.0

        # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
        for i in range(n):
            a = az + (1 - az) * (i + 0.5) / n
            adot = math.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
            DCMR = DCMR + 1. / (a * adot)

        DCMR = (1. - az) * DCMR / n

        # tangential comoving distance
        ratio = 1.0
        x = math.sqrt(abs(WK)) * DCMR
        if x > 0.1:
            if WK > 0:
                ratio = 0.5 * (math.exp(x) - math.exp(-x)) / x
            else:
                ratio = math.math.sin(x) / x
        else:
            y = x * x
            if WK < 0:
                y = -y
            ratio = 1. + y / 6. + y * y / 120.
        DCMT = ratio * DCMR
        DA = az * DCMT
        Mpc = lambda x: c / H0 * x
        DA_Mpc = Mpc(DA)

        return DA_Mpc
    except:
        raise ValueError


def gen_cutouts(image_directory, config=None, buffer=10, progress_bar=False):
    n_cutouts = 10 if config is None else config["N_BACKGROUNDS"]
    cutout_size = 71 if config is None else config["SIZE"]
    if cutout_size % 2 == 0:
        cutout_size += 1 
    
    hw = cutout_size / 2
    
    image_files = get_image_filenames(image_directory)
    
    cutouts, cutout_data = [], []
    for i in range(n_cutouts):
        this_fn = np.random.choice(image_files)
        
        with fits.open(this_fn) as HDUList:
            for i in range(len(HDUList)):
                data = HDUList[i].data
                if type(data) == np.ndarray:
                    this_wcs = wcs.WCS(HDUList[i].header)
                    break
        x, y = np.random.randint(hw + buffer, data.shape[0] - hw - buffer), np.random.randint(hw + buffer, data.shape[1] - hw - buffer)
        ra, dec = this_wcs.wcs_pix2world(x,y,0)
        cutout = Cutout2D(data=data, position=[x,y], size=cutout_size, wcs=this_wcs)
        
        cutouts.append(cutout.data)
        
        cutout_data.append({"IMG":this_fn, "X":x, "Y":y, "RA":float(ra), "DEC":float(dec) })
    
    return CutoutList(cutouts=cutouts, cutout_data=cutout_data)


def cutouts_from_file(filename):
    cutouts, cutout_datasets = [], []
    with fits.open(filename) as HDUList:
        for hdu in HDUList:
            try:
                cutout, cutout_data = hdu.data, hdu.header
                cutouts.append(cutout)
                cutout_datasets.append(cutout_data)
            except:
                continue
        
    return CutoutList(cutouts=cutouts, cutout_data=cutout_datasets)


def ordered_dict(names, params):
    out_dict = {}
    for i, n in enumerate(names):
        out_dict[n] = params[i]
    return out_dict


def dict_extend(d, extension="_1"):
    """ Adds an extension onto the dict keys for a given input dictionary.
    Helpful when needing to add multiple dicts to a given metadata file

    :param d: An input dictionary
    :type d: dict
    :param extension: The extension to add to each key, defaults to "_1"
    :type extension: str, optional
    :return: Dictionary with added extensions onto each key
    :rtype: dict
    """
    d_new = {}
    for n in d:
        d_new[n + extension] = d[n]
    return d_new


def good_isolist_names():
     return ['sma', 'intens', 'intens_err', 'ellipticity', 'ellipticity_err', 
              'pa', 'pa_err', 'x0', 'x0_err', 'y0', 'y0_err']


def table_from_isolist(isolist, hdu = True, names=None):
    """ Generates an astropy table

    :param isolist: The photutils.isophote.isophotelist object 
    :type isolist: photutils.isophote.isophotelist
    :param names: Column names, defaults to galprime.good_isolist_names()
    :type names: list, optional
    :return: The table generated from the isolist
    :rtype: astropy.table.Table
    """

    if names is None:
        names = good_isolist_names()
    isolist_t = isolist.to_table()

    return Table([isolist_t[col] for col in names],  names=names)


def gen_out_prefix(bin_params):
    """Generates a custom string based on the bin params

    :param bin_params: _description_
    :type bin_params: _type_
    :return: _description_
    :rtype: _type_
    """
    prefix = "bin_"

    for i in range(int(len(bin_params) / 2)):
        prefix += str(bin_params[i*2]) + "-" + str(bin_params[i*2+1]) + "_"

    return prefix


def gen_index_prefix(bin_params, config):
    """ Go through the available bin parameters and generate a docstring that corresponds with
    an index-adjusted version of the other prefix (easier for plotting).

    :param bin_params: _description_
    :type bin_params: _type_
    :param config: _description_
    :type config: _type_
    :return: _description_
    :rtype: _type_
    """
    prefix = ""
    for i in range(int(len(bin_params) / 2)):
        param = bin_params[i * 2]
        check = np.array(config["BIN_" + str(i)])
        delta = np.abs(check - param)
        prefix += str(int(np.argmin(delta))) + "_"    
    return prefix


def gen_filestructure(outdir):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for d in ["bare_profiles/", "bgadded_profiles/", "bgsub_profiles/", "additional_info/",
              "bare_medians/", "bgadded_medians/", "bgsub_medians/", "tempfiles/"]:
        os.makedirs(outdir + d, exist_ok=True)
    
    file_dict = {"BARE_PROFILES": outdir + "bare_profiles/",
                 "BGADDED_PROFILES": outdir + "bgadded_profiles/",
                 "BGSUB_PROFILES": outdir + "bgsub_profiles/",
                 "BARE_MEDIANS": outdir + "bare_medians/",
                 "BGADDED_MEDIANS": outdir + "bgadded_medians/",
                 "BGSUB_MEDIANS": outdir + "bgsub_medians/",
                 "ADDITIONAL": outdir + "additional_info/",
                 "TEMP": outdir + "tempfiles/"}
    return file_dict


def construct_table_row(gprime_container, names):
    """Return an individual row for an output table

    :param gprime_container: The galprime container to draw metadata from.
    :type gprime_container: galprime.GalprimeContainer
    :param names: The column names to draw from (must be identical to what is saved in metadata).
    :type names: array_like
    :return: A row containing the relevant parameters.
    :rtype: array_like
    """
    meta = gprime_container.metadata

    row = [meta[n] for n in names]
    return row


def construct_infotable(container_list, names):
    """ Generate an astropy Table with all the relevant data neatly saved. 

    :param container_list: A list of galprime GalPrimeContainer objects
    :type container_list: array_like(GalPrimeContainer)
    :param names: The column names to draw from (must be identical to what is saved in metadata).
    :type names: array_like
    :return: The output table, with column names identical to names
    :rtype: astropy.table.Table
    """
    rows = []

    for container in container_list:
        rows.append(construct_table_row(container, names))
    
    t_out = Table(rows=rows, names=names)

    return t_out


def to_sb(y, m_0=27, arcconv=0.168):
    y_new = -2.5 * np.log10(y / (arcconv ** 2)) + m_0
    return y_new


def save_pickle(out_object, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(out_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        out_object = pickle.load(handle)
    return out_object