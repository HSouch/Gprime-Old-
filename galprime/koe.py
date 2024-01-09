"""
This module contains the methods required to run KOE, which is a tool designed to automatically extract profiles
from a directory of images, with a given input catalogue.
"""

import galprime
import os

from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from pebble import ProcessPool

import numpy as np

import warnings
warnings.filterwarnings("ignore")


def process_object(image, image_wcs, object_info, config):
    """ Process a single object, from the whole image to the final profile, for a given location.

    """
    params = {}
    # Generate the cutout
    loc = np.asarray(image_wcs.wcs_world2pix(object_info[config["RA"]],
                                             object_info[config["DEC"]], 0), dtype=int)
    cutout = Cutout2D(image, (loc[0], loc[1]), size=config["SIZE"], wcs=image_wcs).data

    #
    if config["BG_PARAMS"] == '2D':
        sm = galprime.SourceMask(cutout)
        source_mask = sm.multiple(filter_fwhm=[1, 3, 5], tophat_size=[4, 2, 1])
        bg = galprime.background_2D(cutout, source_mask,
                                   box_size=config["BOX_SIZE"], filter_size=config["FILTER_SIZE"])
        params["2D_BG_MED"] = bg.background_median
        params["2D_BG_RMS_MED"] = bg.background_rms_median
        cutout = cutout - bg.background
    # Mask the cutout
    masked_cutout, mask_params = galprime.mask_cutout(cutout, config=config)

    params.update(mask_params)

    for key in object_info.colnames:
        params[key] = object_info[key]

    # Process extraction
    profile = galprime.isophote_fitting(masked_cutout, config=config)

    # Save the data if applicable

    return {"PROFILE": profile.to_table(), "PARAMS": params}


def koe_pipeline(config):
    """ The entire KOE pipeline configured to work with TBriDGE methods.

    """
    # Load in image
    image_filenames = galprime.get_image_filenames(config["IMAGE_DIRECTORY"])[:]

    # Get objects available for profile extraction
    objects = Table.read(config["CATALOG"])

    for image_filename in image_filenames:
        image, wcs = galprime.select_image(image_filename), galprime.get_wcs(image_filename)

        ra_lims, dec_lims = galprime.extraction_limits(wcs, image.shape, config["SIZE"])

        objects_available = galprime.trim_objects(objects, ra_lims, dec_lims, mag_lim=(0, 20),
                                                 ra_key=config["RA"], dec_key=config["DEC"])

        # Prep the container for all objects
        if not os.path.isdir(config["OUT_DIR"]):
            os.mkdir(config["OUT_DIR"])

        out_filename = os.path.splitext(image_filename.split("/")[len(image_filename.split("/")) - 1])[0]
        output = fits.HDUList()

        # Go through all objects and try to extract a valid result. With this method, we can avoid any pesky
        # infinite loops and maximize computation with multithreading.
        max_index = min([len(objects_available), 10]) if config["TEST_MODE"] else len(objects_available)

        job_list = []
        with ProcessPool(max_workers=config["CORES"]) as pool:
            for i in range(max_index):
                job_list.append(pool.schedule(process_object,
                                              args=(image, wcs, objects_available[i], config),
                                              timeout=config["ALARM_TIME"]))
        # Collect the results
        for i in range(len(job_list)):
            try:
                result = job_list[i].result()

                profile = result["PROFILE"]
                if len(profile) == 0:
                    continue
                # Put the profile into a proper saveable format
                valid_colnames = ["sma", "intens", "intens_err", "ellipticity", "ellipticity_err",
                                  "pa", "pa_err", "x0", "x0_err", "y0", "y0_err", "ndata", "flag", "niter"]
                out_table = Table([profile[col] for col in valid_colnames], names=valid_colnames)

                # Generate hdu header
                header = fits.Header()
                for key in result["PARAMS"]:
                    try:
                        header[key] = result["PARAMS"][key]
                    except:
                        continue

                output.append(fits.BinTableHDU(data=out_table,
                                               header=header))

            except Exception as error:
                print(error.args, i)

        output.writeto(config["OUT_DIR"] + out_filename + "_profiles.fits", overwrite=True)
