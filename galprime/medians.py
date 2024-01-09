import os
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

from astropy.table import Table


def max_sma(table_list, sma_key="sma"):
    """Get the maximum semimajor axis for a table list

    :param table_list: _description_
    :type table_list: _type_
    :param sma_key: _description_, defaults to "sma"
    :type sma_key: str, optional
    :return: _description_
    :rtype: _type_
    """
    global_max = 0
    for tab in table_list:
        this_max = np.max(tab[sma_key])
        if this_max > global_max:
            global_max = this_max
    return global_max



def interp_median(xs, interp_list):
    """Get the median for a list of interp1D object profiles

    :param xs: The x slices to get the median at
    :type xs: array_like
    :param interp_list: The list of interp1D objects (individual profiles)
    :type interp_list: list(scipy.interp1D)
    :return: A 1D interpolation of the median profile.
    :rtype: scipy.interpolate.interp1D
    """
    median_prof = []
    for x in xs:
        median_prof.append(np.median([interp(x) for interp in interp_list]))
    return interp1d(xs, median_prof, bounds_error=False, fill_value="extrapolate")


def array2D_median(array2D):
    """ Get the median of a 2D array of profiles (used in bootstrap_median()) """
    return np.nanmedian(array2D, axis=0)


def boostrap_median(table_list, iterations=500, x_step=0.5, plot_dir=None, plot_prefix=""):
    """ Generate an Astropy Table object containing the values from bootstrapping of a set of profiles.

    :param table_list: A set of tables (outputs from galprime.isophote_fitting, processed using the to_table() 
        method). 
    :type table_list: list(Table())
    :param iterations: The number of bootstrap iterations to run, defaults to 500
    :type iterations: int, optional
    :param x_step: The step between samples along the x-axis, defaults to 0.5
    :type x_step: float, optional
    :param plot_dir: Directory to generate output plots where if None, no plots are made, defaults to None
    :type plot_dir: str, optional
    :param plot_prefix: Prefix to include in the output plots. This is helpful if you have a 
                        bin and want to include which bin the plots are coming from, defaults to ""
    :type plot_prefix: str, optional
    :return: The table of 
    :rtype: astropy Table object
    """
    
    # We will start by making a 2D image with a shared x-axis, so we can sample WAY FASTER than using interp1D objects
    interp_list = [interp1d(tab["sma"], tab["intens"], bounds_error=False, fill_value="extrapolate") for tab in table_list]
    
    x_slices = np.arange(0, max_sma(table_list), x_step)
    full_array = []
    for n in interp_list:
        full_array.append(n(x_slices))
    full_array = np.asarray(full_array)     # This is now a 2D array (basically an image)

    median_prof = array2D_median(full_array)
    bs_1sig_u, bs_1sig_l = [], []
    bs_2sig_u, bs_2sig_l = [], []
    bs_3sig_u, bs_3sig_l = [], []

    lower_index_1sig, upper_index_1sig = int(iterations * 0.159), int(iterations * 0.841)
    lower_index_2sig, upper_index_2sig = int(iterations * 0.023), int(iterations * 0.977)
    lower_index_3sig, upper_index_3sig = int(iterations * 0.002), int(iterations * 0.998)


    boostrapped_medians = []
    indices = np.arange(0, len(table_list))

    # Bootsrap our population, generate a new 2D array, and add the median of that to our list
    for iter in range(iterations):
        pop_indices = np.random.choice(indices, size=len(table_list), replace=True)
        pop_array2D = full_array[pop_indices]
        boostrapped_medians.append(array2D_median(pop_array2D))
    
    boostrapped_medians = np.asarray(boostrapped_medians)
    
    # Now go through the y slices, and get the bootstrap values at each sigma interval
    for i in range(boostrapped_medians.shape[1]):
        y_slice = np.sort(boostrapped_medians[:, i])
        bs_1sig_l.append(y_slice[lower_index_1sig])
        bs_1sig_u.append(y_slice[upper_index_1sig])
        bs_2sig_l.append(y_slice[lower_index_2sig])
        bs_2sig_u.append(y_slice[upper_index_2sig])
        bs_3sig_l.append(y_slice[lower_index_3sig])
        bs_3sig_u.append(y_slice[upper_index_3sig])

    colnames = ["SMA", "MEDIAN", "L_1SIG", "U_1SIG", "L_2SIG", "U_2SIG", "L_3SIG", "U_3SIG",]
    bootstrap_table = Table([x_slices, median_prof, bs_1sig_l, bs_1sig_u, bs_2sig_l, bs_2sig_u, bs_3sig_l, bs_3sig_u],
                            names=colnames)
    
    # Here are some nice plotting routines 
    if plot_dir is not None:
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        

        plt.figure(facecolor="white")
        plt.imshow(np.log10(full_array), vmin=-6, vmax=1)
        plt.yticks([])
        plt.xticks([])
        plt.title("Input Profiles")
        plt.tight_layout()
        plt.savefig(plot_dir + plot_prefix + "_inputs.png", dpi=150)


        plt.figure(facecolor="white")
        plt.imshow(np.log10(boostrapped_medians), vmin=-6, vmax=1)
        plt.yticks([])
        plt.xticks([])
        plt.title("Bootstrapped Median Profiles")
        plt.tight_layout()
        plt.savefig(plot_dir + plot_prefix + "_bootstrapped.png", dpi=150)


        plt.figure(facecolor="white")
        for i in range(full_array.shape[0]):
            plt.plot(x_slices, full_array[i], color="grey")
        
        plt.plot(x_slices, median_prof, color="red")

        plt.yscale("log")
        plt.ylim(1e-6, 10)
        plt.xlabel(r'$SMA$ [pix]')
        plt.ylabel("I(R)")

        plt.title("Profiles")
        plt.tight_layout()
        plt.savefig(plot_dir + plot_prefix + "_profiles.png", dpi=150)


        plt.figure(facecolor="white")
        plt.plot(x_slices, median_prof, color="Red")
        plt.plot(x_slices, bs_3sig_u, color="orange")
        plt.plot(x_slices, bs_3sig_l, color="orange")
        plt.plot(x_slices, bs_2sig_u, color="black")
        plt.plot(x_slices, bs_2sig_l, color="black")
        plt.plot(x_slices, bs_1sig_u, color="grey")
        plt.plot(x_slices, bs_1sig_l, color="grey")

        plt.yscale("log")
        plt.ylim(1e-6, 10)
        plt.xlabel(r'$SMA$ [pix]')
        plt.ylabel("I(R)")

        plt.title("Profiles")
        plt.tight_layout()
        plt.savefig(plot_dir + plot_prefix + "_boostrapped_medians.png", dpi=150)

    return bootstrap_table
