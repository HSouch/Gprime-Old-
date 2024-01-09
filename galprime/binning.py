""" Binning methods

This module contains all necessary methods for binning catalogues according to various parameters.

"""

from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt


class BinList:
    """ 
        Container class for bins
    
    """
    
    def __init__(self, bins=None):
        if bins is None:
            bins = []
        elif type(bins) == Bin:
            bins = [bins]
        
        self.bins = bins
        self.min_objects = 4
    
    def rebin(self, index, bounds):
        new_bins = []
        for b in self.bins:            
            new_bins.extend(b.rebin(index, bounds))
        self.bins = new_bins
        
class Bin:
    """ Class for a single bin of information
        :param objects: The catalog rows that belong to the given bin
        :type bin_param_names: array_like
        :param object_column_names: The column names when the objects are sorted column-wise instead of row-wise.
        :type bin_param_names: array_like
        :param bin_params (arr): The values that define the bounds of the bin.
        :type bin_param_names: array_like
        :param bin_param_names (arr): Array of parameter names that the bin was made with.
        :type bin_param_names: array_like
    """
    
    def __init__(self, objects=None, object_column_names = None, bin_params=None):
        if bin_params is None:
            bin_params = []

        if objects is None:
            objects = []
        
        self.objects = objects
        self.object_column_names = object_column_names
        self.bin_params = bin_params
        
    
    def rebin(self, index, bounds):
        """ Rebin a bin according to a set of bounds and a 
            given index
            
            index: the column index to bin by
            bounds: The bounds to place the objects in 
        """
        # Create new bins
        current_bin_params = np.copy(self.bin_params)
        outbins = []

        # Generate bins with the new proper bin parameters
        for i in range(len(bounds) - 1):
            this_bin = Bin(object_column_names=self.object_column_names, 
                           bin_params=current_bin_params)
            this_bin.bin_params = np.append(this_bin.bin_params, 
                                            [np.round(bounds[i], 2), 
                                             np.round(bounds[i + 1], 2)])
            outbins.append(this_bin)
            
        bin_column = np.copy(np.transpose(self.objects)[index])
        for i in range(len(bin_column)):
            for j in range(len(bounds) - 1):
                if bounds[j] < bin_column[i] < bounds[j + 1]:
                    outbins[j].objects.append(self.objects[i])
        return outbins
    
    def columns(self):
        return np.transpose(self.objects)
    
    def param_dict(self):
        param_dict = {}
        columns = self.columns()
        for i in range(len(self.object_column_names)):
            param_dict[self.object_column_names[i]] = columns[i]
        return param_dict
    
    def file_prefix(self):
        prefix = "bin_"
        for i in range(0, len(self.bin_params)):
            if (i + 1) % 2 != 0:
                prefix += str(self.bin_params[i]) + "-"
            else:
                prefix += str(self.bin_params[i]) + "_"
        return prefix
    
    def to_table(self, filename=None, overwrite=False):
        """ Creates a astropy.table.Table object for the objects and names contained
        within the bin.

        :param filename: Filename for saving, defaults to None
        :type filename: str, optional
        :param overwrite: Overwrite filename to save, defaults to False
        :type overwrite: bool, optional
        :return: Table of data parameters and associated 
        :rtype: astropy.table.Table()
        """
        columns = self.columns()
        t_out = Table()
        # Add each column (and associated name) to the table
        for i, name in enumerate(self.object_column_names):
            t_out[name] = columns[i]
        
        # If supplied with a filename, write the Table to a fits table 
        if filename is not None:
            t_out.write(filename, format='fits', overwrite=overwrite)

        return t_out

    
def bin_catalog(config):
    """ Bin the inputted catalog.

    This is a convenience function to get a list of bins based on redshift, star-formation, and mass.

    :param config: Values from input config file or user inputs.
    :type config: dict
    :return: List of bins that have been binned according to the config settings.
    """
    
    catalog = Table.read(config["CATALOG"], format='fits')
    
    mags = np.array(catalog[config["MAG_KEY"]])
    r50s = np.array(catalog[config["R50_KEY"]])
    ns = np.array(catalog[config["N_KEY"]])
    ellips = np.array(catalog[config["ELLIP_KEY"]])

    catalog_objects = [mags, r50s, ns, ellips]
    column_names = ["MAGS", "R50S", "NS", "ELLIPS"]
    bin_bounds = []
    
    # Add our bin stuff in
    for i in range(len(config["BIN_NAMES"])):
        catalog_objects.append(np.array(catalog[config["BIN_NAMES"][i]]))
        column_names.append(config["BIN_NAMES"][i])
        bin_bounds.append(config["BIN_" + str(i)])
        
    rebin_indices = np.arange(4, 4 + len(config["BIN_NAMES"]))
    catalog_objects = np.transpose(np.asarray(catalog_objects))
    
    # Create our initial bin
    init_bin = Bin(objects=catalog_objects, object_column_names=column_names)

    binlist = BinList(bins=[init_bin])
    
    for index in range(len(config["BIN_NAMES"])):
        binlist.rebin(index + 4, config["BIN_" + str(index)])
    
#     check = binlist.rebin(4, config["BIN_0"])
    
#     print("Binning by mass finished", "\n")
#     check = binlist.rebin(5, config["BIN_1"])

    return binlist


def plot_bin_params(b, columns=4, filename=None, dpi=150):
    """ Generate a nice plot of a bin's structural parameters

    :param b: Input bin
    :type b: Bin
    :param columns: Number of columns per row, defaults to 4
    :type columns: int, optional
    :param filename: Output filename, defaults to None
        If none, plt.show() is run instead
    :type filename: str, optional
    :param dpi: DPI for saved image, defaults to 150
    :type dpi: int, optional
    """
    
    data = b.param_dict()
    data_names = [n for n in data]
    rows = int(np.ceil(len(data) / columns))
    
    fig, ax = plt.subplots(rows,columns, facecolor="white")
    fig.set_figwidth(columns * 3)
    fig.set_figheight(rows * 3)
    
    row, column = 0, 0
    
    for i in range(rows * columns):
        row, column = int(np.floor(i / columns)), i % columns
        if i >= len(data_names):
            ax[row][column].remove()
            continue
        name = data_names[i]
        ax[row][column].hist(data[name], histtype='step', color="black",
                            bins=25, lw=4)
        ax[row][column].set_title(name)
    
    for i in range(rows):
        ax[i][0].set_ylabel("Bin Size")
    
    plt.suptitle("Bin: " + str(b.bin_params), fontsize=20)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
    else:
        plt.show()

