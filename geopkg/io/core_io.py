import numpy as np
import pandas as pd

from geopkg.io.file_length import file_length


def read_core(path, skip=-999.25000, delimiter='\t', header=0):

    """

    Read core data file into dataframe (CCAL/RCAL data only)

    Parameters:
        path (string): path to the core data file
        skip (): NaN identifier in the selected file
        delimiter (char): character or list of characters used to separate file columns
        header (int): no. of lines to skip at the begining of the file

    Returns:
        ccal (pd.DataFrame): direct import of the text file into pandas
        
    """

    ccal = pd.read_csv(path, delimiter=delimiter, header=header)
    
    return ccal



def merge_core(logs, ccal, wellcol, depth_index, transfer_cols=[]):

    """

    Creates new columns in the well logs dataframe with merged data from the core dataframe 
    based on closest depth criteria

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs
        ccal (pd.DataFrame): dataframe containing core data
        wellcol (string): name of the well column in the core AND well logs dataframes
        depth_index (string): name of the depth column in the core AND well logs dataframes
        transfer_cols (list:string): names of the columns to be merged from the core to the logs dataframes
        
    """

    for col in transfer_cols:
        for well in logs[wellcol].unique():
            if well in ccal[wellcol].unique():

                well_logs = logs[logs[wellcol]==well]
                well_logs.reset_index(inplace=True)
                well_core = ccal[ccal[wellcol]==well]

                for _, row in well_core.iterrows():

                    closest_md = well_logs.loc[np.argmin(np.abs(well_logs[depth_index] - row[depth_index])), depth_index]
                    curr_index = logs[(logs[depth_index]==closest_md) & (logs[wellcol]==well)].index
                    logs.loc[curr_index, col] = row[col]

    return logs
