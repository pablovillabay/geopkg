import numpy as np
import pandas as pd

from geopkg.io.file_length import file_length


def read_tops(path, skip=-999.25000, delimiter='\t', header=0, colnames=[]):

    """

    Read well tops file into dataframe

    Parameters:
        path (string): path to the well tops file
        skip (): NaN identifier in the selected file
        delimiter (char): character or list of characters used to separate file columns
        header (int): no. of lines to skip at the begining of the file
        colnames (list:string): names of the columns to be imported

    Returns:
        tops (pd.DataFrame): direct import of the text file into pandas
        
    """

    tops = pd.read_csv(path, delimiter=delimiter, header=header, names=colnames)
    
    return tops



def merge_tops(logs, tops, surfcol, wellcol, depth_index, clip_zones):

    """

    creates a stratigraphic zone column in the well logs dataframe

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs
        tops (pd.DataFrame): dataframe containing well tops
        surfcol (string): name of the surfaces/horizons column in the well tops dataframe
        wellcol (string): name of the well column in the well tops AND well logs dataframes
        depth_index (string): name of the depth column in the well tops AND well logs dataframes
        clip_zones (list:string): stratigraphic zones to be discarded from the well logs dataframe
        
    """

    for well in logs[wellcol].unique():
        if well in tops[wellcol].unique():
            
            well_logs = logs[logs[wellcol]==well]
            well_logs.reset_index(inplace=True)
            well_tops = tops[tops[wellcol]==well]

            for _, row in well_tops.iterrows():
                closest_md = well_logs.loc[np.argmin(np.abs(well_logs[depth_index] - row[depth_index])), depth_index]
                curr_index = logs[(logs[depth_index]==closest_md) & (logs[wellcol]==well)].index
                logs.loc[curr_index, surfcol] = row[surfcol]
    
    logs[surfcol] = logs[surfcol].fillna(method='ffill')
    logs = logs[~logs[surfcol].isin(clip_zones)]

    return logs


