import numpy as np
import pandas as pd
import os

from geopkg.io.file_length import file_length


def read_las(path, skip=-999.25000):

    """

    Read LAS files into numpy arrays

    Parameters:
        path (string): path to the LAS file
        skip (): NaN identifier in the selected file

    Returns:
        log_map (np.array): a two column matrix with log type (mnemonic) and unit respectively
        log_matrix (np.array): n-column matrix with all the logs stored as columns (typically col-1 will have MD values)

    """

    file_lines = file_length(path) 
    log_map = np.empty([0, 2])  

    with open(path) as file:
        j = 0
        section = 'foo'

        for i, line in enumerate(file):
            if i - 1 == file_lines: 
                break

            else:
                if line[0] == '~':  
                    if line[1] == 'C': 
                        section = 'curves'
                    elif line[1] == 'A':  
                        section = 'ascii'
                        log_matrix = np.empty([file_lines - i - 1, log_map.shape[0]]) 

                elif section == 'curves':
                    if line[0] == '#' or line[0] == '~': 
                        pass
                    else:
                        curve_data = line.split() 
                        log_map = np.vstack([log_map, [curve_data[0], curve_data[1][1:]]])

                elif section == 'ascii':
                    if line[0] == '#' or line[0] == '~':
                        pass
                    else:
                        log_data = line.split()
                        for curve in log_data:
                            curve = float(curve)
                        log_matrix[j, :] = log_data
                        j += 1

    for i in range(log_matrix.shape[0]):
        for j in range(log_matrix.shape[1]):
            log_matrix[i, j] = np.nan if log_matrix[i, j] == skip else log_matrix[i, j] 

    return log_map, log_matrix


def las2pandas(path, wellname, skip=-999.25000):

    """

    Read LAS files into pandas dataframe, logs as columns and irrelevant index

    Parameters:
        path (string): path to the LAS file
        wellname (string): name of the well
        skip (): NaN identifier in the selected file

    Returns:
        data (pd.DataFrame): a dataframe with logs as columns plus extra well name column

    """

    
    map_, matrix = read_las(path, skip) 
    data = pd.DataFrame(matrix) 
    data.columns = map_[:, 0]
    data.insert(0, 'Well', wellname)
    
    return data


def las_multi(path, skip=-999.25000):

    """

    Parse all files within a folder and import all LAS files into a single multi-well dataframe. Assumes the file names to be well names too.

    Parameters:
        path (string): path to the LAS file
        skip (): NaN identifier in the selected file

    Returns:
        logs (pd.DataFrame): all logs from all wells in a single pandas dataframe

    """

    logs = pd.DataFrame() 

    for root, dirs, files in os.walk(path):
        for file in files: 
            if file.split(sep='.')[1] in ['las', 'LAS']:
                wellname = file.split(sep='.')[0] 
                filename = os.path.join(root, file) 
                logs = pd.concat([las2pandas(filename, wellname, skip=-999.250000), logs], axis=0, ignore_index=True, sort=False)

    return logs


def logs_categorical(logs, gral_mapper={}):

    """

    Set all the categorical logs as such with string names instead

    Parameters:
        logs (pd.DataFrame): well logs dataframe
        gral_mapper (dict:dict): dict of dicts, the keys of the shallower dict level 
            must be the categorical log names as in the well logs dataframe and the values
            are the inner dicts. The inner level is another dict mapping floats to strings 
            as in the categorical log legends i.e.: {1.0:'Facies1', 2.0:'Facies20, ...} in 
            the desired order of hierarchy

    """

    for log in list(gral_mapper.keys()):
        logs[log] = logs[log].map(gral_mapper[log])
        logs[log] = logs[log].astype('category', copy=False)
        logs[log] = logs[log].cat.set_categories(list(gral_mapper[log].values()), ordered=True)

    return logs


def logs_save(logs, path):
    
    """

    Save the well logs df as a pickle file

    Parameters:
        logs (pd.DataFrame): well logs dataframe
        path (string): path to the save folder

    """

    save_path = os.path.join(path, 'well_logs.pkl')
    logs.to_pickle(save_path)


def logs_load(path):
    
    """

    Load a pickled well logs file as dataframe

    Parameters:
        path (string): path to the save folder

    Returns:
        logs (pd.DataFrame)

    """

    logs = pd.read_pickle(path)

    return logs

