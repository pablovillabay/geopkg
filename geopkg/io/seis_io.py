import pandas as pd
import numpy as np
from scipy import interpolate
import os, sys


def pseudo_wells_model(zmin, zmax, sr, no_wells, zones={}, zones_ss={}, depth='Depth', zone_idx='Zone_idx', zone_col='Zone'):


    depth_log = np.arange(zmin, zmax, sr)
    pseudo_wells = pd.DataFrame(np.zeros((len(depth_log), no_wells)))
    pseudo_wells[depth] = depth_log

    zones_df = pd.DataFrame()
    zones_df[depth] = [float(i) for i in zones.values()]
    zones_df[zone_col] = zones.keys()
    pseudo_wells = pd.merge_asof(pseudo_wells, zones_df, on=depth)

    zone_dict = dict(zip(pseudo_wells[zone_col].unique(), [int(i) for i in range(len(pseudo_wells[zone_col].unique()))]))
    pseudo_wells[zone_idx] = pseudo_wells[zone_col].map(zone_dict)
    
    for zone in zones_ss.keys(): 
        if zones_ss[zone] != 0:  
            for well in range(no_wells): 
                ntg = 100* (well) / (no_wells - 1) 
                zone_list = pseudo_wells[pseudo_wells[zone_col] == zone][well].values 
                
                locs = [] 
                for i in range(zones_ss[zone]): 
                    if zones_ss[zone] > 1:
                        locs.append(int((len(zone_list)-1) * i/(zones_ss[zone]-1))) 
                    else:
                        locs.append(0)
    
                ones = 1  
                while (sum(zone_list)/len(zone_list)) < ntg/100:  
                    zone_list = 0 * zone_list 
                    disp = np.ones(ones)  
                    
                    if zones_ss[zone] == 1: 
                        zone_list[0:ones] = disp  
                        
                    else:  
                        for i in range(len(locs)):  
                            if i == 0:
                                zone_list[0:ones] = disp  
                            elif i == len(locs)-1:
                                zone_list[-ones:] = disp  
                                break
                            else:
                                insert = int(locs[i]-(len(disp)/2))
                                zone_list[insert:insert+len(disp):1] = disp  
                    ones += 1  

                ind = 0
                for idx, row in  pseudo_wells[pseudo_wells[zone_col] == zone].iterrows():
                    pseudo_wells.loc[row.name, well] = zone_list[ind] 
                    ind += 1
    
    return pseudo_wells



def dict_mapper(row, sand, shale, no_wells, zone_col):


    for i in range(no_wells):
        if row[i] == 0:
            row[i] = sand[row[zone_col]]
        else:
            row[i] = shale[row[zone_col]]
    
    return row



def property_mapper(pseudo_wells, sand_density, shale_density, sand_vp, shale_vp, sand_vs, shale_vs, zone_col='Zone'):


    no_wells = len(pseudo_wells.columns) - 3
    density = pseudo_wells.apply(dict_mapper, args=(sand_density, shale_density, no_wells, zone_col), axis=1)
    vp = pseudo_wells.apply(dict_mapper, args=(sand_vp, shale_vp, no_wells, zone_col), axis=1)
    vs = pseudo_wells.apply(dict_mapper, args=(sand_vs, shale_vs, no_wells, zone_col), axis=1)

    return density, vp, vs



def time_model(pseudo_wells, density, vp, vs, wcs_file, skip=1, zones={}, time='Time', depth='Depth', zone_idx='Zone_idx', zone='Zone'):


    wcs = np.loadtxt(wcs_file, skiprows=skip)
    idx1 = (np.abs(np.asarray(wcs[:,0]) - pseudo_wells[depth].min())).argmin()  
    idx2 = (np.abs(np.asarray(wcs[:,0]) - pseudo_wells[depth].max())).argmin()
    
    time_frame = np.arange(np.around(wcs[idx1,1], decimals=0), np.around(wcs[idx2,1], decimals=0), 2)  
    depth_frame = time_frame * 0   
    for i in range(len(depth_frame)):
        idx = (np.abs(np.asarray(wcs[:,1]) - time_frame[i])).argmin() 
        depth_frame[i] = np.around(wcs[idx,0], decimals=0)
    
    df_sampled = pd.DataFrame()
    df_sampled[depth] = depth_frame
    df_sampled[time] = time_frame

    dens_twt = pd.DataFrame() 
    vp_twt = pd.DataFrame()
    vs_twt = pd.DataFrame()
    dens_twt[[time,depth]] = df_sampled[[time,depth]] 
    vp_twt[[time,depth]] = df_sampled[[time,depth]]
    vs_twt[[time,depth]] = df_sampled[[time,depth]]

    for i, row in dens_twt.iterrows():  
        if i > 0:  
            dens_ = density[(density[depth] >= dens_twt.loc[i-1, depth]) & (density[depth] < dens_twt.loc[i, depth])]
            vp_ = vp[(vp[depth] >= vp_twt.loc[i-1, depth]) & (vp[depth] <= vp_twt.loc[i, depth])]
            vs_ = vs[(vs[depth] >= vs_twt.loc[i-1, depth]) & (vs[depth] <= vs_twt.loc[i, depth])]
            for j in range(len(pseudo_wells.columns)-3):
                dens_twt.at[i, j] = dens_.mean()[j]
                dens_twt.at[i, zone_idx] = dens_.min()[zone_idx]
                dens_twt.at[i, zone] = list(zones.keys())[int(dens_.min()[zone_idx])]
                vp_twt.loc[i, j] = vp_.mean()[j]
                vp_twt.loc[i, zone_idx] = vp_.min()[zone_idx]
                vp_twt.loc[i, zone] = list(zones.keys())[int(vp_.min()[zone_idx])]
                vs_twt.loc[i, j] = vs_.mean()[j]
                vs_twt.loc[i, zone_idx] = vs_.min()[zone_idx]
                vs_twt.at[i, zone] = list(zones.keys())[int(vs_.min()[zone_idx])]

    dens_twt.loc[0,:] = dens_twt.loc[1,:] 
    vp_twt.loc[0,:] = vp_twt.loc[1,:]
    vs_twt.loc[0,:] = vs_twt.loc[1,:]

    return df_sampled, dens_twt, vp_twt, vs_twt



def shuey(df_sampled, vp_twt, vs_twt, dens_twt, no_wells, angles, time='Time', depth='Depth'):


    r0_twt = pd.DataFrame()
    G_twt = pd.DataFrame()
    F_twt = pd.DataFrame()
    r0_twt[[time,depth]] = df_sampled[[time,depth]]
    G_twt[[time,depth]] = df_sampled[[time,depth]]
    F_twt[[time,depth]] = df_sampled[[time,depth]]

    for i, row in df_sampled.iterrows():
        if i > 0: 
            for j in range(no_wells):
                dens_ = (dens_twt.loc[i,j] + dens_twt.loc[i-1,j]) / 2 
                vp_ = (vp_twt.loc[i,j] + vp_twt.loc[i-1,j]) / 2
                vs_ = (vs_twt.loc[i,j] + vs_twt.loc[i-1,j]) / 2
                dens_term = (dens_twt.loc[i,j] - dens_twt.loc[i-1,j]) / dens_  
                vp_term = (vp_twt.loc[i,j] - vp_twt.loc[i-1,j]) / vp_
                vs_term = (vs_twt.loc[i,j] - vs_twt.loc[i-1,j]) / vs_
                r0_twt.loc[i, j] = 0.5 * (vp_term + dens_term) 
                G_twt.loc[i,j] = 0.5 * vp_term - 2 * (vp_twt.loc[i,j]/vs_twt.loc[i,j])**2 * (dens_term + 2 * vs_term)
                F_twt.loc[i,j] = 0.5 * vp_term 
    r0_twt.loc[0,:] = r0_twt.loc[1,:] 
    G_twt.loc[0,:] = G_twt.loc[1,:]
    F_twt.loc[0,:] = F_twt.loc[1,:]
    
    reflectivity = np.zeros((len(r0_twt), len(angles), no_wells))  
    for i in range(1, len(r0_twt)-1):
        for j in range(len(angles)):
            for k in range(no_wells):
                reflectivity[i-1,j,k] = r0_twt.loc[i,k] + G_twt.loc[i,k] * np.sin(np.radians(angles[j]))**2 
                reflectivity[i-1,j,k] += F_twt.loc[i,k]*(np.tan(np.radians(angles[j]))**2 - np.sin(np.radians(angles[j]))**2)
    
    return r0_twt, G_twt, F_twt, reflectivity



def ricker(f, length=128, dt=2):
    
    
    length = length / 1000
    dt = dt / 1000

    t = np.arange(-length/2, (length-dt)/2, dt)  
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2)) 

    return 1000*t, y 



def seis_convolve(reflectivity, wavelet, model, angles=None, time='Time', depth='Depth', zone_idx='Zone_idx', zone='Zone'):


    amplitude = 0 * reflectivity
    
    for j in range(reflectivity.shape[1]):
        for k in range(reflectivity.shape[2]):
            amplitude[:, j, k] = np.convolve(reflectivity[:,j,k], wavelet, mode='same')
    
    stack = pd.DataFrame()
    stack[[depth, time, zone, zone_idx]] = model[[depth, time, zone, zone_idx]]
    nears = pd.DataFrame()
    nears[[depth, time, zone, zone_idx]] = model[[depth, time, zone, zone_idx]]
    fars = pd.DataFrame()
    fars[[depth, time, zone, zone_idx]] = model[[depth, time, zone, zone_idx]]
    for k in range(len(model.columns)-3):
        stack[k] = 0
        nears[k] = 0
        fars[k] = 0
    
    if angles is not None:
        for k in range(reflectivity.shape[2]):
            for j in range(reflectivity.shape[1]):
                stack[k] = stack[k] + amplitude[:,j,k]
                if angles[j] <= 10:
                    nears[k] = nears[k] + amplitude[:,j,k]
                elif angles[j] >= 20:
                    fars[k] = fars[k] + amplitude[:,j,k]

    return amplitude, stack, nears, fars



def create_wedge(tmin, tmax, sr, no_traces, top, pad, thick, wedge_dens, wedge_vp, wedge_vs, outer_dens, outer_vp, outer_vs, start, stop, wavelet):


    time_log = np.arange(tmin, tmax, sr)  
    wedge = pd.DataFrame(np.zeros((len(time_log), no_traces))) 
    wedge['TWT'] = time_log

    for idx, row in wedge.iterrows():
        if row.TWT > top and row.TWT < top+thick:
            for well in range(no_traces):
                if well > pad*no_traces:
                    if row.TWT <= top + thick * (well - pad*no_traces) / ((1-pad)*no_traces):
                        wedge.loc[idx, well] = 1

    avg_dens = 1/2 * (wedge_dens + outer_dens)
    avg_vp = 1/2 * (wedge_vp + outer_vp)
    avg_vs = 1/2 * (wedge_vs + outer_vs)

    dens_top = (wedge_dens - outer_dens) / avg_dens
    vp_top = (wedge_vp - outer_vp) / avg_vp
    vs_top = (wedge_vs - outer_vs) / avg_vs
    r0_top = 0.5 * (vp_top + dens_top) 
    g_top = 0.5 * vp_top - 2 * (wedge_vp/wedge_vs)**2 * (dens_top + 2 * vs_top)
    f_top = 0.5 * vp_top 

    dens_base = -dens_top
    vp_base = -vp_top
    vs_base = -vs_top
    r0_base = 0.5 * (vp_base + dens_base) 
    g_base = 0.5 * vp_base - 2 * (outer_vp/outer_vs)**2 * (dens_base + 2 * vs_base)
    f_base = 0.5 * vp_base

    angles = np.arange(start, stop)  
    reflectivity = np.zeros((len(wedge), len(angles), no_traces))  
    amplitude = 0 * reflectivity  
    wedge_stk = np.zeros((len(wedge), no_traces))  

    for k in range(no_traces):
        if k> pad*no_traces:
            top_idx = wedge.loc[wedge[k]==1,k].index[0]  
            base_idx = wedge.loc[wedge[k]==1,k].index[-1]
            for j in range(len(angles)):
                theta = np.radians(angles[j])
                reflectivity[top_idx,j,k] = r0_top + g_top * np.sin(theta)**2 + f_top*(np.tan(theta)**2 - np.sin(theta)**2)
                reflectivity[base_idx,j,k] = r0_base + g_base * np.sin(theta)**2 + f_base*(np.tan(theta)**2 - np.sin(theta)**2)
                amplitude[:, j, k] = np.convolve(reflectivity[:,j,k], wavelet, mode='same')  
    for j in range(len(angles)):
        wedge_stk += amplitude[:, j, :] 
    
    return no_traces, wedge, amplitude, wedge_stk



def syn_well(zmin, zmax, sr, wcs_path, logs, wcs_names=['MD', 'TWT'], skiprows=0, skipna=-999.25):

    depth_well = logs.copy(deep=True)

    wcs = pd.read_csv(wcs_path, delim_whitespace=True, names=wcs_names, skiprows=skiprows, na_values=skipna)
    idx1 = (np.abs(wcs['MD'] - zmin)).idxmin()  
    idx2 = (np.abs(wcs['MD'] - zmax)).idxmin()

    time_frame = np.arange(np.around(wcs.loc[idx1,'TWT'], decimals=0), np.around(wcs.loc[idx2,'TWT'], decimals=0), sr)
    time_well = pd.DataFrame()
    time_well['TWT'] = time_frame

    interp_func = interpolate.interp1d(wcs['TWT'].values, wcs['MD'].values)
    time_well['MD'] = time_well.apply(lambda row: np.around(interp_func(row['TWT']), decimals=3), axis=1)

    for col in ['Vp', 'Vs', 'RHOB']: time_well[col] = 0.0
    for i, row in time_well.iterrows():
        if i > 0: 
            log_slice = depth_well[(depth_well.MD >= time_well.loc[i-1, 'MD']) & (depth_well.MD < time_well.loc[i, 'MD'])]
            time_well.loc[i, ['Vp', 'Vs', 'RHOB']] = log_slice[['Vp', 'Vs', 'RHOB']].mean().values
    time_well.loc[0, ['Vp', 'Vs', 'RHOB']] = time_well.loc[1, ['Vp', 'Vs', 'RHOB']]
    time_well['VpVs'] = time_well['Vp'] / time_well['Vs']
    time_well['PImp'] = time_well['Vp'] * time_well['RHOB']

    for col in ['R0', 'G', 'F']: time_well[col] = 0.0
    for i, row in time_well.iterrows():
        if i > 0: 
            dens_ = (time_well.loc[i,'RHOB'] + time_well.loc[i-1,'RHOB']) / 2  
            vp_ = (time_well.loc[i,'Vp'] + time_well.loc[i-1,'Vp']) / 2
            vs_ = (time_well.loc[i,'Vs'] + time_well.loc[i-1,'Vs']) / 2
            dens_term = (time_well.loc[i,'RHOB'] - time_well.loc[i-1,'RHOB']) / dens_ 
            vp_term = (time_well.loc[i,'Vp'] - time_well.loc[i-1,'Vp']) / vp_
            vs_term = (time_well.loc[i,'Vs'] - time_well.loc[i-1,'Vs']) / vs_
            time_well.loc[i, 'R0'] = 0.5 * (vp_term + dens_term) 
            time_well.loc[i,'G'] = 0.5 * vp_term - 2 * (time_well.loc[i,'Vp']/time_well.loc[i,'Vs'])**2 * (dens_term + 2 * vs_term)
            time_well.loc[i,'F'] = 0.5 * vp_term 
    time_well.loc[0, ['R0', 'G', 'F']] = time_well.loc[1, ['R0', 'G', 'F']]
