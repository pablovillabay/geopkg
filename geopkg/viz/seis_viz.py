import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('seaborn')



def pseudo_well_view(pseudo_wells, zmin, zmax, depth='Depth', zone_idx='Zone_idx', zone='Zone', trace=None, xmin=0, xmax=100, title=''):


    plot_data = pseudo_wells[(pseudo_wells[depth] > zmin) & (pseudo_wells[depth] < zmax)]  
    plot_data.reset_index(inplace=True, drop=True) 
    no_wells = len(pseudo_wells.columns) - 3

    fig, ax = plt.subplots(nrows=1, ncols=no_wells+1, figsize=(15, 10)) 

    if trace is not None:
        trace.loc[:, range(no_wells)] = 0.75 * xmax * trace.loc[:, range(no_wells)] / np.max(np.abs(trace[range(no_wells)].values))

    for i in range(len(ax)-1): 
        if trace is not None:
            ax[i].plot(trace[i], trace[depth], color='black')
            ax[i].fill_betweenx(plot_data[depth], xmin, 2*xmax*plot_data[i]+xmin, color='khaki', alpha=0.5)
            ax[i].fill_betweenx(plot_data[depth], 2*xmax*plot_data[i]+xmin, xmax, color='olive', alpha=0.5)
            ax[i].set_xlim(xmin, xmax)
        else:
            ax[i].step(plot_data[i], plot_data[depth], color='black')  
            ax[i].fill_betweenx(plot_data[depth], 0, plot_data[i], color='khaki') 
            ax[i].fill_betweenx(plot_data[depth], plot_data[i], 2, color='olive')
            ax[i].set_xlim(0, 1)
        ax[i].get_xaxis().set_ticklabels([])
        if i > 0: ax[i].get_yaxis().set_ticklabels([])
        ax[i].set_ylim(zmin, zmax)
        ax[i].invert_yaxis()
        ax[i].set_xlabel(str(int(100 * i / (no_wells-1)))+'% NTG', fontsize=14)
        

    zone_map = np.repeat(np.expand_dims(plot_data[zone_idx].values, 1), 100, 1)
    im = ax[-1].imshow(zone_map, interpolation='none', aspect='auto', cmap='Set3') 
    ax[-1].set_xlabel('Zone', fontsize=14)
    ax[-1].get_xaxis().set_ticklabels([])
    ax[-1].get_yaxis().set_visible(False)

    ymin, ymax = ax[-1].get_ylim()  # this subplot is not MD referenced, we need some ymin, ymax to get an idea of its coordinates
    ax[-1].set_xlim(0,2)
    ax[-1].set_xlabel('Zone')

    zone_list = []
    for zone_ in plot_data[zone].unique():
        zone_list.append((plot_data[zone].values == zone_).argmax())
    zone_list.append(plot_data.index.max())

    depth_list = []
    for i in range(len(zone_list)-1):
        depth_list.append(0.5*(zone_list[i] + zone_list[i+1]))
    for i in range(len(depth_list)):
        position = plot_data.loc[int(depth_list[i]), depth]
        ax[-1].text(x = 0.5, y = (ymin+0.5)*(position-zmin)/(zmax-zmin), s = plot_data[zone].unique()[i], fontsize=14)

    fig.suptitle('Synthetic seismic modelling - '+title, y=1.02, fontsize=24)
    fig.tight_layout()
    plt.show()  



def wavelet_plot(time, wavelet, figsize=(9,7)):


    fig, ax = plt.subplots(figsize=(9,7))
    ax.plot(time, wavelet)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    fig.suptitle('Wavelet', fontsize=20)
    plt.show()



def pseudo_well_synthetic(zmin, zmax, pseudo_wells, density, vp, r0, amplitude, stack, trace, depth='Depth', zone_idx='Zone_idx', zone='Zone', angles=None):


    plot_data = pseudo_wells[(pseudo_wells[depth] > zmin) & (pseudo_wells[depth] < zmax)]
    plot_data.reset_index(inplace=True)
    dens_data = density[(density[depth] > zmin) & (density[depth] < zmax)]
    dens_data.reset_index(inplace=True)
    vp_data = vp[(vp[depth] > zmin) & (vp[depth] < zmax)]
    vp_data.reset_index(inplace=True)
    no_wells = len(pseudo_wells) - 3


    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 12), gridspec_kw={'width_ratios':[1, 1, 1, 1, 6]})

    zone_map = np.repeat(np.expand_dims(plot_data[zone_idx].values, 1), 100, 1)
    im = ax[0].imshow(zone_map, interpolation='none', aspect='auto', cmap='Set3')
    ax[0].set_xlabel('Zone')
    ax[0].get_xaxis().set_ticklabels([])
    ax[0].get_yaxis().set_ticklabels([])
    ymin, ymax = ax[0].get_ylim()
    ax[0].set_xlim(0,2)
    ax[0].set_xlabel('Zone')
    
    zone_list = []
    for zone_ in plot_data[zone].unique():
        zone_list.append((plot_data[zone].values == zone_).argmax())
    zone_list.append(plot_data.index.max())

    depth_list = []
    for i in range(len(zone_list)-1):
        depth_list.append(0.5*(zone_list[i] + zone_list[i+1]))
    for i in range(len(depth_list)):
        position = plot_data.loc[int(depth_list[i]), depth]
        ax[0].text(x = 0.5, y = (ymin+0.5)*(position-zmin)/(zmax-zmin), s = plot_data[zone].unique()[i], fontsize=14)
                
    ax[1].fill_betweenx(plot_data[depth], 0, plot_data[trace], color='khaki')
    ax[1].fill_betweenx(plot_data[depth], plot_data[trace], 2, color='olive')
    ax[1].set_xlim(0,1)
    ax[1].get_xaxis().set_ticklabels([])
    ax[1].get_yaxis().set_ticklabels([])
    ax[1].set_ylim(zmin, zmax)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(str(int(100 * trace / (no_wells-1)))+'% NTG')

    ax[2].plot(np.multiply(dens_data[trace], vp_data[trace]), plot_data[depth], color='purple')
    ax[2].get_xaxis().set_ticklabels([])
    ax[2].set_ylim(zmin, zmax)
    ax[2].invert_yaxis()
    ax[2].get_yaxis().set_ticklabels([])
    ax[2].set_xlabel('P-Impedance')

    ax[3].plot(stack[trace], stack[depth], color='black')
    ax[3].fill_betweenx( stack[depth], 0, stack[trace], where=stack[trace]>0, color='darkgrey')
    ax[3].fill_betweenx( stack[depth], 0, stack[trace], where=stack[trace]<0, color='white')
    ax[3].get_xaxis().set_ticklabels([])
    ax[3].set_xlim(-25,25)
    ax[3].set_ylim(zmin, zmax)
    ax[3].invert_yaxis()
    ax[3].get_yaxis().set_ticklabels([])
    ax[3].set_xlabel('Full stack')

    for j in range(len(angles)):
        ax[4].plot(amplitude[:,j,trace] + angles[j], r0[depth], color='dimgrey') 
    ax[4].pcolormesh(angles, r0[depth], amplitude[:, :, trace], vmin=-1., vmax=1., cmap='seismic', alpha=0.75)
    ax[4].set_xlim(angles[0]-2,angles[-1]+2)
    ax[4].set_ylim(zmin, zmax)
    ax[4].invert_yaxis()
    ax[4].get_yaxis().set_ticklabels([])
    ax[4].set_xlabel('Pseudo-gather -- AoI(deg)')


    fig.suptitle('Synthetic seismogram', y=0.92, fontsize=24)
    plt.show()



def avo_extractions(df_sampled, zones, angles, amplitude, no_wells, picked_zones=[]):


    idx = []
    for zone in picked_zones:
        idx.append((np.abs(np.asarray(df_sampled['Depth']) - zones[zone])).argmin())

    fig, ax = plt.subplots(nrows=1, ncols=len(idx), figsize=(18, 6), sharey=True)
    norm = mpl.colors.Normalize(vmin=100, vmax=0) 
    for i in range(no_wells):
        for j in range(len(idx)):
            ax[j].plot(angles, amplitude[idx[j],:,i], color=cm.copper((i) / (no_wells - 1)))  
            ax[j].set_xlim(0, 35)
            ax[j].set_ylim(-1.5, 1.5)
            ax[j].set_title('Top of {}'.format(picked_zones[j]), fontsize=12)
            ax[j].set_xlabel('Angle of incidence', fontsize='14')
            if j > 0: ax[j].set_yticklabels([])
    ax[0].set_ylabel('Relative amplitude', fontsize='14')
    sm = plt.cm.ScalarMappable(cmap=cm.copper, norm=norm) 
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.linspace(100,0,10))  
    cbar.ax.set_ylabel('NTG (%)', rotation=-90, va="bottom")
    fig.suptitle('Amplitude extraction by NTG', y=1.01, fontsize=16)

    plt.show()


