import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.style.use('seaborn')



def well_track(plot_logs, depth, zmin, zmax, zones, zone_colors=[], tracks={}, figsize=(20,10)):

    """

    Display a one-well log section defined with a 2-level dict

    Parameters:
        plot_logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        depth (string): name of the depth column in the logs df
        zmin (float): top of the well section
        zmax (float): base of the well section
        zone_colors ([string-colors]): list of matplotlib named colors for zone track coloring
        tracks (dict_dict): 2 level dict, outer defines tracks, inned defines log curves. The inner dict values are
                            lists with curve display settings and a 'core' indicator when needed
        figsize (tuple_int): same keyword argument as in matplotlib

    Returns:
        fig, ax: matplotlib's fig and axes array

    """

    plot_logs = plot_logs[(plot_logs[depth]>zmin) & (plot_logs[depth]<zmax)]
    plot_logs.reset_index(inplace=True)

    fig, ax = plt.subplots(1, len(tracks.keys())+1, figsize=figsize)

    cmap_zones = colors.ListedColormap(zone_colors[0:len(zone_colors)], 'indexed')
    zone_mapper = dict(zip(plot_logs[zones].unique(), range(len(plot_logs[zones].unique()))))
    zone_map = np.repeat(np.expand_dims(plot_logs[zones].map(zone_mapper).values, 1), 100, 1)
    ax[-1].imshow(zone_map, interpolation='none', aspect='auto', cmap=cmap_zones)
    ax[-1].set_xlabel('Zone', fontsize=14)
    ax[-1].get_xaxis().set_ticklabels([])
    ax[-1].get_yaxis().set_visible(False)

    ymin, _ = ax[-1].get_ylim()
    ax[-1].set_xlim(0,2)
    ax[-1].set_xlabel('Zone')

    zone_list = []
    for zone_ in plot_logs[zones].unique():
        zone_list.append((plot_logs[zones].values == zone_).argmax())
    zone_list.append(plot_logs.index.max())

    depth_list = []
    for i in range(len(zone_list)-1):
        depth_list.append(0.5*(zone_list[i] + zone_list[i+1]))
        
    for i in range(len(depth_list)):
        position = plot_logs.loc[int(depth_list[i]), depth]
        ax[-1].text(x = 0.5, y = (ymin+0.5)*(position-zmin)/(zmax-zmin), s = plot_logs[zones].unique()[i], fontsize=14)

    
    i = 0
    for track in tracks:
        curve = 'first'

        for log in tracks[track].keys():

            settings = tracks[track][log]
            if settings[-1] != 'core':
                if curve == 'first': 
                    for j in range(len(zone_list)-1):
                        top = np.array([1, 1]) * float(plot_logs.loc[int(zone_list[j]), depth]) 
                        ax[i].plot([settings[1], settings[2]], top, color='black')
                    curve = 'second'
                    ax_ = ax[i]
                else:
                    ax_ = ax[i].twiny()
                    ax_.xaxis.set_label_position('top')
                ax_.plot(plot_logs[log], plot_logs[depth], color=settings[3], linewidth=0.8)
                ax_.set_xscale(settings[0])
                ax_.set_xlim(settings[1], settings[2])
                ax_.set_xlabel(log, fontsize=14)
                
            else:
                ax[i].scatter(plot_logs[log], plot_logs[depth], c='black', marker='o', s=15)
                ax[i].set_xscale(settings[0])
                ax[i].set_xlim(settings[1], settings[2])
        i += 1
    
    for i in range(0, len(ax)-1):
        ax[i].set_ylim(zmin, zmax)
        ax[i].invert_yaxis()
        if i > 0:
            ax[i].get_yaxis().set_ticklabels([])

    fig.tight_layout()

    return fig, ax



pass



