import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
from seaborn import palettes

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib import cm

from geopkg.io.las_io import summarize_logs

plt.style.use('seaborn')



def single_cat_stats(logs, label, cpal, figsize=(8,6)):

    """

    Display a colored histogram for categorical/discrete variables

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        label (string): variable (logs column name) to create the histogram
        cpal (list:string): list of matplotlib color names for the categorical classes
        figsize (tuple_int): same keyword argument as in matplotlib

    Returns:
        df (pd.DataFrame): result data frame with computed histogram frequencies
        g (sns.axes): Seaborn axes figure

    """

    df = logs[[label]].dropna(how='any', axis=0)
    df = 100 * df[label].value_counts() / len(df)
    df = df.sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.barplot(x=df.index, y=df.values, palette=cpal, ax=ax)
    fig.suptitle('Histogram for ' + str(label), size=18)
    ax.set_ylabel('labelortion (%)')
    ax.set_xlabel(label)
    fig.tight_layout()
    
    return df, g



def double_cat_counts(df, label, column):

    """

    Calculate labelortion/fractions for a categorical variable with another categorical variable as 
    classifier

    Parameters:
        df (pd.DataFrame): dataframe containing the two categorical variables and no NaN values
        label (string): variable (logs column name) to create the histogram
        column (strin): name of the filtering column

    Returns:
        df3 (pd.DataFrame): result data frame with computed histogram frequencies

    """


    _2fold = df.groupby([column, label]).agg({label:'count'})
    _1fold = df.groupby([column]).agg({label:'count'})
    df = _2fold.div(_1fold, level=column) * 100

    df2 = df.unstack()
    df2.columns = df2.columns.droplevel(0).values
    df3 = pd.DataFrame(data=df2.values, columns=list(df2.columns))
    df3[column] = df2.index.values
    df3 = df3.melt(id_vars=column).sort_values([column, 'variable'], axis=0, ascending=[True, True])
    df3.reset_index(inplace=True, drop=True)

    return df3



def cat_stats(logs, label, filter, cpal, ylim=100):

    """

    Display colored histograms for a categorical variable given other two as a filters per figure column and row

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        label (string): variable (logs column name) to create the histogram
        filter (list:string): name of the filtering categories (1 or 2)
        cpal (list:string): list of matplotlib color names for the categorical classes
        ylim (float): max value for the vertical axis

    Returns:
        df3 (pd.DataFrame): result data frame with computed histogram frequencies
        g (sns.axes): Seaborn axes figure

    """   

    subset_ = []
    subset_.append(label)
    if len(filter) == 1: subset_.append(filter[0])
    else: [subset_.append(a) for a in filter]
    df = logs[subset_].dropna(axis=0, how='any')

    if len(filter) == 1:    
        df3 = double_cat_counts(df, label, filter[0])
        df3.rename(mapper={filter[0]:filter[0], 'variable':label, 'value':'fraction (%)'}, inplace=True, axis=1)
        
        g = sns.FacetGrid(df3, col=filter[0], height=3.5, margin_titles=True, 
                            gridspec_kws={'wspace': 0.03}, subplot_kws={'ylim':[0,ylim]})


    elif len(filter) == 2:     
        df_list = []
        sorted_cols = np.sort(list(logs[filter[1]].unique()))
        for elem in sorted_cols:
            df1 = double_cat_counts(df[df[filter[1]]==elem], label, filter[0])
            df1[filter[1]] = elem
            df1.rename(mapper={filter[1]:filter[1], 'variable':label, 'value':'fraction (%)'}, inplace=True, axis=1)
            df_list.append(df1)
        df3 = pd.concat(df_list)
        
        g = sns.FacetGrid(df3, row = filter[0], col=filter[1], height=3.5, margin_titles=True, 
                            gridspec_kws={'wspace': 0.03}, subplot_kws={'ylim':[0,ylim]})
    
    g.map(sns.barplot, label, 'fraction (%)', palette=cpal, order=np.sort(df3[label].unique()))
    plt.show()

    return df3



def cat_hist(logs, label, filter, log_=False):


    """

    Displays histrgrams and kde pdf's for continuous variables classified with 
    single or double fold categories

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        label (string): variable (logs column name) to create the histogram
        filter (list:string): name of the filtering categries
        log_ (bool): option to turn x axis to logarithmic scale

    """   
        
    subset_ = []
    subset_.append(label)
    if len(filter) == 1: subset_.append(filter[0])
    else: [subset_.append(a) for a in filter]
    df = logs[subset_].dropna(axis=0, how='any')

    xlims = [df[label].min(), df[label].max()]
    if log_ == True: xlims = [round(np.log10(df[label].min())), round(np.log10(df[label].max()))]

    plot_no = 1
    row_no = 1 
    col_no = 1
    if len(filter) == 2:
        fig = plt.figure(figsize=(20, 20)) 
        ax = []  
        for row in np.sort(df[filter[0]].unique()):
            for col in np.sort(df[filter[1]].unique()):
                ax.append(plt.subplot(len(df[filter[0]].unique()), len(df[filter[1]].unique()), plot_no))
                data_array = df[(df[filter[0]]==row) & (df[filter[1]]==col)][label].values
                if log_ == True: data_array = np.log10(data_array)
                
                sns.distplot(data_array, hist=True, kde=True, bins=20, color = 'darkred', ax=ax[plot_no-1],
                             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})

                ax[plot_no-1].set_xlim(xlims[0], xlims[1])
                
                if row_no != len(df[filter[0]].unique()): ax[plot_no-1].get_xaxis().set_ticklabels([])
                else: ax[plot_no-1].set_xlabel(str(filter[1]) + ': ' + str(col), fontsize=12)
                        
                if col_no != 1: ax[plot_no-1].get_yaxis().set_ticklabels([])
                else: ax[plot_no-1].set_ylabel(str(filter[0]) + ': ' + str(row), fontsize=12)  # the first column also reports row classes
                    
                if log_ == True and row_no == len(df[filter[0]].unique()):
                    ax[plot_no-1].set_xticklabels([int(10**i) for i in np.linspace(xlims[0], xlims[1], len(ax[plot_no-1].get_xticklabels()))])
                plot_no += 1
                col_no +=1
            col_no = 1
            row_no += 1
    
    elif len(filter) == 1: 
        fig = plt.figure(figsize=(20, 5)) 
        ax = []  
        for col in np.sort(df[filter[0]].unique()):
                ax.append(plt.subplot(1, len(df[filter[0]].unique()), plot_no))
                data_array = df[df[filter[0]]==col][label].values
                if log_ == True: data_array = np.log10(data_array)
                    
                sns.distplot(data_array, hist=True, kde=True, bins=20, color = 'darkred', ax=ax[plot_no-1],
                             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})

                ax[plot_no-1].set_xlabel(str(filter[0]) + ': ' + str(col), fontsize=12)
                ax[plot_no-1].set_xlim(xlims[0], xlims[1])
                
                if col_no != 1: ax[plot_no-1].get_yaxis().set_ticklabels([])   
                if log_ == True: 
                    ax[plot_no-1].set_xticklabels([int(10**i) for i in np.linspace(xlims[0], xlims[1], len(ax[plot_no-1].get_xticklabels()))])
                    
                plot_no += 1
        
    fig.suptitle(label, fontsize=20, y=1.00)
    fig.tight_layout()
    plt.show()



def class_xplot(logs, x_, y_, cat=None, cpal=None,  log_x=False, log_y=False, lim_x=None, lim_y=None, shade=0.0):
    
    """

    Tool for custom crossplotting well log data with the possibility of plotting by discrete class (e.g. facies) and 
    display also shaded 2D pdf areas behind the crossplot

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, use appropiate filters in advance
        x_ (string): name of the x axis variable
        y_ (string): name of the y axis variable
        cat (string): name of the discrete variable to classify, leave None for 1-class crossplots
        cpal (list:string): color list for discrete class color coding
        log_x, log_y (bool): turn axis into logarithmic scale
        lim_x, lim_y (list:float): customize axis limits
        shade (float): turn on and select opacity for shaded 2D pdf display
    
    """   
    

    rect_scatter = [0.1, 0.1, 0.65, 0.65]
    rect_histporo = [0.1, 0.77, 0.65, 0.2]
    rect_histperm = [0.77, 0.1, 0.2, 0.65]

    fig = plt.figure(1, figsize=(10,10)) 
    ax_scatter = plt.axes(rect_scatter)  
    ax_histhor = plt.axes(rect_histporo)
    ax_histvert = plt.axes(rect_histperm)
    
    if cat != None: 
        df = logs[[x_, y_, cat]].dropna(axis=0, how='any')
        z_categories = list(df[cat].cat.categories)  
        color_dict = dict(zip(z_categories, cpal))
        ax_scatter.scatter(df[x_], df[y_], c=df[cat].map(color_dict), edgecolors=None, s=10)
        circles=[Line2D(range(1), range(1), color='w', marker='o', markersize=10, markerfacecolor=item) for item in cpal]
        leg = plt.legend(circles, z_categories, loc = "lower left", bbox_to_anchor = (1, 0.5), numpoints = 1, title=cat)
    else: 
        df = logs[[x_, y_]].dropna(axis=0, how='any')
        ax_scatter.scatter(df[x_], df[y_], c='darkred', edgecolors=None, s=10)
        z_categories = ['ALL']

    if len(df[x_].values > 5000): df = df.sample(5000) 

    df2 = df[(df[x_] < df[x_].quantile(0.99)) & (df[y_] < df[y_].quantile(0.99)) & 
        (df[x_] > df[x_].quantile(0.01)) & (df[y_] > df[y_].quantile(0.01))]
    
    if lim_x is not None: 
        ax_scatter.set_xlim(lim_x)
    if lim_y is not None:
        ax_scatter.set_ylim(lim_y)
    
    ax_histhor.set_xlim(ax_scatter.get_xlim()) 
    ax_histvert.set_ylim(ax_scatter.get_ylim())
    
    if log_x: 
        ax_scatter.set_xscale('log')
        df[x_] = np.log10(df[x_]) 
        ax_histhor.set_xlim(np.log10(ax_scatter.get_xlim()))  
    if log_y:
        ax_scatter.set_yscale('log')
        df[y_] = np.log10(df[y_])
        ax_histvert.set_ylim(np.log10(ax_scatter.get_ylim()))
    
    if cat != None:
        for idx_cat in range(len(z_categories)):
            z_cat = z_categories[idx_cat]
            sns.distplot(df[df[cat]==z_cat][x_], hist=True, kde=True, bins=20, color = color_dict[z_cat], ax=ax_histhor,
                                norm_hist =True, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2})
            sns.distplot(df[df[cat]==z_cat][y_], hist=True, kde=True, bins=20, color = color_dict[z_cat], ax=ax_histvert,
                                norm_hist =True, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, vertical=True)
            
            if shade > 0.0:
                vals = np.ones((256, 4))
                for i in range(3): 
                    vals[:, i] = np.linspace(colors.to_rgba(color_dict[z_cat])[i], 1, 256)
                newcmp = ListedColormap(vals)
                sns.kdeplot(df2[df2[z_]==z_cat][x_], df2[df2[z_]==z_cat][y_], cmap=newcmp, shade=True, 
                            shade_lowest=False, ax=ax_scatter, alpha=shade, zorder=0)
    else:
        sns.distplot(df[x_], hist=True, kde=True, bins=20, color = 'darkred', ax=ax_histhor,
                                norm_hist =True, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2})
        sns.distplot(df[y_], hist=True, kde=True, bins=20, color = 'darkred', ax=ax_histvert,
                                norm_hist =True, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, vertical=True)
    
    ax_scatter.set_xlabel(x_, fontsize=14)
    ax_scatter.set_ylabel(y_, fontsize=14)
    ax_histhor.set_xlabel('')
    ax_histhor.set_ylabel('')
    ax_histvert.set_xlabel('')
    ax_histvert.set_ylabel('')    
       
    ax_histhor.get_xaxis().set_ticklabels([])   
    ax_histhor.get_yaxis().set_ticklabels([])   
    ax_histvert.get_xaxis().set_ticklabels([])   
    ax_histvert.get_yaxis().set_ticklabels([]) 
     
    fig.suptitle(x_ + ' vs ' + y_, fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()



def summary_boxplot(logs, wells, cat, props, cpal, sr=0.5):

    """

    Display well petrophysical summaries as boxplots per category and property for a quicklook overview of the entire
    petrophysical results and their statistical spread

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, use appropiate filters in advance to provide net logs
        wells (string): column with well names in the logs dataframe
        cat (string): column with filtering discrete property in the logs dataframe
        props (list:string): list of properties (logs) to be summarized
        cpal (list:string): list of named colors per filtering property
        sr (float): log sampling rate in project units for net thickness calculations
    
    """   
    
    df1 = summarize_logs(logs, wells, cat, props) 
    
    fig, ax = plt.subplots(len(df1.columns)-2, 1, sharex=True, figsize=(18,15))
    
    box_array = []  
    for i in range(len(ax)): 
        data_array = [] 
        for cat_ in df1[cat].unique():
            data_array.append(df1[df1[cat]==cat_][df1.columns[i+2]].dropna().values)
        new_box = ax[i].boxplot(data_array, labels=df1[cat].unique(), medianprops = dict(linestyle='--', linewidth=2, color='black'), 
                      showmeans=True, meanprops = dict(marker='o', markersize=8, markerfacecolor='black'), patch_artist=True)
        box_array.append(new_box)  
        ax[i].set_ylabel(df1.columns[i+2], fontdict={'fontsize':20}) 


    for bplot in box_array:
        for patch, color in zip(bplot['boxes'], cpal):
            patch.set_facecolor(color) 

    fig.suptitle('Property summary - Well Logs', fontsize=20, y=1.00)
    ax[-1].set_xticklabels(ax[-1].get_xticklabels(), fontdict={'fontsize':20})

    fig.tight_layout(pad=1.5)
    plt.show()