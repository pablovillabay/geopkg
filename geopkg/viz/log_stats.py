import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import palettes

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
        df1 (pd.DataFrame): result data frame with computed histogram frequencies
        g (sns.axes): Seaborn axes figure

    """

    df = logs[[label]].dropna(how='any', axis=0)
    df1 = 100 * df[label].value_counts() / len(df)
    df1 = df1.sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.barplot(x=df1.index, y=df1.values, palette=cpal, ax=ax)
    fig.suptitle('Histogram for ' + str(label), size=18)
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel(label)
    fig.tight_layout()
    
    return df1, g



def double_cat_counts(df, label, column):

    """

    Calculate proportion/fractions for a categorical variable with another categorical variable as 
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
    df1 = _2fold.div(_1fold, level=column) * 100

    df2 = df1.unstack()
    df2.columns = df2.columns.droplevel(0).values
    df3 = pd.DataFrame(data=df2.values, columns=list(df2.columns))
    df3[column] = df2.index.values
    df3 = df3.melt(id_vars=column).sort_values([column, 'variable'], axis=0, ascending=[True, True])
    df3.reset_index(inplace=True, drop=True)

    return df3



def double_cat_stats(logs, label, column, cpal, ylim=100):

    """

    Display colored histograms for a categorical variable given another as a filter per figure column

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        label (string): variable (logs column name) to create the histogram
        column (strin): name of the filtering column
        cpal (list:string): list of matplotlib color names for the categorical classes
        ylim (float): max value for the vertical axis

    Returns:
        df3 (pd.DataFrame): result data frame with computed histogram frequencies
        g (sns.axes): Seaborn axes figure

    """   
    
    df = logs[[label, column]].dropna(how='any', axis=0)
    df3 = double_cat_stats(df, label, column)
    df3.rename(mapper={column:column, 'variable':label, 'value':'fraction (%)'}, inplace=True, axis=1)
    
    g = sns.FacetGrid(df3, col=column, height=3.5, margin_titles=True, 
                        gridspec_kws={'wspace': 0.03}, subplot_kws={'ylim':[0,ylim]})
    g.map(sns.barplot, label, 'fraction (%)', palette=cpal, order=np.sort(df3[label].unique()))

    return df3, g



def triple_cat_stats(logs, label, row, column, cpal, ylim=100):

    """

    Display colored histograms for a categorical variable given other two as a filters per figure column and row

    Parameters:
        logs (pd.DataFrame): dataframe containing well logs data, filtered to ONE WELL only
        label (string): variable (logs column name) to create the histogram
        row (strin): name of the filtering row
        column (strin): name of the filtering column
        cpal (list:string): list of matplotlib color names for the categorical classes
        ylim (float): max value for the vertical axis

    Returns:
        df3 (pd.DataFrame): result data frame with computed histogram frequencies
        g (sns.axes): Seaborn axes figure

    """   
        
    df = logs[[label, column, row]].dropna(how='any', axis=0)

    df_list = []
    sorted_cols = np.sort(list(logs[column].unique()))
    for elem in sorted_cols:
        df1 = double_cat_counts(df[df[column]==elem], label, row)
        df1[column] = elem
        df1.rename(mapper={column:column, 'variable':label, 'value':'fraction (%)'}, inplace=True, axis=1)
        df_list.append(df1)
    df3 = pd.concat(df_list)
    
    g = sns.FacetGrid(df3, row = row, col=column, height=3.5, margin_titles=True, 
                        gridspec_kws={'wspace': 0.03}, subplot_kws={'ylim':[0,ylim]})
    g.map(sns.barplot, label, 'fraction (%)', palette=cpal, order=np.sort(df3[label].unique()))

    return df3, g