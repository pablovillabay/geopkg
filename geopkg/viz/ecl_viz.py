import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.style.use('seaborn')

def date_plot(observed, vectors, cases=[], wells=[], lims={}):

    fig, ax = plt.subplots(len(vectors), 1, figsize=(20, 8*len(vectors)), sharex=True, )

    subax = []
    for i, plot in enumerate(vectors):
        for j, vector in enumerate(plot):
            if j == 0:
                subax.append(ax[i])
            else:
                subax.append(ax[i].twinx())
            if len(cases) > 0:
                for case in cases:
                    for well in wells:
                        if well in case.index:
                            subax[-1].plot(case.loc[well][vector], linewidth=0.85)
            for well in wells:
                plot_ = observed.loc[well]
                if well in observed.index:
                    if vector in plot_.columns:
                        plot_ = plot_[plot_[vector] > 1]
                        subax[-1].scatter(plot_.index, plot_[vector], label=well, s=8)
                        if vector in list(lims.keys()):
                            subax[-1].set_ylim(lims[vector][0], lims[vector][1])
                        else:
                            subax[-1].set_ylim(0.00)
                        subax[-1].set_ylabel(vector)

    subax[-1].set_xlabel('Date')
    subax[-1].legend(loc='best')
    fig.suptitle('Production data - date plot', y=0.90, fontsize=16)            
    fig.tight_layout(rect=[0.025, 0.975, 1.025, 0.975])
    return fig, ax
