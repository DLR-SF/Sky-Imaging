import os
import pickle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def plot_density(x, y, pairs=False, ax=None, xlabel=None, ylabel=None, title='', xlim=(None, None), ylim=(None, None),
                 cbar_scale='linear', cbar_lim=(None, None), metrics=None, quantiles=False, quan_bin_num=10, save=False,
                 cut_bins=True, print_perc_legend=True, print_metrics_box=True):
    """
    Create density plot with defined grid size allowing to interpret color code quantitatively.

    :param x: (list, DataFrame column) x-axis values 
    :param y: (list, DataFrame column) y-axis values 
    :param pairs: (bool) If True, parity plot will be produced (equal aspect and angle bisecting)
    :param ax: (matplotlib axis) Here a subplots axis can be passed. If None, a single plot will be returned
    :param xlabel: (str) Custom xlabel
    :param ylabel: (str) Custom ylabel
    :param title: (str) Desired plot title
    :param xlim: (tuple) x-axis limits
    :param ylim: (tuple) y-axis limits
    :param cbar_scale: (string) use 'linear' or 'log' color coding and colorbar
    :param cbar_lim: (tuple) Limits of colorbar
    :param metrics: (str) Choose between rel for relative or abs for absolute or both
    :param quantiles: (bool) If True, 5%, 25%, 50%, 75%, and 95% quantiles will be plotted
    :param quan_bin_num: (int) Number of bins for quantiles
    :param save: (bool or str) If Truthy, plot is saved as pickle. If truthy str, file is named to the value of save
    :param cut_bins: If False, bin edges will not cut the x values
    :param print_perc_legend: If True, print a legend describing plotted percentiles
    :param print_metrics_box: If True, print deviation metrics in plot
    """

    # calculate metrics
    if metrics is not None:
        valid = np.isfinite(x) & np.isfinite(y)
        rmse = np.round(mean_squared_error(x[valid], y[valid], squared=False), 1)
        r_rmse = np.round(rmse / x[valid].mean() * 100, 1)

        bias = np.round((y[valid] - x[valid]).mean(), 1)
        r_bias = np.round(bias / x[valid].mean() * 100, 1)

        r2 = np.round(r2_score(x[valid], y[valid]), 3)
        corr_coef = np.round(pearsonr(x[valid], y[valid])[0], 3)

        mae = np.round(abs((y[valid] - x[valid])).mean(), 1)
        r_mae = np.round(mae / x[valid].mean() * 100, 1)

    def range_replace_none(v, vec):
        v0, v1 = v
        if v0 is None:
            v0 = np.nanmin(vec)
        if v1 is None:
            v1 = np.nanmax(vec)
        return v0, v1

    xlim = range_replace_none(xlim, x)
    ylim = range_replace_none(ylim, y)

    vmin = cbar_lim[0]
    vmax = cbar_lim[1]

    if vmin is None or vmax is None:
        h, _, _ = np.histogram2d(x, y, bins=[100, 100], range=[xlim, ylim])
        (vmin, vmax) = range_replace_none((vmin, vmax), h[h > 0])
        vmin = np.nanmax([vmin, 1e-30])

    # create plot
    if ax is None:
        if pairs:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.gcf()

    ax.minorticks_on()
    ax.grid(which='minor', linestyle='--', alpha=0.3, zorder=0)
    ax.grid(which='major', linestyle='-', zorder=1)

    if cbar_scale == 'linear':
        _, _, _, im = ax.hist2d(x, y, vmin=vmin, vmax=vmax, bins=(100, 100), range=[xlim, ylim], cmin=1e-29, zorder=3,
                                label='_nolegend_')
    elif cbar_scale == 'log':
        _, _, _, im = ax.hist2d(x, y, norm=matplotlib.colors.LogNorm(vmin, vmax), bins=(100, 100), range=[xlim, ylim],
                                cmin=1e-29, zorder=3, label='_nolegend_')

    # set axes
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    if pairs:
        ax.set_xlim((0, 1.1 * x.max()))
        ax.set_ylim((0, 1.1 * x.max()))
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, fontweight='bold')

    if quantiles:
        # create DataFrame from the two series x and y
        df = pd.concat([x, y], axis=1)

        try:
            bins = np.arange(x.min(), x.max() + (x.max() - x.min()) / quan_bin_num, (x.max() - x.min()) / quan_bin_num)
        except:
            bins = np.arange(0, len(set(list(x))) + len(set(list(x))) / quan_bin_num, len(set(list(x))) / quan_bin_num)

        # do not cut bins if variable is categorical e.g. DNI variability classes
        if not cut_bins:
            df[f'{x.name}_bins'] = df[x.name]
            centers = sorted(df[x.name].unique())
        else:
            # cut bins
            df[f'{x.name}_bins'] = pd.cut(df.iloc[:, 0], bins)
            centers = []

            for b in bins[:-1]:
                centers.append(b + (x.max() - x.min()) / quan_bin_num / 2)

        # calculate quantiles for bins
        def calculate_quantiles(df, x, y, centers, quan):
            quantile = []
            for i in df.groupby(f'{x.name}_bins')[y.name].quantile(quan):
                quantile.append(i)
            return quantile

        quantile2_5 = calculate_quantiles(df, x, y, centers, quan=0.025)
        quantile15_85 = calculate_quantiles(df, x, y, centers, quan=0.1585)
        quantile50 = calculate_quantiles(df, x, y, centers, quan=0.5)
        quantile84_15 = calculate_quantiles(df, x, y, centers, quan=0.8415)
        quantile97_5 = calculate_quantiles(df, x, y, centers, quan=0.975)

        ax.plot(centers, quantile2_5, linewidth=1.7, c='black', ls='--', zorder=4)
        ax.plot(centers, quantile15_85, linewidth=1.7, c='black', ls=':', zorder=4)
        ax.plot(centers, quantile50, linewidth=1.7, c='black', ls='-', zorder=4)
        ax.plot(centers, quantile84_15, linewidth=1.7, c='black', ls=':', zorder=4)
        ax.plot(centers, quantile97_5, linewidth=1.7, c='black', ls='--', zorder=4)

        if print_perc_legend:
            if metrics is None or not pairs:
                ax.legend(['97.5th Percentile', '84.15th Percentile', '50th Percentile', '15.85th Percentile',
                           '2.5th Percentile'], loc='upper center', bbox_to_anchor=(0.6, 1), ncol=3, fancybox=False,
                          shadow=True, prop={'size': 10})
            elif metrics is not None:
                ax.legend(['97.5th Percentile', '84.15th Percentile', '50th Percentile', '15.85th Percentile',
                           '2.5th Percentile'], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fancybox=False,
                          shadow=True, prop={'size': 10})
            else:
                ax.legend(['97.5th Percentile', '84.15th Percentile', '50th Percentile', '15.85th Percentile',
                           '2.5th Percentile'], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fancybox=False,
                          shadow=True, prop={'size': 10})

    if print_metrics_box:
        if metrics == 'rel':
            anchored_text = AnchoredText(f'rRMSE = {r_rmse}%\nrMAE = {r_mae}%\nrBIAS = {r_bias}%\nR$^{2}$ = {r2}',
                                         loc='upper left', frameon=True,
                                         prop=dict(fontsize=plt.rcParams['legend.fontsize']))
            ax.add_artist(anchored_text)
        elif metrics == 'abs':
            anchored_text = AnchoredText(
                f'RMSE = {rmse}W/m$^{2}$\nMAE = {mae}W/m$^{2}$\nBIAS = {bias}W/m$^{2}$\nR$^{2}$ = {r2}',
                loc='upper left', frameon=True, prop=dict(fontsize=plt.rcParams['legend.fontsize']))
            ax.add_artist(anchored_text)
        elif metrics == 'both':
            anchored_text = AnchoredText(f'RMSE = {rmse}W/m$^{2}$ | rRMSE = {r_rmse}%\nMAE = {mae}W/m$^{2}$ | rMAE ='
                                         f' {r_mae}%\nBIAS = {bias}W/m$^{2}$ | rBIAS = {r_bias}%\nR$^{2}$ = {r2}',
                loc='upper left', frameon=True,
                prop=dict(fontsize=plt.rcParams['legend.fontsize']))
            ax.add_artist(anchored_text)
        if metrics == 'corr':
            anchored_text = AnchoredText(f"Pearson's r = {corr_coef}\nR$^{2}$ = {r2}", loc='upper left',
                                         frameon=True, prop=dict(fontsize=12))
            ax.add_artist(anchored_text)

    cbar = plt.colorbar(im, ax=ax)  #, aspect=40, pad=0.25)
    cbar.set_label('Counts [-]', labelpad=15, fontweight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)

    if save:
        # save figure to file
        if type(save) is str:
            save_name = save
        else:
            save_name = 'density_scatter'
        save_path = os.path.join('plots', save_name)
        pickle.dump(fig, open(save_path + '.pickle', 'wb'))
        fig.savefig(save_path + '.png')
