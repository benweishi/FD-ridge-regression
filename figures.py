import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from cycler import cycler
import math


linestyles = {'RFDRR':'1-r', 'FDRR':'2-c', 'iSVDRR':'3-m', '2LFDRR':'4-y', 'RPRR':'x-b', 'CSRR':'+-g', 'RR':'+--k'}
linestyles_nomark = {'RFDRR':'-r', 'FDRR':'-c', 'iSVDRR':'-m', '2LFDRR':'-y', 'RPRR':'-b', 'CSRR':'-g', 'RR':'-k'}
legend_order = ['RR', 'RPRR', 'CSRR', 'iSVDRR', '2LFDRR', 'FDRR', 'RFDRR']
legend_text = {'RFDRR':r'$\textsc{RFDrr}$', 'FDRR':r'\textsc{FDrr}', 'iSVDRR':r'\textsc{iSVDrr}', '2LFDRR':r'\textsc{2LFDrr}', 'RPRR':r'\textsc{RPrr}', 'CSRR':r'\textsc{CSrr}', 'RR':r'\textsc{rr}'}
plt.rcParams['text.usetex'] = True
#print(plt.rcParams.keys())
params = {'font.size': 7,
          #'legend.fontsize': 'small',
          #'axes.titlesize': 'medium',
          'lines.linewidth': 1,
          'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']) + cycler('marker', ['1', '2', '3', '4', '+'])}
plt.rcParams.update(params)
# https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html

def best_gamma(path):
    """Draw gammas.pdf.

    Args:
        path (str): Path of output dir, with dataset dirs as sub dirs.

    Outputs:
        file: path/gammas.pdf.

    """
    print("singular values.")
    fig, ax = plt.subplots()
    data_dirs = glob(os.path.join(path, '*/'))
    for data_dir in data_dirs:
        data_name = os.path.basename(os.path.normpath(data_dir))
        gammas = np.loadtxt(os.path.join(data_dir, 'ridge', 'gammas.txt'))
        y_pred_rr = np.load(os.path.join(data_dir, 'ridge', 'preds.npy'))  # shape: ell*g*r*n_test
        y_true = np.load(os.path.join(data_dir, 'data_only', 'preds.npy'))
        y_true = y_true[0]  # shape r*n_t
        y_dif = y_pred_rr - y_true  # shape ell*g*r*n_test
        y_errors = np.mean(y_dif**2, axis=(0, 2, 3))  # shape g
        print(f'The best gamma value for dataset {data_name} is {gammas[np.argmin(y_errors)]}.')
        ax.plot(gammas, y_errors, label=data_name)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('mean square pred. error')
    plt.legend()
    fig.set_size_inches(2.3, 2)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'gammas.pdf'))
    plt.show()


def singulars(path):
    """Draw singulars.pdf.

    Args:
        path (str): Path of output dir, with dataset dirs as sub dirs.

    Outputs:
        file: path/singulars.pdf.

    """
    print("singular values.")
    fig, ax = plt.subplots()
    data_dirs = glob(os.path.join(path, '*/'))
    for data_dir in data_dirs:
        data_name = os.path.basename(os.path.normpath(data_dir))
        singulas = np.loadtxt(os.path.join(data_dir, 'data_singular.txt'))
        singulas /= singulas[0]
        ax.plot(np.arange(0, 1, 1/len(singulas))+1/len(singulas), singulas, '-', label=data_name)
    ax.set_xlabel('Normalized singular indices')
    ax.set_ylabel('Normalized singular values')
    plt.legend()
    fig.set_size_inches(2, 1.5)
    plt.tight_layout(pad=0.0)
    plt.savefig(os.path.join(path, 'singulars.pdf'))
    plt.show()


def errors(dir, ymaxes=[0,0,0]):
    print("Errors figures.")
    data_dirs = glob(os.path.join(dir, '*/'))
    n_dataset = len(data_dirs)
    # create axes with shape n_dataset*(2+2)
    fig = plt.figure(figsize=(6, 1.2*n_dataset))
    outer = gridspec.GridSpec(1, 2)
    axes = []
    for i in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(n_dataset, 2,
                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2*n_dataset):
            ax = plt.Subplot(fig, inner[j])
            if j % 2 > 0:
                if j > 1:
                    fig.add_subplot(ax, sharey=axes[-1], sharex=axes[-2])
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    fig.add_subplot(ax, sharey=axes[-1])
                    plt.setp(ax.get_yticklabels(), visible=False)
            else:
                if j > 1:
                    fig.add_subplot(ax, sharex=axes[-2])
                else:
                    fig.add_subplot(ax)
            if j < 2*(n_dataset-1):
                plt.setp(ax.get_xticklabels(), visible=False)
            axes.append(ax)
    axes = np.array(axes)
    axes = axes.reshape((-1, 2*n_dataset))
    # hardcoded index, should be changed with number of datasets.
    axes = np.concatenate((axes[0:2, 0:2], axes[0:2, 2:4], axes[0:2, 4:6]), axis=0)
    axes = axes.reshape((-1, 4))
    # start plotting
    for ax_row, data_dir in zip(axes, data_dirs):
        for ax in ax_row:
            ax.set_xscale('log', basex=2)
        (ax_coef, ax_coef_time, ax_pred, ax_pred_time) = ax_row
        data_name = os.path.basename(os.path.normpath(data_dir))
        d, n, _ = np.loadtxt(os.path.join(data_dir, 'data_summary.txt'), dtype='i')
        alg_dirs = glob(os.path.join(data_dir, '*RR/'))
        alg_dirs = [alg_dirs[0], alg_dirs[5], alg_dirs[2], alg_dirs[1], alg_dirs[3], alg_dirs[4], alg_dirs[6]]
        for alg_dir in alg_dirs[:-1]:
            alg_name = os.path.basename(os.path.normpath(alg_dir))
            ells = np.loadtxt(os.path.join(alg_dir, 'ells.txt'))
            coefs_errors = np.load(os.path.join(alg_dir, 'coef_error.npy'))  # shape: ell
            ax_coef.plot(ells, coefs_errors, linestyles[alg_name], label=alg_name)
            pred_errors = np.loadtxt(os.path.join(alg_dir, 'pred_error.txt'))  # shape: ell
            ax_pred.plot(ells, pred_errors, linestyles[alg_name], label=alg_name)
            train_times = np.loadtxt(os.path.join(alg_dir, 'train_time.txt'))
            query_times = np.loadtxt(os.path.join(alg_dir, 'query_time.txt'))  # shape: ells
            running_time = train_times + query_times
            real_times = train_times + query_times / ells * n
            ax_coef_time.plot(real_times, coefs_errors, linestyles[alg_name], label=alg_name)
            ax_pred_time.plot(real_times, pred_errors, linestyles[alg_name], label=alg_name)
        # Ridge
        alg_dir = alg_dirs[-1]
        alg_name = os.path.basename(os.path.normpath(alg_dir))
        ells = np.loadtxt(os.path.join(alg_dir, 'ells.txt'))
        ax_coef.plot(ells[1], 0, '+k', label=alg_name)
        pred_errors = np.loadtxt(os.path.join(alg_dir, 'pred_error.txt'))  # shape: ell
        ax_pred.plot(ells[1], pred_errors[1], '+k', label=alg_name)
        train_times = np.loadtxt(os.path.join(alg_dir, 'train_time.txt'))
        query_times = np.loadtxt(os.path.join(alg_dir, 'query_time.txt'))  # shape: ells
        real_times = train_times + query_times / ells[1] * n  # shape: ells
        running_time = train_times + query_times
        ax_coef_time.plot(real_times[0], 0, '+k', label=alg_name)
        ax_pred_time.plot(real_times[0], pred_errors[1], '+k', label=alg_name)
        ax_coef.set_ylabel(f'{data_name} coef. error')
        ax_pred.set_ylabel(f'{data_name} pred. error')
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axes[-1, 0].set_xlabel(r'$\ell$')
    axes[-1, 1].set_xlabel('time(s)')
    axes[-1, 1].xaxis.set_major_formatter(formatter)
    #axes[-1, 1].set_xlim([4, 64])
    axes[-1, 2].set_xlabel(r'$\ell$')
    axes[-1, 3].set_xlabel('time(s)')
    axes[-1, 3].xaxis.set_major_formatter(formatter)
    #axes[-1, 3].set_xlim([4, 64])
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend([handles[i] for i in [labels.index(x) for x in legend_order]], [
        legend_text[l] for l in legend_order], loc='center right', labelspacing=3.0, frameon=False, bbox_to_anchor=(1.06, 0.56))
    fig.set_size_inches(5.5, 1.2*len(data_dirs))
    for txt in legend.get_texts():
        txt.set_x(-20)  # x-position
        txt.set_y(-10)  # y-position
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(os.path.join(dir, 'errors.pdf'))
    plt.show()


def running_time(dir):
    print("Running time figure.")
    fig, axes = plt.subplots(3, 4, sharey=True)
    # row 1, t(l)
    path = os.path.join(dir, 'algs_time_error/HR/')
    n = np.loadtxt(os.path.join(path, 'data_summary.txt'), dtype='i')[1]
    alg_dirs = glob(os.path.join(path, '*RR/'))
    axes_row = axes[0]
    for alg_dir in alg_dirs:
        alg_name = os.path.basename(os.path.normpath(alg_dir))
        train_times = np.loadtxt(os.path.join(
            alg_dir, 'train_time.txt'))  # shape: ell
        ells = np.loadtxt(os.path.join(alg_dir, 'ells.txt'))
        query_times = np.loadtxt(os.path.join(
            alg_dir, 'query_time.txt'))  # shape: ell
        running_times = train_times + query_times
        real_times = train_times + query_times / ells * n
        if alg_name == 'RR':
            axes_row[0].plot(ells[1], train_times[0], '+--k', label=alg_name)
            axes_row[1].plot(ells[1], query_times[0], '+--k', label=alg_name)
            axes_row[2].plot(ells[1], running_times[0], '+--k', label=alg_name)
            axes_row[3].plot(ells[1], real_times[0], '+--k', label=alg_name)
        else:
            axes_row[0].plot(ells, train_times,
                             linestyles[alg_name], label=alg_name)
            axes_row[1].plot(ells, query_times,
                             linestyles[alg_name], label=alg_name)
            axes_row[2].plot(ells, running_times,
                             linestyles[alg_name], label=alg_name)
            axes_row[3].plot(ells, real_times,
                             linestyles[alg_name], label=alg_name)
    # row 2, t(d)
    path = os.path.join(dir, 'dVtime/')
    axes_row = axes[1]
    n = 2**8
    ell = 2**6
    ds = np.loadtxt(os.path.join(path, 'ds.txt'), dtype='i')
    alg_files = glob(os.path.join(path, '*RR.txt'))
    for alg_file in alg_files:
        alg_name = os.path.basename(os.path.normpath(alg_file))[:-4]
        # shape: 2*len(ds), train and query
        times = np.clip(np.loadtxt(alg_file), a_min=0.00001, a_max=None)
        axes_row[0].plot(ds, times[0], linestyles[alg_name], label=alg_name)
        axes_row[1].plot(ds, times[1], linestyles[alg_name], label=alg_name)
        axes_row[2].plot(ds, times.sum(axis=0),
                     linestyles[alg_name], label=alg_name)
        axes_row[3].plot(ds, times[1] / ell * n + times[0],
                         linestyles[alg_name], label=alg_name)
    # row 3, t(n)
    path = os.path.join(dir, 'nVtime/')
    axes_row = axes[2]
    n = 2**15
    ell = 2**6
    n_list = np.arange(0, n+1, ell)
    alg_files = glob(os.path.join(path, '*RR.txt'))
    idx = np.array([2**p for p in range(int(math.log(n//ell, 2)+1))])
    for alg_file in alg_files:
        alg_name = os.path.basename(os.path.normpath(alg_file))[:-4]
        times = np.loadtxt(alg_file)  # train and query
        axes_row[0].plot(n_list[idx], times[0, idx],
                     linestyles[alg_name], label=alg_name)
        axes_row[1].plot(n_list[idx], np.diff(times[1])[idx-1],
                     linestyles[alg_name], label=alg_name)
        axes_row[2].plot(n_list[idx], times[0, idx] + np.diff(times[1])
                     [idx-1], linestyles[alg_name], label=alg_name)
        axes_row[3].plot(n_list[idx], times.sum(axis=0)[idx],
                     linestyles[alg_name], label=alg_name)
    # format
    for r in range(len(axes)):
        axes[r][0].set_ylabel(r'time (s)')
        for ax in axes[r]:
            ax.set_xscale('log', basex=2)
            ax.set_yscale('log', basey=2)
    axes[0][0].set_title("Training")
    axes[0][1].set_title("Query")
    axes[0][2].set_title("Training + query")
    axes[0][3].set_title(r"Training+query*$(n/\ell)$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend([handles[i] for i in [labels.index(x) for x in legend_order]], [
        legend_text[l] for l in legend_order], loc='center right', labelspacing=3.0, frameon=False, bbox_to_anchor=(1.06, 0.5))
    for txt in legend.get_texts():
        txt.set_x(-20)  # x-position
        txt.set_y(-10)  # y-position
    fig.set_size_inches(5.5, 3.6)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    fig.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(dir, 'running_time.pdf'))
    plt.show()
    

def figures(out_path):
    #print(plt.rcParams.keys())
    params = {'font.size': 7,
              'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']) + cycler('marker', ['1', '2', '3', '4', '+'])}
    plt.rcParams.update(params)
    # https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
    singulars(os.path.join(out_path, 'algs_time_error/'))
    running_time(out_path)
    errors(os.path.join(out_path, 'algs_time_error/'))

if __name__ == '__main__':
    figures('./output/')
