import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

colors = {
    'DISTILL': '#EDB732',  # 黄
    'QRM-RS': '#C565C7',  # 紫
    'QRM': '#1F77B4',  # 蓝
    'EQUIV': '#008800',  # 绿
    'LSRM': '#FF6347',  # 红

    'LSRM-nsr': '#1F77B4',  # 蓝
    'LSRM-nstd': '#C565C7',  # 紫
    'LSRM-and-max': '#008800',  # 绿
    'LSRM-or-max': '#EDB732',   # 黄
    'LSRM-then-right': '#C10066'    # 品红
}


def read_csv_file(filename):
    data = dict()
    data_list = transpose_csv(filename)
    num_legends = (len(data_list) - 1) // 3

    for i in range(num_legends):
        legend_name = data_list[3 * i + 1][0]
        avg = list(map(float, data_list[3 * i + 1][1:]))
        low = list(map(float, data_list[3 * i + 2][1:]))
        high = list(map(float, data_list[3 * i + 3][1:]))
        data[legend_name] = [avg, low, high]
    return data


def transpose_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        csv_data = list(reader)
    transposed_csv_data = list(map(list, zip(*csv_data)))

    return transposed_csv_data


def smooth_data(data, weight=0.8):
    s_array = [0 for _ in range(len(data))]
    s_array[0] = data[0]
    for i in range(1, len(data)):
        s_array[i] = weight * s_array[i - 1] + (1 - weight) * data[i]
    return s_array


def plot_curves(filename, **kwargs):
    base_path = os.path.join(os.path.dirname(__file__), '..')
    filename = os.path.join(base_path, 'data', kwargs['env_name'], filename)
    data = read_csv_file(filename)
    plt.figure(figsize=kwargs['figsize'])
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(kwargs['title'], fontsize=20)

    plt.grid(color='#FFFFFF', linewidth=2)
    num_steps = 0
    y_max = 0
    for curve_name in kwargs['curve_names']:
        avg, low, high = data[curve_name]
        num_steps = len(avg)
        avg = smooth_data(avg, kwargs['smooth_weight'])
        low = smooth_data(low, kwargs['smooth_weight'])
        high = smooth_data(high, kwargs['smooth_weight'])
        y_max = max(y_max, max(high))
        num_units = len(avg)
        x = [i for i in range(num_units)]
        if kwargs.get('mark_LSRM', True):
            display_name = "LSRM（本文）" if curve_name=="LSRM" else curve_name
        else:
            display_name = curve_name
        # display_name = "LSRM（本文）" if curve_name == "LSRM" else curve_name
        plt.plot(x, avg, linewidth=2, label=display_name, c=colors[curve_name])
        plt.fill_between(x, low, high, color=colors[curve_name], alpha=0.25)

    x_tick_interval = num_steps // 5
    y_tick_interval = kwargs['y_tick_interval']
    num_y_ticks = int(y_max / y_tick_interval) + 1
    plt.xlim(-2, num_steps - 1)
    plt.ylim(-0.2 * y_tick_interval, y_max + 0.2 * y_tick_interval)
    fontsize = 15
    x_ticks = [0, ] + [i * x_tick_interval - 1 for i in range(1, 6)]
    if kwargs['env_name'] == 'office':
        x_tick_labels = ['0', ] + [str(i * x_tick_interval // 2) + 'k' for i in range(1, 6)]
    else:
        x_tick_labels = ['0', ] + [str(i * x_tick_interval) + 'k' for i in range(1, 6)]
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=fontsize)
    plt.yticks(ticks=[y_tick_interval * i for i in range(num_y_ticks)], fontsize=fontsize)
    plt.xlabel('训练步数', fontsize=fontsize)
    plt.ylabel('单步平均奖励', fontsize=fontsize)
    # plt.locator_params(axis='x', nbins=5)
    plt.gca().set_facecolor('#EBF0F2')
    if kwargs['show_legend']:
        plt.legend(loc=kwargs.get('legend_loc', 'lower right'), fontsize=int(0.8 * fontsize))

    for spine in plt.gca().spines.values():
        spine.set_color('none')
    if kwargs['savefig']:
        plt.savefig(kwargs['savefig'], bbox_inches='tight')
    plt.show()


def plot_office():
    env_name = 'office'
    figsize = (5, 4)
    plot_curves(filename='phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/office_p0.pdf'
                )
    plot_curves(filename='phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/office_p1.pdf'
                )
    plot_curves(filename='phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=True,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/office_p2.pdf'
                )


def plot_craft():
    env_name = 'craft'
    figsize = (5, 4)
    plot_curves(filename='phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/craft_p0.pdf'
                )
    plot_curves(filename='phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/craft_p1.pdf'
                )
    plot_curves(filename='phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/craft_p2.pdf'
                )
    plot_curves(filename='phase3.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=True,
                legend_loc="upper left",
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/craft_p3.pdf'
                )


def plot_water():
    env_name = 'water'
    figsize = (5, 4)
    plot_curves(filename='phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/water_p0.pdf'
                )
    plot_curves(filename='phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/water_p1.pdf'
                )
    plot_curves(filename='phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/water_p2.pdf'
                )
    plot_curves(filename='phase3.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=True,
                curve_names=["QRM", "QRM-RS", "EQUIV", "DISTILL", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/water_p3.pdf'
                )

def plot_ab_office():
    env_name = 'office'
    figsize = (5, 4)
    plot_curves(filename='ab_phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_office_p0.pdf'
                )
    plot_curves(filename='ab_phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_office_p1.pdf'
                )
    plot_curves(filename='ab_phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.5,
                show_legend=True,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_office_p2.pdf'
                )

def plot_ab_water():
    env_name = 'water'
    figsize = (5, 4)
    plot_curves(filename='ab_phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_water_p0.pdf'
                )
    plot_curves(filename='ab_phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_water_p1.pdf'
                )
    plot_curves(filename='ab_phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_water_p2.pdf'
                )
    plot_curves(filename='ab_phase3.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=True,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.01,
                title='',
                savefig='../figures_cn/ab_water_p3.pdf'
                )

def plot_ab_craft():
    env_name = 'craft'
    figsize = (5, 4)
    plot_curves(filename='ab_phase0.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/ab_craft_p0.pdf'
                )
    plot_curves(filename='ab_phase1.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.95,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/ab_craft_p1.pdf'
                )
    plot_curves(filename='ab_phase2.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=False,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/ab_craft_p2.pdf'
                )
    plot_curves(filename='ab_phase3.csv',
                env_name=env_name,
                figsize=figsize,
                smooth_weight=0.99,
                show_legend=True,
                mark_LSRM=False,
                curve_names=["LSRM-nsr", "LSRM-nstd", "LSRM-and-max", "LSRM-or-max", "LSRM-then-right", "LSRM"],
                legend_loc="upper left",
                y_tick_interval=0.005,
                title='',
                savefig='../figures_cn/ab_craft_p3.pdf'
                )


if __name__ == "__main__":
    # plot_office()
    # plot_craft()
    # plot_water()
    # plot_ab_office()
    # plot_ab_craft()
    plot_ab_water()