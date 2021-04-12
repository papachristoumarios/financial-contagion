import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
from eba_dataloader import expon_fit_helper
from german_banks_dataloader import pareto_fit_helper

sns.set_theme()

def load_poverty_tracker_data():

    data = pd.read_csv('data/poverty_tracker_full.csv')

    num_children = data['childx']
    head_income = data['imp_earnhd_tc']
    spouse_income = data['imp_earnsp_tc']

    other_income = data['imp_incdis_tc'] + data['imp_incsnap_tc'] + data['imp_incui_tc'] + data['imp_incret_tc'] + data['imp_increg_tc'] + data['imp_incoth_tc']

    is_minority = data['imp_race'] != '1. White Non-Hispanic'

    total_income = head_income + spouse_income + other_income

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)

    sns.distplot(total_income, kde=False, fit=scipy.stats.expon, label=expon_fit_helper(total_income), ax=axs[0], color='olive')

    sns.distplot(total_income[is_minority == False], fit=scipy.stats.expon, kde=False, label='{}'.format(expon_fit_helper(total_income[is_minority == False])), ax=axs[1], color='skyblue')
    sns.distplot(total_income[is_minority == True], fit=scipy.stats.expon, kde=False, label='{}'.format(expon_fit_helper(total_income[is_minority == True])), ax=axs[2], color='teal')
    axs[0].set_xlim(0, total_income.max())
    axs[1].set_xlim(0, total_income.max())
    axs[2].set_xlim(0, total_income.max())

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].title.set_text('Overall')
    axs[1].title.set_text('Non-minority')
    axs[2].title.set_text('Minority')
    plt.suptitle('Poverty Tracker Dataset Income Distribution')

    axs[0].get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(round(int(x) / 1000, 2))))
    axs[1].get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(round(int(x) / 1000, 2))))
    axs[2].get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(round(int(x) / 1000, 2))))

    plt.xlabel('Total Income')
    plt.ylabel('Frequency')
    plt.savefig('poverty_tracker_income.png')

    plt.figure(figsize=(10, 10))
    sns.distplot(num_children, kde=False, fit=scipy.stats.pareto, label=pareto_fit_helper(num_children))
    plt.xlim(0, num_children.max())
    plt.legend()
    plt.title('Poverty Tracker Number of Children Distribution')
    plt.xlabel('Number of Children')
    plt.ylabel('Frequency')
    plt.savefig('poverty_tracker_num_children.png')


    plt.show()



if __name__ == '__main__':
    load_poverty_tracker_data()
