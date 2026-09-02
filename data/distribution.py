import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def parse_data(filename):
    with open(filename, 'r') as f:
        data = f.read()

    sections = data.split('\n\n')
    parsed_data = {}

    for section in sections:
        lines = section.split('\n')
        if len(lines) > 1:
            key = lines[0].strip(':')
            values = [line.split(':') for line in lines[1:] if ':' in line]
            parsed_data[key] = np.array([(float(v[0]), int(v[1])) for v in values])

    return parsed_data


data = parse_data(r'dataset')


titles = {
    "Enthalpy of Formation Distribution": "ΔHf_298K (kcal/mol)",
    "Entropy Distribution": "S298K (cal/mol/K)",
    "Heat Capacity Distribution at T1": "ΔCp_300K (cal/mol/K)",
    "Heat Capacity Distribution at T2": "ΔCp_400K (cal/mol/K)",
    "Heat Capacity Distribution at T3": "ΔCp_500K (cal/mol/K)",
    "Heat Capacity Distribution at T4": "ΔCp_600K (cal/mol/K)",
    "Heat Capacity Distribution at T5": "ΔCp_800K (cal/mol/K)",
    "Heat Capacity Distribution at T6": "ΔCp_1000K (cal/mol/K)",
    "Heat Capacity Distribution at T7": "ΔCp_1500K (cal/mol/K)"
}


fig, axs = plt.subplots(3, 3, figsize=(15, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, (key, ax) in enumerate(zip(data.keys(), axs.ravel())):
    values = data[key][:, 0]
    counts = data[key][:, 1]


    ax.bar(values, counts, width=np.diff(values).mean(), alpha=0.5, label='Data', color='orange')


    kde = gaussian_kde(np.repeat(values, counts.astype(int)))
    x = np.linspace(values.min(), values.max(), 1000)
    ax.plot(x, kde(x), color='blue', label='KDE')


    ax.set_title(titles[key])
    ax.set_xlabel(titles[key])
    ax.set_ylabel('Number of Species')
    ax.legend()

plt.suptitle('Thermochemical Properties of the Collected Dataset', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('distribution_plots.png', dpi=300)