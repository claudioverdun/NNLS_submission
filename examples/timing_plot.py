# Figures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')

# Need 8 styles:
symbols = ['o', 's', 'D', '^', 'v', '>', '<', 'p']

# Load results
import pickle
# res_small = np.load('results_small.npz', allow_pickle=True)
large_res = pickle.load(open('results/timings_large.pkl', 'rb'))

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
for k, d in large_res.items():
    if k == 'TNT':
        continue
    if isinstance(d, dict):
        mark = symbols.pop(0)
        dims = []
        errs = []
        std_errs = []
        dts = []
        std_dts = []
        for dim, v in d.items():
            if len(v) == 0:
                continue
            mean_err = np.mean([vi[1] for vi in v])
            std_err = np.std([vi[1] for vi in v])
            mean_dt = np.mean([vi[2] for vi in v])
            std_dt = np.std([vi[2] for vi in v])
            dims.append(dim)
            errs.append(mean_err)
            dts.append(mean_dt)
            std_errs.append(std_err)
            std_dts.append(std_dt)
        ax1.errorbar(dims, dts, std_dts, label=k, marker=mark)
        ax2.errorbar(dims, errs, std_errs, label=k, marker=mark)

ax1.set_title('Solve time')
ax2.set_title('Error')

for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlabel('Dimension')
    ax.legend()
    ax.grid(True)

fig1.savefig('figures/timing_plot_large_notnt.pdf')
fig2.savefig('figures/err_plot_large_notnt.pdf')
plt.show()
