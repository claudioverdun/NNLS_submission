import numpy as np
import numpy.linalg as la

import solver_interfaces

# Parameters
#
x_dim = 500
y_dim = 200
s = 25
print(f'Dimension = ({x_dim},{y_dim}), sparsity = {s}')

res = solver_interfaces.basic_test(x_dim, y_dim, s, seed=3, nreps=3, verbose=2)

#
# Comparison
#

target_l1 = np.mean([la.norm(vi[0], 1) for vi in res['Targets']])
for k, v in res.items():
    if isinstance(v, list):
        mean_l1 = np.mean([la.norm(vi[0], 1) for vi in v])
        mean_err = np.mean([vi[1] for vi in v])
        mean_dt = np.mean([vi[2] for vi in v])
        print(f'{str(k):>30}: {mean_l1/target_l1:6.3f} {mean_err:.3e}, {mean_dt:.5f}')
