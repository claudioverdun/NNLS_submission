import numpy as np
import numpy.linalg as la

import solver_interfaces

res_small = solver_interfaces.square_dim_scaling([50,100,200], 2)
for k, d in res_small.items():
    if isinstance(d, dict):
        for dim, v in d.items():
            if len(v) == 0:
                continue
            mean_l1 = np.mean([la.norm(vi[0], 1) for vi in v])
            mean_err = np.mean([vi[1] for vi in v])
            mean_dt = np.mean([vi[2] for vi in v])
            print(f'{str(k):>30} {dim:4}: {mean_err:.3e}, {mean_dt:.5f}')

do_large = True
if do_large:
    res_large = solver_interfaces.square_dim_scaling([500,1000,2000, 4000], 2)
    for k, d in res_large.items():
        if isinstance(d, dict):
            for dim, v in d.items():
                if len(v) == 0:
                    continue
                mean_l1 = np.mean([la.norm(vi[0], 1) for vi in v])
                mean_err = np.mean([vi[1] for vi in v])
                mean_dt = np.mean([vi[2] for vi in v])
                print(f'{str(k):>30} {dim:4}: {mean_err:.3e}, {mean_dt:.5f}')
    # Save
    import pickle
    with open('results/timings_large.pkl', 'wb') as f:
        pickle.dump(res_large, f)
