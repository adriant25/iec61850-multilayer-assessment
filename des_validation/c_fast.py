"""Case C with 1 Gbps source links (near-Poisson arrivals at the 100 Mbps egress)."""
import os
import numpy as np
import pandas as pd
import validate_des as v

rng = np.random.default_rng(7)
k, t_sim, share = 10, 0.4, 0.3
rows = []
for rho in (0.3, 0.5, 0.7, 0.85):
    lam = rho / v.S
    pk = {}
    for j in range(k):
        n = rng.poisson(lam / k * t_sim)
        t = np.sort(rng.uniform(0, t_sim, n))
        pr = np.where(rng.uniform(size=n) < share, 6, 4)
        pk[f'SRC{j + 1}'] = list(zip(t, [v.S_BYTES] * n, pr))
    w = v.waits(v.run_des(pk, t_sim + 0.02, src_mbps=1000.0), warmup=0.02)
    th = v.cobham([lam * share, lam * (1 - share)], v.S, [6, 4])
    for p, name in ((6, 'high'), (4, 'low')):
        rows.append({'Case': f'C_priority_{name}', 'rho': rho, 'Samples': int((w.prio == p).sum()),
                     'DES_wait_us': w[w.prio == p].w_egr.mean() * 1e6, 'Theory_wait_us': th[p] * 1e6})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df['Error_pct'] = 100 * (df.DES_wait_us - df.Theory_wait_us) / df.Theory_wait_us
df.to_csv(os.path.join(v.OUT, 'des_validation_priority_1G_sources.csv'), index=False, float_format='%.4f')
print(df.round(3).to_string())
