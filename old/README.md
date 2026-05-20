## rITE/ITE experiment pipeline

All experiment code is in `paper/rcite_sim/` and writes outputs to:

- `paper/data/results.npz` (single compressed results bundle)
- `paper/figs/fig_exact_and_ideal.{pdf,png}`
- `paper/figs/fig_noise_sweep.{pdf,png}`
- `paper/figs/fig_sample_complexity.{pdf,png}`

### Run

From repo root:

`python paper/run_experiments.py`

Optional overrides:

`python paper/run_experiments.py --n 4 --g 1.4 --beta-min 0.5 --beta-max 5 --beta-num 10 --M-exact 500 --M-circuit 30 --shots 2000 --noise-levels 0 1e-4 1e-3`

### What it produces

- Energy vs `beta`
- Spin-spin covariance vs `beta`
- Trace distance vs `beta`
- Estimated sample complexity vs `beta` for a target trace-distance threshold (`--target-trace-distance`)

The code does **not** edit notebooks and is fully standalone/reusable from scripts.
