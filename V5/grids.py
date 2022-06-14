"""
Generate data for grid plots.

Sample command:
python grids.py --smin 62 --smax 74 --mmin 2 --mmax 102 
"""


import numpy as np
import pandas as pd
import argparse
from itertools import product

from EFBP import EFBPSim, minimize_squared_error
from util import detect_cycle


rng = np.random.default_rng(874)


def pick_optimal_choices(rng,predictions,observations,threshold,strategies):
    # how often does each predictor match the desired ooutcome?
    performances = np.tile(observations < threshold, (strategies, 1)) == (predictions[:, :-1] < threshold)
    
    # add noise to randomize tie breaker
    scores = rng.uniform(0,0.1, size=strategies) + performances.sum(axis=1)
    return np.argmax(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smin', type=int, nargs='?',
                        help='minimum number of strategies in range')
    parser.add_argument('--smax', type=int, nargs='?',
                        help='maximum number of strategies in range')
    parser.add_argument('--mmin', type=int, nargs='?',
                        help="minimum memory length in range")
    parser.add_argument('--mmax', type=int, nargs='?',
                        help="maximum memory length in range")
    parser.add_argument('--n_trials', type=int, nargs='?', default=5,
                        help="number of trials for each memory length")
    parser.add_argument('--step', type=int, nargs='?', default=3,
                        help="step size for grid")
    parser.add_argument('--cf', type=str, nargs="?",default='mse',
                        help='Strategy selection function.  Options aer "mse" or "choice".' )

    
    args = parser.parse_args()

    threshold = 0.6
    agents = 10000
    distribution="uniform"
    writefile = "data/grid.csv"

    if args.cf == 'mse':
        cf = minimize_squared_error
    else:
        cf = pick_optimal_choices

    # placeholder to track results of each trial
    tmp = np.array(args.n_trials)

    strategies, memory = np.meshgrid(
        np.arange(args.smin, args.smax, args.step), 
        np.arange(args.mmin, args.mmax, args.step))

    stds = np.zeros((*strategies.shape, args.n_trials))

    (s, m) = strategies.shape

    s_range = list(range(s))
    m_range = list(range(m))
    iter_range = list(range(args.n_trials))

    does_not_cross = []

    for k,i,j in product(iter_range, s_range, m_range):

        s = strategies[i,j]
        m = memory[i,j]
        h = rng.choice(agents+1, size=m*2)
        n_iter = max(100, memory[i,j] * 3)

        sim = EFBPSim(
            memory=m, 
            strategies=s,
            threshold=threshold  * agents,
            start=h, 
            n_iter=n_iter,
            agents=agents,
            distribution="uniform",
            best_strat_func=cf,
            seed=rng.choice(100000)
        )

        end_window = max(m * 2, 50)

        # does it cross the threshold within the memory window?
        under_t = (sim.hist[-end_window:] < threshold).sum()
        crosses =  (under_t > 0 and under_t < end_window)

        cycle = detect_cycle(sim.hist, m)
        if cycle:
            end_window = cycle

        std = (sim.hist[-end_window:]).std()

        with open(writefile, "a", encoding="utf-8") as f:
            f.write(f"{s},{m},{threshold*agents},uniform,{args.cf},{crosses},{std}\n")