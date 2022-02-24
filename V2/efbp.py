import numpy as np


def make_strategies(rng, strategies, memory):
    # weights associated with each strategy for each week of memory
    w = rng.uniform(-1, 1, size=(strategies, memory))

    # scale each row to sum to 1
    # this implies that predictions will be centered
    # around mean of memory window
    return 1/w.sum(axis=1).reshape((strategies,1)) * w


def run_simulation(
    agents = 100, # number of agents
    threshold = 60, # threshold for attendance
    strategies = 10, # number of strategies
    # number of weeks back in predictor function
    # AND number of weeks back to look when selecting a predictor
    memory = 8,
    # number of rounds to run the simulation
    n_iter = 500,
    # random seed for numpy
    seed=23
):
    """
    Run a simulation of the El Farol Bar Problem
    for *n_iter* weeks.  Return the weekly attendance history,
    the strategy chosen by each agent, and the prediction
    made by each agent each week using their optimal predictor.
    """
    rng = np.random.default_rng(seed)

    # each row is a strategy
    strats = [
        make_strategies(rng, strategies, memory + 1) for _ in range(agents)
    ]

    start = rng.uniform(agents, size=(memory*2))

    # weekly attendance count
    # the first 2*memory weeks are randomly generated
    # to seed the strategies
    hist = np.hstack([start, np.zeros(n_iter)]).astype(int)

    # index of week
    # we need some starting history to begin making selections
    t = memory * 2

    # Record the index of the optimal strategy 
    # on each iteration.
    # each row corresponds to an agent
    # each column corresponds to a week
    best_strats = np.zeros((agents, len(hist)))

    # record each agent's prediction on each iteration
    pred_history = np.zeros((agents, len(hist)))

    while t < memory * 2 + n_iter:
        
        # construct time windows for evaluating strategies
        windows = np.vstack([
            hist[t-memory-i: t-i]
            for i in range(memory)
        ])
        
        # Each column contains the memory window for a particular week.
        # The rightmost column is the most recent.
        windows = np.vstack([windows, np.ones(shape=(1, memory), dtype=int)])
        
        for agent in range(agents):
            strat = strats[agent]
            # each row is a strategy
            # each column is the predicted attendance for week t-i
            predictions = strat.dot(windows)
            observations = windows[0, :]
            errs = np.abs(predictions - observations).sum(axis=1)

            best_strat = np.argmin(errs)
            best_strats[agent, t] = best_strat

            pred = strat[best_strat].dot(windows[:,-1])
            pred_history[agent, t] = pred

        hist[t] = (pred_history[:, t] < threshold).sum()
        t += 1
        
    return hist, best_strats, pred_history