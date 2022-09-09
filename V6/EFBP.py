import numpy as np


def make_unbiased_strategies(rng, strategies, memory):
    # weights should sum to 1
    # essentially, we are partitioning the [0,1] interval
    # and taking the size of each sub-interval
    # TODO: add negative weights?
    w = rng.uniform(-1, 1, size=(strategies, memory-1))
    w.sort(axis=1)
    offsets = np.hstack([w[:, :], np.ones(shape=(strategies,1))])
    return offsets - np.hstack([np.zeros(shape=(strategies,1)), w[:, :]])


def make_uniform_strategies(rng, strategies, memory):
    return np.flip(rng.uniform(-1,1, size=(strategies, memory)), axis=1)


def minimize_squared_error(rng,predictions,observations,threshold,strategies):
        # calculate the absolute error of predictions
        # here, we discard the rightmost prediction as
        # this is the prediction for the future.
        # if we knew the correct answer for that,
        # we wouldn't need to predict it!
        errs = ((predictions - observations)**2).sum(axis=1)

        return np.argmin(errs)

    
def make_centered_strategies(rng, strategies, memory):
    return rng.uniform(-1,1, size=(strategies, memory)) + 1/memory


def pick_optimal_choices(rng,predictions,observations,threshold,strategies):
    # how often does each predictor match the desired outcome?
    performances = np.tile(observations < threshold, (strategies, 1)) == (predictions < threshold)
    
    # add noise to randomize tie breaker
    scores = rng.uniform(0,0.1, size=strategies) + performances.sum(axis=1)
    return np.argmax(scores)


class EFBPSim:
    def __init__(
        self,
        agents = 100, # number of agents
        threshold = 60, # threshold for attendance
        strategies = 10, # number of strategies
        # number of weeks back in predictor function
        # AND number of weeks back to look when selecting a predictor
        memory = 8,
        # number of rounds to run the simulation
        n_iter = 500,
        # current options are "unbiased", "uniform"
        distribution="unbiased",
        # alternately, caller may pass a new strategy function
        strategy_func=None,
        # function for selecting the best strategy
        best_strat_func=minimize_squared_error,
        # starting history
        start=None,
        # file to write attendance history as it is generated
        writefile=None,
        # random seed for numpy
        seed=23
    ):
        
        rng = np.random.default_rng(seed)

        # each row is a strategy
        if not strategy_func:
            if distribution=="unbiased":
                strategy_func =  make_unbiased_strategies
            elif distribution=="uniform":
                strategy_func = make_uniform_strategies

        # shape is (agents, strategies, memory)
        strats = np.stack([
            strategy_func(rng, strategies, memory) for _ in range(agents)
        ])

        if start is None:
            start = rng.choice(agents, size=(memory*2))

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
        best_strats = np.zeros((agents, len(hist)), dtype=int)
        
        # set up the predictions for initial history
        windows = np.vstack([
            hist[t-2*memory+i: t-memory + i]
            for i in range(memory)
        ])
        
        

        pred_history = np.zeros((agents, strategies, len(hist)))
        pred_history[:, :, memory:2*memory] = strats.dot(windows)
        
        best_preds = np.zeros(agents)

        while t < memory * 2 + n_iter:
            window = hist[t-memory:t]
            pred_history[:,:, t] = strats.dot(window)
            
            for agent in range(agents):
                strat = strats[agent]

                predictions = pred_history[agent, :, t-memory:t]
                observations = hist[t-memory:t]
                best_strat = best_strat_func(rng, predictions, observations, threshold, strategies)
                best_strats[agent, t] = best_strat

                best_preds[agent] = pred_history[agent, best_strat, t]

            hist[t] = (best_preds < threshold).sum()
            
            if writefile:
                with open(writefile, "a") as f:
                    f.write(str(hist[t]) + ",")
            
            # stop the simulation if it reaches a stable cycle
            # it seems like these are generally of period 3 so just
            # look for those
            """
            if best_strat_func==minimize_squared_error and t >= 2*memory + 6:
                if (hist[t-2:t+1] == hist[t-5:t-2]).all():
                    hist = hist[:t+1]
                    break
            """

            t += 1
    
        self.t = t
        self.hist = hist[2*memory:t]
        self.best_strats = best_strats[:, 2*memory:t]
        self.pred_history = pred_history[:,:, 2*memory:t]
        self.strats = strats
        
    def first_a(self):
        return [self.strats[i][int(s), 0] for i, s in enumerate(self.best_strats)]
    
    def jth_a(self, j):
        return [self.strats[i][int(s), j] for i, s in enumerate(self.best_strats)]