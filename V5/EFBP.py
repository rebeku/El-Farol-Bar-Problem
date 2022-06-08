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
    return rng.uniform(-1,1, size=(strategies, memory))


def minimize_squared_error(rng,predictions,observations,threshold,strategies):
        # calculate the absolute error of predictions
        # here, we discard the rightmost prediction as
        # this is the prediction for the future.
        # if we knew the correct answer for that,
        # we wouldn't need to predict it!
        errs = ((predictions[:, :-1] - observations)**2).sum(axis=1)

        return np.argmin(errs)

    
def make_centered_strategies(rng, strategies, memory):
    return rng.uniform(-1,1, size=(strategies, memory)) + 1/memory


def pick_optimal_choices(rng,predictions,observations,threshold,strategies):
    # how often does each predictor match the desired ooutcome?
    performances = np.tile(observations < threshold, (strategies, 1)) == (predictions[:, :-1] < threshold)
    
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

        strats = [
            strategy_func(rng, strategies, memory) for _ in range(agents)
        ]

        if start is None:
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

            # The columns begin at
            # t - m - 1 
            # t - m
            # ...
            # t - 1
            # as you go down the column you are looking back
            # to that week's history.
            # so the column beginning at *t - m - 1*
            # generates the prediction for week *t - m*
            # and the rightmost column generates a
            # prediction for next week.
            windows = np.vstack([
                hist[t-memory-i-1: t-i]
                for i in range(memory)
            ])

            for agent in range(agents):
                strat = strats[agent]
                # each row is a strategy
                # each column is predicted attendance
                # in increasing order.
                # the last column has the prediction for
                # next week
                predictions = strat.dot(windows)

                # these are the observations that we use to
                # to test our predictions.
                # note that the observation from column 0
                # is not used since its prediction would come
                # from a previous week's history.
                observations = windows[0, 1:]
                best_strat = best_strat_func(rng, predictions, observations, threshold, strategies)
                best_strats[agent, t] = best_strat

                pred = strat[best_strat].dot(windows[:,-1])
                pred_history[agent, t] = pred

            hist[t] = (pred_history[:, t] < threshold).sum()
            
            # stop the simulation if it reaches a stable cycle
            # it seems like these are generally of period 3 so just
            # look for those
            if t >= 2*memory + 6:
                if (hist[t-2:t+1] == hist[t-5:t-2]).all():
                    hist = hist[:t+1]
                    break
            t += 1
    
        self.hist = hist[2*memory:]
        self.best_strats = best_strats[:, 2*memory:]
        self.pred_history = pred_history[:, 2*memory:]
        self.strats = strats
        
    def first_a(self):
        return [self.strats[i][int(s), 0] for i, s in enumerate(self.best_strats)]
    
    def jth_a(self, j):
        return [self.strats[i][int(s), j] for i, s in enumerate(self.best_strats)]
        
        
        
start = np.array([80, 23, 35, 72])
                
efbp = EFBPSim(n_iter=1, memory=2, start=start)
#[efbp.strats[i][int(s), 0] for i, s in enumerate(efbp.best_strats)]

efbp.first_a()