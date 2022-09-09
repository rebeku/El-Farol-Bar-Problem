from matplotlib import pyplot as plt
import numpy as np


def detect_cycle(hist, memory):
    """
    return the cycle length if there is a cycle longer 2*memory
    
    Repeated patterns of length less than 2*memory may not repeat
    indefinitely as attendance at each timestep relies on the previous
    2 * memory timesteps.
    """
    l = 2 * memory # minimum cycle length

    tail = hist[-l :]

    # we must leave space for an entire pattern repeat
    # of length *l* before the beginning of the tail
    for i in range(len(hist) - 2*l -1, 0, -1):
        if (hist[i:i+l] == tail).all():
            return len(hist) - l - i
    return None


def match_cycle(hist, cycle):
    l = len(cycle)
    offset = -1
    for i in range(l):
        if (hist[-2*l + i: -2*l + i + 4] == cycle[:4]).all():
            offset = i

    if offset == -1:
        return False

    reset = np.hstack([hist[-l + offset:], hist[-l:-l + offset]])
    return (reset - cycle).max() == 0