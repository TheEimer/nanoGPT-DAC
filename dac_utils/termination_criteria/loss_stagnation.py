def loss_stagnation(iterations, statistics):
    if len(statistics['loss_history']) < 5:
        return False
    return abs(statistics['loss_history'][-1] - statistics['loss_history'][-5]) < statistics['loss_history'][-1]*0.05
