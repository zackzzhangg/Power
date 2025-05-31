# -*- coding: utf-8 -*-
import numpy as np
from deap import base, creator, tools, algorithms

def flatten_weights(model):
    """把 Keras 权重展为 1-D 向量，并返回反向填充函数"""
    shapes = [w.shape for w in model.get_weights()]
    flat = np.concatenate([w.flatten() for w in model.get_weights()])

    def set_weights_from_flat(flat_arr):
        new_weights, idx = [], 0
        for s in shapes:
            size = np.prod(s)
            new_weights.append(flat_arr[idx : idx + size].reshape(s))
            idx += size
        model.set_weights(new_weights)

    return flat, set_weights_from_flat

def ga_optimize_weights(model, X_val, y_val, n_gen=20, pop_size=30, cxpb=0.7, mutpb=0.2, seed=42):
    """用 GA 在固定结构 & 超参数下微调初始权重，最小化验证集 MSE。"""
    rng = np.random.default_rng(seed)
    flat0, setter = flatten_weights(model)
    n_param = len(flat0)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: rng.uniform(-1, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=n_param)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(individual):
        setter(np.asarray(individual))
        loss = model.evaluate(X_val, y_val, verbose=0)
        return (loss,)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(1)
    pop, _ = algorithms.eaSimple(pop, toolbox,
                                 cxpb=cxpb, mutpb=mutpb,
                                 ngen=n_gen, halloffame=hof,
                                 verbose=False)
    setter(np.asarray(hof[0]))
    return model
