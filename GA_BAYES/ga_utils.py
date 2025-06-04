# ga_utils.py
# =========================================
# 遗传算法微调 Keras/LSTM 权重  (验证集目标)
# =========================================
import numpy as np
from deap import base, creator, tools, algorithms


# ---------- 展平 / 回填 ----------
def _flatten_weights(model):
    shapes = [w.shape for w in model.get_weights()]
    flat   = np.concatenate([w.flatten() for w in model.get_weights()])

    def setter(flat_arr):
        new_w, idx = [], 0
        for s in shapes:
            size = np.prod(s)
            new_w.append(flat_arr[idx:idx+size].reshape(s))
            idx += size
        model.set_weights(new_w)

    return flat.astype(np.float32), setter


# ---------- GA 主函数 ----------
def ga_optimize_weights(model,
                        X_val, y_val,
                        n_gen=30, pop_size=40,
                        cxpb=0.7, mutpb=0.2,
                        seed=42):
    """
    Parameters
    ----------
    model : tf.keras.Model
        已编译好、且有初始权重的模型
    X_val, y_val : ndarray
        用于评估适应度的验证集
    n_gen, pop_size : int
        GA 迭代次数、种群规模
    cxpb, mutpb : float
        交叉、变异概率
    seed : int
        随机种子

    Returns
    -------
    model : tf.keras.Model
        权重已替换为 GA 搜索到的最优个体
    """
    rng              = np.random.default_rng(seed)
    flat0, set_weight = _flatten_weights(model)
    n_param          = len(flat0)

    # ---------- DEAP 设置 ----------
    # 重复 import 时防止重复注册
    try:
        creator.FitnessMin
    except AttributeError:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: float(rng.uniform(-1, 1)))
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=n_param)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ---------- 适应度 ----------
    def eval_ind(ind):
        set_weight(np.asarray(ind, dtype=np.float32))
        loss = model.evaluate(X_val, y_val, verbose=0)
        return (loss,)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate",    tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutGaussian,
                     mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("select",  tools.selTournament, tournsize=3)

    # ---------- 运行 GA ----------
    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox,
                        cxpb=cxpb, mutpb=mutpb,
                        ngen=n_gen, halloffame=hof,
                        verbose=False)

    # ---------- 回填最优权重 ----------
    best_weights = np.asarray(hof[0], dtype=np.float32)
    set_weight(best_weights)
    return model
