# issa_utils.py  -------------------------------------------------------------
import numpy as np
from numpy.random import default_rng


# -------- Weight flatten / unflatten ---------------------------------------
def flatten_model(model):
    """Return flat weight vector + setter(flat_vec)."""
    shapes = [w.shape for w in model.get_weights()]
    flat   = np.concatenate([w.flatten() for w in model.get_weights()])

    def setter(flat_vec):
        new_w, idx = [], 0
        for s in shapes:
            sz = np.prod(s)
            new_w.append(flat_vec[idx:idx+sz].reshape(s))
            idx += sz
        model.set_weights(new_w)
    return flat.astype(np.float32), setter


# -------- Improved Sparrow Search Algorithm --------------------------------
class ISSA:
    """
    Improved SSA: Tent chaotic init + adaptive weight + Lévy flight + spiral.
    """
    def __init__(self, obj_func, dim, *, n_pop=30, n_iter=50,
                 lb=-1.0, ub=1.0, seed=42):
        self.f, self.dim = obj_func, dim
        self.n_pop, self.n_iter = n_pop, n_iter
        self.lb, self.ub = lb, ub
        self.rng = default_rng(seed)

        # Tent chaotic initialization
        chaos = self.rng.random((n_pop, dim))
        chaos = np.where(chaos < .7, chaos/0.7, (1-chaos)/0.3)
        self.X = lb + chaos * (ub - lb)

        self.fit = np.apply_along_axis(self.f, 1, self.X)
        self.best_idx = self.fit.argmin()
        self.best = self.X[self.best_idx].copy()
        self.best_val = self.fit[self.best_idx]

    # Lévy step
    def _levy(self, beta=1.5):
        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = self.rng.normal(0, sigma, self.dim)
        v = self.rng.normal(0, 1,     self.dim)
        return u / (np.abs(v)**(1/beta))

    def run(self):
        for t in range(1, self.n_iter+1):
            n_dis = int(.2*self.n_pop)
            R     = self.rng.random()
            for i in range(self.n_pop):
                if i < n_dis:                         # discoverers
                    w = 0.9 - 0.5 * t / self.n_iter   # adaptive weight
                    self.X[i] += w*self.rng.normal(0,1,self.dim) if R<.8 \
                                  else self._levy()
                else:                                 # followers
                    spiral = np.cos(2*np.pi*self.rng.random()) * \
                             np.exp(-t/self.n_iter)
                    self.X[i] += spiral * (self.best - self.X[i])

                self.X[i] = np.clip(self.X[i], self.lb, self.ub)

            self.fit = np.apply_along_axis(self.f, 1, self.X)
            idx = self.fit.argmin()
            if self.fit[idx] < self.best_val:
                self.best_val = self.fit[idx]
                self.best     = self.X[idx].copy()
        return self.best, self.best_val
