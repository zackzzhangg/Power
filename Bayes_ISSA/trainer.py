# trainer.py  ---------------------------------------------------------------
import numpy as np, pandas as pd, optuna
from numpy.random import default_rng
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from issa_utils import flatten_model, ISSA


class BayesISSALSTM:
    """Bayesian hyper-param + ISSA weight tuning for LSTM."""

    def __init__(self, csv_path, target,
                 window=10, test_ratio=.2, seed=42):
        self.csv, self.target = csv_path, target
        self.window, self.test_ratio = window, test_ratio
        self.seed = seed
        self.rng  = default_rng(seed)

        self._prepare_data()
        self.callbacks = [
            EarlyStopping(monitor='loss', patience=10,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', patience=5, factor=.5)
        ]

    # ---------- data & sliding window ----------
    def _prepare_data(self):
        df = pd.read_csv(self.csv).sample(frac=1, random_state=self.seed)
        num_df = df.drop(columns=[self.target]).select_dtypes(include=["number"])
        X_raw  = num_df.values.astype(np.float32)
        y_raw = df[self.target].values.astype(np.float32)

        X_seq, y_seq = [], []
        for i in range(len(df) - self.window):
            X_seq.append(X_raw[i:i+self.window])
            y_seq.append(y_raw[i+self.window])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        cut = int((1-self.test_ratio)*len(X_seq))
        self.X_tr, self.X_te = X_seq[:cut], X_seq[cut:]
        self.y_tr, self.y_te = y_seq[:cut], y_seq[cut:]

        self.sx, self.sy = MinMaxScaler(), MinMaxScaler()
        shp = self.X_tr.shape
        self.X_tr = self.sx.fit_transform(self.X_tr.reshape(-1, shp[-1])
                          ).reshape(shp)
        self.X_te = self.sx.transform(self.X_te.reshape(-1, shp[-1])
                          ).reshape(self.X_te.shape)
        self.y_tr = self.sy.fit_transform(self.y_tr.reshape(-1, 1)).ravel()
        self.y_te = self.sy.transform(self.y_te.reshape(-1, 1)).ravel()

    # ---------- model ----------
    def _build(self, n_cells, n_layers, lr, dropout):
        model = Sequential()
        model.add(Input(shape=self.X_tr.shape[1:]))   # ← 新增 Input 层

        for i in range(n_layers):
            model.add(
                LSTM(n_cells,
                    activation='tanh',
                    return_sequences=(i < n_layers - 1))
            )
            if dropout:
                model.add(Dropout(dropout))

        model.add(Dense(1))
        model.compile(optimizer=Adam(lr), loss='mse')
        return model

    # ---------- bayesian optuna ----------
    def bayes(self, trials=60):
        def obj(t):
            hp = dict(
                n_cells  = t.suggest_int   ('n_cells',  8, 64, step=8),
                n_layers = t.suggest_int   ('n_layers', 1, 3),
                lr       = t.suggest_float ('lr', 1e-5, 5e-3, log=True),
                dropout  = t.suggest_float ('dropout', 0.0, 0.3)
            )
            m = self._build(**hp)
            m.fit(self.X_tr, self.y_tr, epochs=50,
                  batch_size=32, verbose=0)
            p = m.predict(self.X_tr, verbose=0).ravel()
            return mean_squared_error(self.y_tr, p)
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(obj, n_trials=trials, show_progress_bar=True)
        self.hp = study.best_params
        return self.hp

    # ---------- issa weight fine-tune ----------
    def issa(self, n_pop=30, n_iter=50):
        base = self._build(**self.hp)
        flat0, setter = flatten_model(base)

        def f(vec):
            setter(vec.astype(np.float32))
            return base.evaluate(self.X_tr, self.y_tr, verbose=0)

        best, _ = ISSA(f, dim=len(flat0), n_pop=n_pop,
                       n_iter=n_iter, lb=-1, ub=1, seed=self.seed).run()
        setter(best)

        base.fit(self.X_tr, self.y_tr, epochs=200,
                 batch_size=32, callbacks=self.callbacks, verbose=0)
        self.model = base

    # ---------- evaluate ----------
    def evaluate(self):
        p = self.model.predict(self.X_te, verbose=0).ravel()
        y_pred = self.sy.inverse_transform(p[:,None]).ravel()
        y_true = self.sy.inverse_transform(self.y_te[:,None]).ravel()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        print("\nBest hyper-params:", self.hp)
        print(f"[Test]  RMSE={rmse:.4f}  MAE={mae:.4f}")
        return rmse, mae

    # ---------- pipeline ----------
    def run(self):
        self.bayes(trials=60)
        self.issa(n_pop=30, n_iter=50)
        return self.evaluate()
