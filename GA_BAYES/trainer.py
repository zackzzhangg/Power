# trainer.py  (完整)
import json, numpy as np, pandas as pd, optuna
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ga_utils import ga_optimize_weights


OPTIMIZERS = {
    "adam": Adam, "sgd": SGD, "adagrad": Adagrad,
    "adadelta": Adadelta, "adamax": Adamax, "nadam": Nadam
}

class Trainer:
    def __init__(self, data_path, target_col, time_col=None,
                 test_ratio=0.2, tscv_splits=3, seed=42):
        self.data_path   = data_path
        self.target_col  = target_col
        self.time_col    = time_col
        self.test_ratio  = test_ratio
        self.tscv_splits = tscv_splits
        self.seed        = seed
        self.rng         = np.random.default_rng(seed)

        self._load_and_split()
        self.scaler_x, self.scaler_y = MinMaxScaler(), MinMaxScaler()

        self.callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5)
        ]

    # ---------- 数据 ----------
    def _load_and_split(self):
        df = pd.read_csv(self.data_path)
        if self.time_col:
            df = df.sort_values(self.time_col)

        y = df[self.target_col].values.astype(np.float32)
        X = df.drop(columns=[self.target_col] +
                    ([] if self.time_col is None else [self.time_col])
                    ).values.astype(np.float32)

        cut = int(len(df) * (1 - self.test_ratio))
        self.X_train_raw, self.X_test_raw = X[:cut], X[cut:]
        self.y_train_raw, self.y_test_raw = y[:cut], y[cut:]

    def _scale(self):
        self.X_train = self.scaler_x.fit_transform(self.X_train_raw)
        self.X_test  = self.scaler_x.transform(self.X_test_raw)
        self.y_train = self.scaler_y.fit_transform(
            self.y_train_raw.reshape(-1, 1)).ravel()
        self.y_test  = self.scaler_y.transform(
            self.y_test_raw.reshape(-1, 1)).ravel()

        self.X_train = self.X_train.reshape(-1, 1, self.X_train.shape[1])
        self.X_test  = self.X_test.reshape(-1, 1, self.X_test.shape[1])

    # ---------- 粗筛 ----------
    def rough_hyper_scan(self, grid, n_iter=60):
        space = list(ParameterSampler(grid, n_iter=n_iter,
                                      random_state=self.seed))
        tscv = TimeSeriesSplit(n_splits=self.tscv_splits)
        results = []

        for params in tqdm(space, desc="TS CV Scan"):
            mse_fold = []
            for tr, val in tscv.split(self.X_train):
                m = self._build_model(**params)
                m.fit(self.X_train[tr], self.y_train[tr],
                      epochs=10, batch_size=params['batch_size'],
                      callbacks=self.callbacks, verbose=0)
                p = m.predict(self.X_train[val], verbose=0).ravel()
                mse_fold.append(mean_squared_error(self.y_train[val], p))
            results.append({
                'params': {k: (float(v) if isinstance(v, np.floating)
                               else int(v)   if isinstance(v, np.integer)
                               else v)
                           for k, v in params.items()},
                'mse': float(np.mean(mse_fold))
            })

        top = sorted(results, key=lambda d: d['mse'])[:max(1, int(0.2*len(results)))]
        new = {}
        for k, g in grid.items():
            vals = [r['params'][k] for r in top]
            if isinstance(g[0], (int, np.integer, float, np.floating)):
                new[k] = (min(vals), max(vals))
            else:
                new[k] = list(set(vals))

        json.dump(results, open("tscv_scan_log.json", "w", encoding="utf-8"),
                  indent=2)
        return new

    # ---------- 贝叶斯 ----------
    def bayes_optimize(self, space, n_trials=120):
        tscv = TimeSeriesSplit(n_splits=self.tscv_splits)

        def obj(t):
            hp = dict(
                batch_size = int(t.suggest_int('batch_size', *space['batch_size'])),
                activation = t.suggest_categorical('activation', space['activation']),
                optimizer  = t.suggest_categorical('optimizer',  space['optimizer']),
                n_hidden   = int(t.suggest_int('n_hidden',  *space['n_hidden'])),
                num_layers = int(t.suggest_int('num_layers',*space['num_layers'])),
                dropout    = t.suggest_float('dropout', *space['dropout']),
                lr         = t.suggest_float('lr', *space['lr'], log=True)
            )
            mse_fold = []
            for tr, val in tscv.split(self.X_train):
                m = self._build_model(**hp)
                m.fit(self.X_train[tr], self.y_train[tr],
                      epochs=50, batch_size=hp['batch_size'],
                      callbacks=self.callbacks, verbose=0)
                p = m.predict(self.X_train[val], verbose=0).ravel()
                mse_fold.append(mean_squared_error(self.y_train[val], p))
            return np.mean(mse_fold)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
        self.best_hp = study.best_params
        return self.best_hp

    # ---------- GA ----------
    def train_with_ga(self):
        val_cut = int(0.85 * len(self.X_train))
        X_tr, y_tr = self.X_train[:val_cut], self.y_train[:val_cut]
        X_val, y_val = self.X_train[val_cut:], self.y_train[val_cut:]

        m = self._build_model(**self.best_hp)
        m = ga_optimize_weights(m, X_val, y_val,
                                n_gen=30, pop_size=40, seed=self.seed)
        m.fit(X_tr, y_tr, epochs=300, batch_size=self.best_hp['batch_size'],
              callbacks=self.callbacks, verbose=0)
        self.final_model = m

    # ---------- 评估 ----------
    def evaluate(self):
        pred_n = self.final_model.predict(self.X_test, verbose=0).ravel()
        y_pred = self.scaler_y.inverse_transform(pred_n[:, None]).ravel()
        y_true = self.scaler_y.inverse_transform(self.y_test[:, None]).ravel()
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        mae    = mean_absolute_error(y_true, y_pred)
        print("\nBest hyper-params:")
        mapping = {
            'batch_size':'Batch Size', 'activation':'Layer Activation',
            'optimizer':'Model Optimizer', 'num_layers':'No. LSTM Layers',
            'n_hidden':'No. LSTM Cells',  'lr':'Learning Rate',
            'dropout':'Dropout Rate'
        }
        for k,v in self.best_hp.items():
            print(f"{mapping[k]:<16}: {v}")
        print(f"[Test]  RMSE={rmse:.4f}  MAE={mae:.4f}")
        return rmse, mae

    # ---------- Model builder ----------
    def _build_model(self, batch_size=32, activation='relu', optimizer='adam',
                     n_hidden=16, num_layers=2, dropout=0.2, lr=1e-3):
        n_hidden = int(n_hidden)
        model = Sequential([Input(shape=(1, self.X_train.shape[2]))])
        for i in range(num_layers):
            model.add(LSTM(n_hidden, activation=activation,
                           return_sequences=i < num_layers-1))
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=lr),
                      loss='mse')
        return model

    # ---------- run ----------
    def run(self):
        self._scale()
        grid = {
            'batch_size': [12, 24, 48, 96, 120],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
            'optimizer' : list(OPTIMIZERS.keys()),
            'n_hidden'  : np.arange(3, 37, 3),      # 3–36
            'num_layers': [2,3,4,5,6,8,10,12,15,20],
            'lr'        : list(np.logspace(-3, -1, 30)),  # 0.001–0.1
            'dropout'   : list(np.linspace(0.1, 0.5, 5))
        }
        refined = self.rough_hyper_scan(grid, n_iter=60)
        self.bayes_optimize(refined, n_trials=120)
        self.train_with_ga()
        return self.evaluate()
