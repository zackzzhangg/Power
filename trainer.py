# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, optuna, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterSampler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from mbb_utils import moving_block_bootstrap
from ga_utils import ga_optimize_weights

class Trainer:
    def __init__(self,
                 data_path,
                 target_col,
                 time_col=None,            # 如果有时间戳列，传列名
                 test_ratio=0.2,
                 block_len=None,
                 mbb_rounds=20,
                 seed=42):
        self.data_path = data_path
        self.target_col = target_col
        self.time_col = time_col
        self.test_ratio = test_ratio
        self.block_len = block_len
        self.mbb_rounds = mbb_rounds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._load_and_split()
        self.scaler_x, self.scaler_y = MinMaxScaler(), MinMaxScaler()

    # ---------- 0. 数据 ----------
    def _load_and_split(self):
        df = pd.read_csv(self.data_path)
        if self.time_col:
            df = df.sort_values(self.time_col)
        y = df[self.target_col].values.astype(np.float32)
        X = df.drop(columns=[self.target_col] + ([] if self.time_col is None else [self.time_col])).values.astype(np.float32)

        split_idx = int(len(df) * (1 - self.test_ratio))
        self.X_train_raw, self.X_test_raw = X[:split_idx], X[split_idx:]
        self.y_train_raw, self.y_test_raw = y[:split_idx], y[split_idx:]

    def _scale(self):
        self.X_train = self.scaler_x.fit_transform(self.X_train_raw)
        self.X_test  = self.scaler_x.transform(self.X_test_raw)
        self.y_train = self.scaler_y.fit_transform(self.y_train_raw.reshape(-1,1)).flatten()
        self.y_test  = self.scaler_y.transform(self.y_test_raw.reshape(-1,1)).flatten()

    # ---------- 1. MBB 粗筛 ----------
    def rough_hyper_scan(self, param_grid, n_iter=50):
        """在 MBB 复制体上随机扫描超参，返回候选区间建议。"""
        sample_space = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=self.seed))
        block_len = self.block_len or int(len(self.X_train) ** (1/3))
        boot_samples = moving_block_bootstrap(self.X_train, self.y_train, block_len,
                                              n_samples=self.mbb_rounds, rng=self.rng)
        results = []
        for params in tqdm(sample_space, desc="MBB Scan"):
            mse_list = []
            for Xb, yb in boot_samples:
                model = self._build_model(**params)
                model.fit(Xb, yb, epochs=20, batch_size=32, verbose=0)
                pred = model.predict(Xb, verbose=0).flatten()
                mse_list.append(mean_squared_error(yb, pred))
            results.append({'params': params, 'mse': np.mean(mse_list)})
        # 取前 20% 最优样本，统计各超参的 min/max → 收缩搜索空间
        top_k = int(len(results)*0.2) or 1
        top_res = sorted(results, key=lambda d: d['mse'])[:top_k]
        new_space = {}
        for p in param_grid:
            vals = [r['params'][p] for r in top_res]
            new_space[p] = (min(vals), max(vals)) if isinstance(param_grid[p][0], (int, float)) else list(set(vals))
        # 保存粗筛日志
        with open("mbb_scan_log.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return new_space

    # ---------- 2. BayesOpt ----------
    def bayes_optimize(self, search_space, n_trials=40):
        def objective(trial):
            hp = {
                'n_hidden': trial.suggest_int('n_hidden', *search_space['n_hidden']),
                'lr'      : trial.suggest_float('lr', *search_space['lr'], log=True)
            }
            model = self._build_model(**hp)
            model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)
            pred = model.predict(self.X_train, verbose=0).flatten()
            return mean_squared_error(self.y_train, pred)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_hp = study.best_params
        return self.best_hp

    # ---------- 3. GA 优化初始权重 ----------
    def train_with_ga(self):
        model = self._build_model(**self.best_hp)
        model = ga_optimize_weights(model, self.X_train, self.y_train,
                                    n_gen=30, pop_size=40, seed=self.seed)
        model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, verbose=0)
        self.final_model = model

    # ---------- 4. 评估 ----------
    def evaluate(self):
        pred = self.final_model.predict(self.X_test, verbose=0).flatten()
        y_true = self.y_test
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae  = mean_absolute_error(y_true, pred)
        print(f"[Test]  RMSE={rmse:.4f}   MAE={mae:.4f}")
        return rmse, mae

    # ---------- Helper ----------
    def _build_model(self, n_hidden=10, lr=1e-3):
        model = Sequential([
            Dense(n_hidden, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    # ---------- 主流程 ----------
    def run(self):
        self._scale()
        # 1) MBB 粗筛 —— 先用较宽网格
        param_grid = {
            'n_hidden': np.arange(4, 21),          # 4~20 个节点
            'lr'      : np.logspace(-5, -2, 50)    # 1e-5 ~ 1e-2
        }
        refined_space = self.rough_hyper_scan(param_grid, n_iter=60)
        # 2) 贝叶斯优化超参
        self.bayes_optimize(refined_space, n_trials=40)
        print("Best hyper-params:", self.best_hp)
        # 3) GA 微调权重
        self.train_with_ga()
        # 4) 评估
        return self.evaluate()
