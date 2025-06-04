# # lstm_gridsearch_timesplit.py
# # ===============================================================
# # 生成合成时间序列 → 归一化 → 滑窗样本 → LSTM + GridSearch
# # * 时间序列交叉验证：TimeSeriesSplit(n_splits=3)
# # * 训练轮次：epochs=80，与另一条流水线对齐
# # * 预测后反归一化再计算 RMSE / MAE
# # ===============================================================

# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

# # 可复现随机种子
# np.random.seed(2025)
# import tensorflow as tf
# tf.random.set_seed(2025)

# # ---------------- 1. 生成时间序列数据 ----------------
# n_samples = 500
# dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
# y = (
#     np.sin(np.arange(n_samples) * 2 * np.pi / 30)
#     + np.arange(n_samples) * 0.01
#     + np.random.normal(0, 0.3, n_samples)
# )
# feature1 = np.random.normal(10, 2, n_samples)
# feature2 = np.random.uniform(5, 10, n_samples)

# df = pd.DataFrame(
#     {"date": dates, "feature1": feature1, "feature2": feature2, "target": y}
# )

# # ---------------- 2. 归一化 ----------------
# from sklearn.preprocessing import MinMaxScaler

# X_raw = df[["feature1", "feature2"]].values
# y_raw = df["target"].values.reshape(-1, 1)

# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# X_scaled = scaler_x.fit_transform(X_raw)
# y_scaled = scaler_y.fit_transform(y_raw)

# # ---------------- 3. 滑动窗口构样本 ----------------
# def create_sequences(features, target, window=10):
#     X, y = [], []
#     for i in range(len(features) - window):
#         X.append(features[i : i + window])
#         y.append(target[i + window])
#     return np.array(X), np.array(y)

# WINDOW = 10
# X, y = create_sequences(X_scaled, y_scaled, WINDOW)

# # ---------------- 4. 时间切分 (70/15/15) ----------------
# split1 = int(0.7 * len(X))
# split2 = int(0.85 * len(X))

# X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
# y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

# # ---------------- 5. LSTM 构建函数 ----------------
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, Adamax, Nadam

# def build_lstm_model(
#     num_layers=2,
#     num_cells=16,
#     activation="relu",
#     optimizer="adam",
#     learning_rate=0.001,
#     dropout=0.2,
# ):
#     model = Sequential()
#     for i in range(num_layers):
#         return_seq = i < num_layers - 1
#         if i == 0:
#             model.add(
#                 LSTM(
#                     num_cells,
#                     activation=activation,
#                     return_sequences=return_seq,
#                     input_shape=(X_train.shape[1], X_train.shape[2]),
#                 )
#             )
#         else:
#             model.add(LSTM(num_cells, activation=activation, return_sequences=return_seq))
#         model.add(Dropout(dropout))

#     model.add(Dense(1))

#     optim = {
#         "adam": Adam(learning_rate=learning_rate),
#         "sgd": SGD(learning_rate=learning_rate),
#         "adagrad": Adagrad(learning_rate=learning_rate),
#         "adadelta": Adadelta(learning_rate=learning_rate),
#         "adamax": Adamax(learning_rate=learning_rate),
#         "nadam": Nadam(learning_rate=learning_rate),
#     }
#     model.compile(optimizer=optim[optimizer], loss="mse")
#     return model


# # ---------------- 6. KerasRegressor 包装 ----------------
# from scikeras.wrappers import KerasRegressor

# reg = KerasRegressor(
#     build_fn=build_lstm_model,
#     epochs=80,                    # 训练轮次对齐
#     batch_size=24,                # param_grid 会覆盖
#     num_layers=2,
#     num_cells=8,
#     activation="relu",
#     optimizer="adam",
#     learning_rate=0.001,
#     dropout=0.2,
#     verbose=0,
#     random_state=2025,
# )

# # ---------------- 7. 参数网格 ----------------
# param_grid = {
#     "batch_size": [24, 48],
#     "num_layers": [2, 3],
#     "num_cells": [8, 16],
#     "activation": ["relu", "tanh"],
#     "optimizer": ["adam", "sgd"],
#     "learning_rate": [0.001, 0.01],
#     "dropout": [0.2, 0.4],
# }

# # ---------------- 8. 时间序列交叉验证 ----------------
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# tscv = TimeSeriesSplit(n_splits=3)   # 滚动窗口交叉验证

# grid = GridSearchCV(
#     estimator=reg,
#     param_grid=param_grid,
#     scoring="neg_mean_squared_error",
#     cv=tscv,
#     verbose=2,
# )

# grid_result = grid.fit(X_train, y_train)

# # ---------------- 9. 输出最优超参 ----------------
# print("\nBest hyperparameters:")
# for k, v in grid_result.best_params_.items():
#     print(f"  {k}: {v}")

# # ---------------- 10. 使用最优模型预测 ----------------
# best_est = grid_result.best_estimator_
# y_pred_scaled = best_est.predict(X_test).reshape(-1, 1)

# # 反归一化
# y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
# y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# # ---------------- 11. 计算指标 ----------------
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# mae = mean_absolute_error(y_true, y_pred)
# print(f"\nTimeSeriesCV Test RMSE: {rmse:.4f}")
# print(f"TimeSeriesCV Test MAE : {mae:.4f}")

# # ---------------- 12. 可视化 ----------------
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 4))
# plt.plot(y_true, label="True")
# plt.plot(y_pred, label="Predicted")
# plt.title("LSTM Grid-Search (TimeSeries CV) Prediction")
# plt.xlabel("Time Index")
# plt.ylabel("Target (original scale)")
# plt.legend()
# plt.tight_layout()
# plt.show()
