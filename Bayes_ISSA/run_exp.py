from trainer import BayesISSALSTM

if __name__ == "__main__":
    DATA = r"D:\\workspace\\Power\\data_ts\\synthetic_timeseries.csvsynthetic_timeseries.csv"  # ← 修改成真实文件
    runner = BayesISSALSTM(csv_path=DATA,
                           target="target",   # ← 目标列
                           window=10,
                           test_ratio=0.2,
                           seed=42)
    runner.run()
