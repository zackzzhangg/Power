from trainer import Trainer

if __name__ == "__main__":
    # 数据文件
    DATA_PATH = r"D:/workspace/Power/data_ts/synthetic_timeseries.csvsynthetic_timeseries.csv"

    # 实例化 Trainer —— 只传现有参数
    t = Trainer(
        data_path  = DATA_PATH,
        target_col = "target",
        time_col   = "date",
        test_ratio = 0.2,
        tscv_splits= 3,   # ← 唯一交叉验证参数
        seed       = 42
    )



    # 跑完整流程
    t.run()
