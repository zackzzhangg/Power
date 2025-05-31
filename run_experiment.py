from trainer import Trainer

if __name__ == "__main__":
    DATA_PATH = r"E:\Data\dataset\your_timeseries.csv"   # TODO
    t = Trainer(data_path=DATA_PATH,
                target_col="target",     # TODO
                time_col="date",         # 若无可置 None
                test_ratio=0.2,
                block_len=None,          # 默认 T^{1/3}
                mbb_rounds=30)
    t.run()
