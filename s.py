import pandas as pd
import numpy as np

# 创建一个空的数据框用于存储光谱数据
spectra_df = pd.DataFrame()

# 模拟计算得到的光谱数据
for i in range(5):  # 模拟5次计算
    nu = np.linspace(0, 1000, 1000)  # 频率点
    absorbtance = np.random.rand(1000)  # 模拟随机的吸收光谱数据

    # 创建一行数据，并添加到数据框中
    spectrum_data = pd.DataFrame({
        f'spectrum_{i+1}': absorbtance})

    spectra_df = pd.concat([spectra_df, spectrum_data], axis=1)

# 打印数据框
print(spectra_df.head())