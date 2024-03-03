import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#打印不同浓度下的透射谱
from hapi import *
import matplotlib
import pandas as pd
db_begin('CO2_20%')
fetch('CO2',2,1,6357.82,6359.67)
tableList()
describeTable('CO2')

def calculate_absorption_spectrum(concentration, temperature=296, pressure=1,path_length=1):
    volume_portion = concentration / 100
    nu, coef = absorptionCoefficient_Voigt(((2,1),),'CO2', HITRAN_units=False,
                                           Environment={"p": pressure, "T": temperature},
                                           WavenumberStep=0.0015,
                                           Diluent={'air': 1-volume_portion,
                                                    'self': volume_portion})
    coef *= volume_portion
    absorbance = coef * path_length

    return nu, absorbance
def calculate_transmission_spectrum(concentration,temperature=296,pressure=1,path_length=100):
    volume_portion = concentration / 100
    spectra_df = pd.DataFrame()

    for index,i in enumerate(volume_portion):

        Nu, Coef = absorptionCoefficient_Voigt(((2, 1),), 'CO2', WavenumberStep=0.0015, Environment={'p': pressure, 'T': temperature},
                                           OmegaStep=0.01, HITRAN_units=False, GammaL='gamma_self',
                                           Diluent={'air': 1 -i, 'self': i})
        Coef *= i
        _ ,tras = transmittanceSpectrum(Nu,Coef,Environment={'l':path_length})
        spectrum_data = pd.DataFrame({
            f'spectrum_{index + 1}': tras})
        spectra_df = pd.concat([spectra_df,spectrum_data],axis=1)
    return spectra_df.T

concentration = np.linspace(0.01,1,20)  # 10% CO2
temperature = 296  # 296 K
pressure = 1  # 1 atm
path_length =4000  # cm
spectra = calculate_transmission_spectrum(concentration,temperature,pressure,path_length)
print(spectra.head())
print(spectra.shape)
concentration_df = pd.DataFrame(concentration)


# 绘制每行的光谱数据
for (index, row),(index1,row1) in zip(spectra.iterrows(), concentration_df.iterrows()):
    concentration_value = row1.iloc[0].item()
    print(f'Concentration Value: {concentration_value}')
    plt.plot(row)
    plt.axhline(min(row),linestyle='--', color='gray')
    plt.text(len(row) - 10, min(row), f'Minimum: {min(row):.4f}', color='red', verticalalignment='center')
    plt.text(0, min(row) , f'Concentration: {row1.iloc[0].item():.2f}', color='red', verticalalignment='center')

# 添加标签和标题
plt.xlabel('Wavenumber')
plt.ylabel('Transmittance')
plt.title('Transmission Spectra for Different Concentrations')

# 显示图形
plt.show()