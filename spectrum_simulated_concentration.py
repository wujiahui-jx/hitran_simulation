import random
from hapi import *
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plb
import scipy.constants as C
import tensorflow as tf
from pandas import read_csv #使用Pandas导入csv数据
from pandas.core.frame import DataFrame
from tempfile import NamedTemporaryFile
from os.path import getsize
import matplotlib.pyplot as plt

#载入组分和对应编号
hitran_component_name = pd.read_excel('hitran_name.xlsx',header=None)
hitran_component_name.columns = ['label','component','component_name']
print(hitran_component_name)
print(type(hitran_component_name))
print("The program is used to form blended absorption spectra")
db_begin('data_hitran')#下载数据保存路径

# 读入混合组分和摩尔分数
component_number = int(input("please input the number of mixture component:"))
component_list = pd.DataFrame()
component_concentration = pd.DataFrame()


for i in range(component_number):
    component = input("please input component name:")
    s = pd.Series({'concentration': float(input("please input component mixture concentration :"))})
    component_concentration = pd.concat([component_concentration, s], ignore_index=True)
    bool = hitran_component_name.component.str.contains(str(component))
    component_list = pd.concat([component_list, hitran_component_name[hitran_component_name['component'] == component]],
                               ignore_index=True)

component_concentration.columns = ['concentration']
component_list = pd.concat([component_list, component_concentration], axis=1)

print(component_list)

#fetch(TableName, M, I, numin, numax)
# vmin = float(input("please input the vmin:"))
# vmax = float(input("please input the vmax:"))
vmin = 6358.9
vmax = 6360.5
for i in range(component_number):
    fetch(component_list.iloc[i].iat[1],component_list.iloc[i].iat[0],1, vmin, vmax)  # 下载数据

#生成单组分的吸收光谱
def calculate_absorption_spectrum(concentration, path_length,gas_table, temperature=296, pressure=1):
    # temperature = K
    # pressure = atm
    volume_portion = concentration / 1000000
    nu, coef = absorptionCoefficient_Voigt(SourceTables=gas_table, HITRAN_units=False,
                                           Environment={"p": pressure, "T": temperature},
                                           WavenumberStep=0.0602,
                                           Diluent={'air': (1 - volume_portion),
                                                    'self': volume_portion})
    coef *= volume_portion
    absorbance = coef * path_length

    return nu, absorbance


#生成混合吸收光谱
def date_simulated(p,l,T):
    #生成各组分随机浓度
    global component_list
    component_random_concentration = pd.DataFrame()
    for i in range(component_number):
        # if i == 0:
        #     a = 1
        # if i == 1:
        #     a = 1000
        # if i == 2:
        #     a = 200
        #random_concentration = pd.Series({'concentration':random.uniform(0,component_list.iloc[i].iat[3])})
        random_concentration = pd.Series({'concentration': random.uniform(0, 1)})
        component_random_concentration = pd.concat([component_random_concentration,random_concentration], ignore_index=True)
    #调用函数计算各组分吸收光谱
    absorbtance_mix = pd.DataFrame()
    nu_mix = pd.DataFrame()
    for i in range(component_number):
        nu,absorbtance = calculate_absorption_spectrum(component_random_concentration.iloc[i].iat[0],l,component_list.iloc[i].iat[1],T,p)
        absorbtance_mix= pd.concat([absorbtance_mix, pd.DataFrame(absorbtance)], axis=1)
        if len(nu_mix) <= len(nu):
            nu_mix = pd.DataFrame(nu)
    absorbtance_mix.fillna(0, inplace=True)
    absorbtance_mix['mix'] = absorbtance_mix.apply(lambda x: x.sum(), axis=1)
    return nu_mix,pd.DataFrame(absorbtance_mix['mix']),component_random_concentration.values.tolist()


sample_label = pd.DataFrame()
sample_spectra = pd.DataFrame()
for i in range(50):
    print(i)
    nu, blended_spectra, concentration_list = date_simulated(1, 500, 500)
    label = [0 if concentration_list[i]==0 else 1 for i in range(component_number)]
    for j in range(component_number):
        label.append(concentration_list[j][0])
    label = pd.DataFrame(label).T
    sample_label = pd.concat([sample_label,label])
    blended_spectra = blended_spectra.T
    sample_spectra = pd.concat([sample_spectra,blended_spectra])
sample_spectra = sample_spectra.reset_index(drop=True)
sample_spectra = sample_spectra.iloc[:,0:3321]



#将数据存入.npy文件
np.save('光谱数据.npy',sample_spectra)
np.save('标签数据验证集.npy',sample_label)
print("dataset is formed!")




plt.figure()
plt.plot(sample_spectra)
plt.show()

