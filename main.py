# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:05:45 2023

@author: 11527
"""
from hapi import *
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.signal import argrelextrema
matplotlib.use('TkAgg')
'''
HITRAN 模拟 20% H2O 在 6360.51 - 6362.05 cm-1 范围内，温度296K,压强 1 atm 
吸收光程 40000cm 下的谱线
'''
db_begin('CO2_20%')
fetch('CO2',2,1,6358.9,6360.51)
tableList()
describeTable('CO2')

#绘制线强图
x,y = getStickXY('CO2')
# 用voigt线性计算吸收系数Kv
concentration = 1/100
Nu,Coef = absorptionCoefficient_Voigt(((2,1),),'CO2',WavenumberStep=0.0015,Environment={'p':1,'T':297},
OmegaStep = 0.01,HITRAN_units=False,GammaL='gamma_self',Diluent={'air':1-concentration,'self':concentration})
Coef *= 0.1
#计算吸光度光谱
plt.figure()
plt.plot(Nu,Coef*4000)
plt.xlabel('wavenumber($cm^{-1}$)')
plt.ylabel('trasition')
plt.title('$ CO2 absorption spectra @ 1atm, 296K,L=130cm $')
#计算投射光谱
Nu , tras_hitran = transmittanceSpectrum(Nu,Coef,Environment={'l':100})

#计算投射光谱
plt.figure()
plt.plot(Nu,tras_hitran)
plt.xlabel('wavenumber($cm^{-1}$)')
plt.ylabel('trasition')
plt.title('$ CO2 absorption spectra @ 1atm, 296K,L=130cm $')

path=os.getcwd()
print(1)
# 定义源文件和目标文件的路径
source_file_path = 'F:\Wujiahui\pythonProject\\CO2_20.npy'
data = np.load('CO2_20.npy')
plt.figure()
plt.plot(data)
plt.show()
'''# 使用with语句打开文件，确保它们在处理后被正确关闭
with open(source_file_path, 'r') as source_file:
    lines = source_file.readlines()
    print(type(lines))

# 获取从第25行开始的内容

selected_lines = lines[22:]
data_array = np.array([list(map(float, item.split('\t'))) for item in selected_lines if '\t' in item])
print(np.argmax(data_array ))
wave = data_array[:,0]
tras = data_array[:,1]
'''
# 拟合多项式，这里选择用1次多项式，你可以根据需要选择不同次数的多项式
degree = 4
fit_points_front= 250
fit_points_back = 100

x = np.arange(len(data))

coefficients = np.polyfit(np.concatenate([x[:fit_points_front], x[-fit_points_back:]]),
                          np.concatenate([data[:fit_points_front], data[-fit_points_back:]]),
                          degree)
# 生成拟合曲线的 y 值
y_fit = np.polyval(coefficients, x)

# 绘制原始数据和拟合曲线
plt.figure()
plt.plot(x, data, label='Original Data')
plt.plot(x, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
plt.legend()
plt.show()

tras_spectra = data/y_fit
simu2 = pd.read_csv('spectraplot.csv')
simu2 = exp(-simu2)
plt.figure()
plt.plot(tras_hitran,label='hiran_Sim',color = 'b')
plt.plot(tras_spectra,label='experiment data',color ='r')
plt.show()
index1 = np.argmin(tras_hitran)
index2 = np.argmin(tras_spectra)
print(index1)
print(index2)

plt.figure()
plt.plot(tras_hitran,label='hiran_Sim',color = 'b')
plt.plot(tras_spectra,label='experiment data',color ='r')
plt.show()
sub = abs(index1-index2)
lh = len(tras_hitran)
tras_hitran1 =tras_hitran[sub:]
plt.figure()
plt.plot(tras_hitran1,label='hiran_Sim',color = 'b')
plt.plot(tras_spectra,label='experiment data',color ='r')
plt.show()

#透射光谱计算FWHM
def FWHM(signal):
    half_index = np.argmin(signal)
    half = min(signal)*2
    #生成半高直线
    length = len(signal)
    half_line = half*np.ones(length)
    diff = abs(half_line - signal)

    point0 = np.argmin(diff[:half_index])
    point1 = np.argmin(diff[half_index:])+half_index
    fwhm = signal[point1] - signal[point0]
    plt.figure()
    plt.plot(signal,label='Spectrum')
    plt.axvline(x=point0, color='r', linestyle='--', label='Half Maximum')
    plt.axvline(x=point1, color='r', linestyle='--')
    plt.axhline(y=half)
    plt.show()
    print('光谱FWHM为：',fwhm)
    return fwhm
f1 = FWHM(tras_hitran)
f2 = FWHM(tras_spectra)
wavelengths = np.arange((len(tras_hitran)))
print(f1)
print(f2)
half = min(tras_spectra)*2
length = len(tras_spectra)

half_line = half * np.ones(length)
plt.plot(tras_spectra,label='Spectrum')
plt.axvline(x=585, color='r', linestyle='--', label='Half Maximum')
plt.axhline(y=0.6491)
plt.legend()
plt.show()
plt.figure()
plt.plot(abs(half_line-tras_spectra))
plt.show()