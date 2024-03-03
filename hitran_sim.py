# -*- coding: utf-8 -*-
from hapi import *
from numpy import arange
import numpy as np
import matplotlib.pyplot as plt
from pylab import show,plot,subplot,xlim,ylim,title,legend,xlabel,ylabel
import os
'''
HITRAN 模拟 10% H2O 在 7178 - 7192 cm-1 范围内，温度300K,压强 2 atm 
吸收光程 5cm 下的谱线
'''
db_begin('CH4')
fetch('CH4',6,1,6046.27,6047.73)
tableList()
describeTable('CH4')

#绘制线强图
x,y = getStickXY('CH4')

plt.figure()

plot(x,y)

xlabel('wavenumber($cm{-1}$)')
ylabel('$HCH4 linestrength  ')
# 用voigt线性计算吸收系数Kv
Nu,Coef = absorptionCoefficient_Voigt(((6,1),),'CH4',WavenumberStep=0.00095,Environment={'p':1,'T':297},
OmegaStep = 0.01,HITRAN_units=False,GammaL='gamma_self',Diluent={'air':0.9,'self':0.1})
Coef *= 0.1
plt.figure()
plot(Nu,Coef)
xlabel('wavenumber($cm{-1}$)')
ylabel('absorption cofficient($cm{-1}$)')
title('$CH4 absorption cofficient  @ 1atm,297K $')
absorbance = Coef*100
plt.figure()
plot(Nu,absorbance)
xlabel('wavenumber($cm{-1}$)')
ylabel('absorbance$)')
title('$CH4 absorbance  @ 1atm,297K $')

#计算吸收光谱
Nu , tras = transmittanceSpectrum(Nu,Coef,Environment={'l':4000})
plt.figure()
plot(Nu,tras)
xlabel('wavenumber($cm^{-1}$)')
ylabel('tras')
title('$1% CH4 absorption spectra @ 1atm, 296K,L=130cm $')
tras_1 = exp(-Coef*100)

plt.figure()
plt.plot(Nu,tras_1)
plot(Nu,tras)
plt.show()


# plot Gaussian Function
# 注：正态分布也叫高斯分布
# import matplotlib.pyplot as plt
# import numpy as np

# u1 = 0  # 第一个高斯分布的均值
# sigma1 = 1  # 第一个高斯分布的标准差
# x = np.arange(-5.55, 5.56, 0.01)
# # 表示第一个高斯分布函数
# y1 = 1-np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))*0.1
# # 表示第二个高斯分布函数
# plt.figure()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# plt.subplot(121)
# plt.plot(x, y1, 'b-', linewidth=2)
# plt.title("高斯分布函数图像")
# plt.show()

