# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:08:41 2016

@author: m.reuss
"""

import errno
from datetime import datetime
import numpy as np
import geopandas as gpd
import os
import networkx as nx
import pandas as pd
import time
import sys
from IPython.display import display
from scipy.cluster import vq
from shapely.geometry import Point
from shapely.geometry import LineString
from cycler import cycler
import os.path as path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
if mpl.__version__[0]=="1":
    STYLEPATH=path.join(path.dirname(__file__),"..","data","matplotlibrcEES.mplstyle")
else:
    STYLEPATH=path.join(path.dirname(__file__),"..","data","matplotlibrc36.mplstyle")
plt.style.use(STYLEPATH)
decimalSep=","
#CP.set_config_string(
#    CP.ALTERNATIVE_REFPROP_PATH,
#    'C:\\Program Files (x86)\\REFPROP\\')
np.seterr(divide='ignore', invalid='ignore')

#%%
bg_area="#fff0de"
bg_lines=(128/255, 130/255, 133/255)

# Tell matplotlib to use the locale we set above
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_ALL, '')
plt.rcParams['axes.formatter.use_locale'] = True
bbox=dict(facecolor='white', alpha=0.75, edgecolor='none',pad=0.15)
#%%
colorNumber=5
hatches=[" "]*colorNumber+["*"]*colorNumber+["\\"]*colorNumber+["/"]*colorNumber
colorList=plt.rcParams['axes.prop_cycle'].by_key()['color']

#%%
import types
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.collections import LineCollection

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()

RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
#%%H2 Constant Values at Normal Conditions
class H2Values (object):

    def __init__(self):
        self.M = 2.01588  # [kg/kmol], molare Masse von Wasserstoff
        # [J/kg K], spezifische Gaskonstante von Wasserstoff
        self.R_i = 4124.48269490247
        # [kg/m^3], Dichte von Wasserstoff im Normzustand
        self.roh_n = 0.089882
        # [-] Realgasfaktor von Wasserstoff im Normzustand
        self.Z_n = 1.00062387922965
        self.LHV_n = 119.833493175241  # [MJ/kg]
        self.g = 9.81  # [m/s^2], Erdbeschleunigung
        self.T_n = 273.15  # [K]
        self.p_n = 1.01325e5  # [Pa]

#%% Supporting Functions
def parabel(para, p):
    return (para[0] / 1e6) * (p / 1e5)**para[1] + \
        8.79676122460001e-06  # Parabelgleichung

def square(para, p):
    return para[0] * p**para[1] + para[2]  # squaregleichung
    

def getDiaPress(demArr, distArr, p_1, p_min):
    '''
    Calculation of Pipeline diameter and end pressure:
    Input Parameter:
    demArr=demand Array in kg/day
    distArr= distance Array in km
    p_1=Input Pressure at start of pipeline in bar
    p_min=minimal output pressure in bar
    
    deprecated!!! Not used anymore!!!
    '''
    #             Initialization              #
    V_para_parabel_20 = np.array([0.000125571318762396, 1.50162559878953])
    D_para_square_20 = np.array(
        [3.24859458677547e-06, 0.912591206027628, -0.166716162511868])
    Z_para_square_20 = np.array(
        [3.23101813258933e-09, 1.03880932425032, 1.00048097412768])
    T_m = np.array(20 + 273.15)  # K
    k = 0.02  # mm

    # Vanessas Diameter
    DK=np.array([0.1063, 0.1307, 0.1593, 0.2065, 0.3063, 0.3356,
    0.3844,0.432, 0.4796, 0.527, 0.578, 0.625, 0.671, 0.722, 0.7686, 0.814,
    0.864, 0.915, 0.96, 1.011, 1.058, 1.104, 1.155, 1.249, 1.342, 1.444,
    1.536])    #Average class of diameter

    # Less diameter variances
    #DK = np.linspace(0.1, 1.0, 901)  # Average class of diameter

    propH2 = H2Values()

    demHourly = demArr / 24 / 3600  # kg/day to kg/s

    distMeter = distArr * 1000  # km to m

    p_1 = p_1 * 1e5  # bar to Pa
    ###             Calculation                 ###
    res1 = len(distArr)
    res2 = demArr.shape[1]

    p_2 = np.zeros((res1, res2))
    w_1 = np.zeros((res1, res2))
    Re_1 = np.zeros((res1, res2))
    diameter = np.ones((res1, res2)) / 1000
    x = np.zeros((res1, res2))
    for i1 in range(demArr.shape[1]):
        for i2 in range(len(distArr)):
            while p_2[i2, i1] <= p_min * 1e5 or np.isnan(p_2[i2, i1]):
                # Calculation of Norm Volume Flow
                V_n = demHourly[0, i1] / propH2.roh_n  # m^3/s (i.N.)
                # Startwerte
                # Calculation of input density
                roh_1 = square(D_para_square_20, p_1[i2, i1])  # kg/m3
                # Volume flow at entrance
                V_1 = demHourly[0, i1] / roh_1  # m^3/s
                # inner diameter of the Pipeline
                diameter[i2, i1] = DK[x[i2, i1]]      # m
                # Velocity Entrance
                w_1[i2, i1] = V_1 / (np.pi * diameter[i2, i1]**2 / 4)
                # Calculation of dynamic viscosity
                eta_1 = parabel(V_para_parabel_20, p_1[i2, i1])  # Pa*s
                # Calculation of kinetic viscosity
                nu_1 = eta_1 / roh_1  # m^2/s
                # Calculation of reynolds number
                Re_1[i2, i1] = w_1[i2, i1] * diameter[i2, i1] / nu_1  # -
                # frcition coefficient
                # starting value
                alpha = np.e**(-1 * np.e**(6.75 - 0.0025 * Re_1[i2, i1]))
                lambda_1 = (64 / Re_1[i2, i1]) * (1 - alpha) + alpha * (-2 * np.log10((2.7 * (np.log10(
                    Re_1[i2, i1]))**1.2 / Re_1[i2, i1]) + (k / (3.71 * 1000 * diameter[i2, i1]))))**(-2)  # -
                # Simplification: Re_1 = Re_m --> lambda_m = lambda_1
                lambda_m = lambda_1
                # characteristics of the pipe
                # kg/(m s^2)=Pa
                C_1 = (lambda_1 * distMeter[i2] * roh_1 *
                       w_1[i2, i1]**2) / (diameter[i2, i1] * 2)
                # input pressure
                p_20 = p_1[i2, i1] - C_1  # Pa
                # assuption: average pressure ~ input pressure
                p_m0 = p_20  # [Pa)
                # assumption: avg real gas factor = f(p_m0) 
                Z_m = square(Z_para_square_20, p_m0)
                # compressibility factor
                K_m = Z_m / propH2.Z_n
                # pipe characteristics
                C = (lambda_m * 16 * propH2.roh_n * T_m * propH2.p_n *
                     K_m) / (np.pi**2 * propH2.T_n)  # kg Pa/m^3
                # outlet pressure
                p_2[i2, i1] = (p_1[i2, i1]**2 - (C * distMeter[i2]
                                                 * V_n**2) / diameter[i2, i1]**5)**0.5  # Pa

                if x[i2, i1] == len(DK):
                    break
                if p_2[i2, i1] <= p_min * 1e5 or np.isnan(p_2[i2, i1]):
                    x[i2, i1] += 1
                    x[i2:, i1:] = x[i2, i1]

    p_2 = p_2 * 1e-5
    diameter = diameter * 1000
    return diameter, p_2, w_1  # Diameter in mm and outlet pressure in bar

# %% Compressor Energy Demand per Stage (with isentropic coefficient)
# direct Method from Tietze


def getCompressionEnergyStage(p_1, p_2, T_1, eta_is_S):
    '''
    calculation of specific hydrogen compression energy in every compression stage
    Input:
    p_1=Inlet Pressure
    p_2=outlet Pressure
    T_1 = Inlet Temperature
    eta_is_S = isentropic efficiency
    '''
    import CoolProp.CoolProp as CP
    fluid = 'HYDROGEN'
    # fluid='REFPROP::HYDROGEN'
    # Entropy
    s = CP.PropsSI('S', 'T', T_1, 'P', p_1 *
                   100000, fluid)  # [T]=K, [P]=kPa, [h]=J/kg
    # Enthalpy input 
    h_1 = CP.PropsSI('H', 'P', p_1 * 100000, 'S', s, fluid)
    # isentropic enthalpy
    h_2_is = CP.PropsSI('H', 'P', p_2 * 100000, 'S', s,
                        fluid)  # [T]=K, [P]=kPa, [h]=J/kg
    # isentropic temperature
    # T_2_is = CP.PropsSI('T','P',p_2*100,'S',s,fluid); # [T]=K, [P]=kPa, [h]=J/kg
    # isentropic work
    w_is = (h_2_is - h_1) / 1000  # [kJ/kg], massenspez. Verdichterarbeit
    # compressor work
    w = w_is / eta_is_S  # [kJ/kg], massenspez. Verdichterarbeit
    w_spec = w / 3600
    # enthalpy after compression
    h_2 = w * 1000 + h_1  # [h]=J/kg
    #  Temperature after compression
    T_2 = CP.PropsSI('T', 'P', p_2 * 100000, 'H', h_2,
                     fluid)  # [T]=K, [P]=kPa, [h]=J/kg
    return [w_spec, T_2]

# %% CompressionDemand


def getCompressionEnergy(
        p_1,
        p_2,
        demand,
        T_1=20,
        eta_isen=0.88,
        eta_mech=0.95,
        p_highlow_max=2.1,
        max_stages=2):
    '''
    calculation of specific hydrogen compression energy
    Input:
    p_1=Inlet Pressure in bar
    p_2=outlet Pressure in bar
    demand = hydrogen demand in kg/day
    T_1 = Inlet Temperature
    eta_is_S = isentropic efficiency
    '''
    # eta_isen=0.92-p_2/880*(0.24)
    if p_2 > p_1:

        compressorStages = np.log(p_2 / p_1) / np.log(p_highlow_max)
        compressorStages = np.ceil(compressorStages).astype(int)

        if compressorStages > max_stages:
            compressorStages = max_stages
        p_highlow = (p_2 / p_1)**(1 / compressorStages)
        # Initialize
        p_in = np.zeros(compressorStages)
        p_out = np.zeros(compressorStages)
        T_in = np.zeros(compressorStages)
        T_out = np.zeros(compressorStages)
        w_stage = np.zeros(compressorStages)
        # Stagedependent Calculation
        for i in range(compressorStages):
            if i == 0:
                p_in[i] = p_1
                T_in[i] = 273.15 + T_1
            else:
                p_in[i] = p_out[i - 1]
                T_in[i] = 273.15 + 40.
            p_out[i] = p_in[i] * p_highlow
            w_stage[i], T_out[i] = getCompressionEnergyStage(p_in[i],
                                                             p_out[i],
                                                             T_in[i],
                                                             eta_isen)
        T_out = T_out - 273.15
        w_mech = np.sum(w_stage) / eta_mech
        P_shaft = demand * w_mech / 24
        #print(np)
        eta_motor = np.array([8e-5 * np.log(x)**4 - 0.0015 * np.log(x)**3 + 0.0061 * np.log(x)**2 + 0.0311 * np.log(x) + 0.7617 for x in P_shaft])
        eta_motor[eta_motor>0.98]=0.98
        P_el = P_shaft / eta_motor
        w_total = w_mech / eta_motor
    else:
        w_total = 0
        P_el = 0

    return w_total, P_el

#%%
def getFuelingStationInvest(n1, C1,
                            C0=212,
                            n0=400,
                            I0=600000,
                            scale=0.7,
                            learning=0.06,
                            installationFactor=1.3):
    '''
    scaling function for elaborating average cost for hydrogen refueling stations:
        Inputs:
            n1: number of stations
            c1: capacity of wished station
            n0: number of station - base case
            C0: capacity of base station
            I0: investment cost per station - base case
            installationFactor: muliplier for station cost
            learning: learning rate (double capacity: -x% costs)
            scale: scaling factor for increased sizing of stations
    '''
    V0=C0*n0
    V1=n1*C1
    beta=np.log2(1-learning)
    I1_avg=I0*((C1/C0)**scale)*((V1/V0)**beta)/(1+beta)*installationFactor
    return I1_avg
#%%
def createFolder(foldername):
    mydir = os.path.join(
        foldername, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

#%%
def setupLogger(PATHLOG):
    import logging
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(PATHLOG)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
#%%
def testBoolColumns(gdf):
    '''
    checking geopandas dataframe for non save-able datatypes in columns
    converts boolean to int (False:0, True:1)
    '''
    df = gdf.copy()
    for colname, coltype in df.dtypes.items():
        if coltype == 'bool':
            df[colname] = df[colname].astype('int')
        if coltype == 'object':
            if colname=="geometry": continue
            else: df[colname] = df[colname].astype('str')
    df.columns=[str(x) for x in df.columns]
    return df
#%%
def builtDemDistArray(demMax, distMax, res):
    if type(res)==list:
        resDist=res[0]
        resDem=res[1]
    else:
        resDist=res
        resDem=res

    distDamp = distMax / resDist
    distArr = np.linspace(distDamp, distMax, resDist)
    distArr = np.array([distArr]).T
    # Resolution Demand
    demDamp = demMax / resDem
    demArr = np.linspace(demDamp, demMax, resDem)
    demArr = np.array([demArr])
    return demArr, distArr