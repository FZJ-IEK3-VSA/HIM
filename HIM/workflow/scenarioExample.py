# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:00:59 2017

@author: m.reuss
"""
from HIM.utils import *
import os
                             #%%
dataPath=path.join(os.path.dirname(os.path.realpath(__file__)),"..","..","data")
dfTable = pd.read_excel(path.join(dataPath,"ImportTablesTechnologies.xlsx"), sheet_name=None, index_col=0)
dfTable["General"]=pd.read_excel(path.join(dataPath,"Scenarios.xlsx"), sheet_name="General", index_col=0).append(pd.read_excel(path.join(dataPath,"Scenarios.xlsx"), sheet_name="Example", index_col=0))
specificDemand=dfTable["General"].loc["specificDemand","General"]
mileage=dfTable["General"].loc["mileage","General"]
speed={"motorway":dfTable["General"].loc["truckSpeedHighway","General"],
       "urban": dfTable["General"].loc["truckSpeedRural","General"],
       "beeline": dfTable["General"].loc["truckSpeed","General"]}
beeline=[False, False]
weight="time"
clustering=bool(dfTable["General"].loc["clustering","General"])
clusterSize=dfTable["General"].loc["clusterSize","General"]
#targetFS=9968
targetCapacityFS=dfTable["General"].loc["targetStationSize","General"]
fuelingMax_kg_d=dfTable["General"].loc["utilization Station","General"]*targetCapacityFS
detourFactorPipeline=dfTable["General"].loc["detourFactorPipeline","General"]
detourFactorTruck=dfTable["General"].loc["detourFactorTruck","General"]
#%%
weightFtoF=detourFactorPipeline*pd.Series([1., 1.25, 1.25, 1.5, 1.5, 1.5, 1.75, 1.75, 2.],
                     index=["1to1","1to2","2to1","2to2","1to3","3to1","2to3","3to2","3to3"])

crs={'ellps': 'GRS80', 'no_defs': True, 'proj': 'utm', 'units': 'm', 'zone': 32}