# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:06:45 2017

@author: Markus Reuß and Paris Dimos
"""
from HIM import dataHandling as sFun
from HIM import optiSetup as optiFun
import pandas as pd
import networkx as nx
import numpy as np
import geopandas as gpd
#%%
def shortest_path_calculation(nxGraph, dfSources, dfSinks, weight="weightedDistance", distribution=False):
    '''
    Create shortest path matrix for the calculation from all sinks to all sources
    input:
        -nxGraph: networkx graph with all edges and weights
        -dfSources: geopandas dataframe with source locations
        -dfSinks: geopandas dataframe with sink locations
        weight: str, name of the weight that shall be used for the dijkstra
        distribution: boolean, if the system wants distribution or transmission
    return:
        pandas dataframe with distance matrice from every source (row) to every sink (col)
    '''
    print("-------Calculation time for shortest_path_length begins-------")
    dictA={}
    sinkIDs=dfSinks.index
    for sourceID in dfSources.index:
         dictA[sourceID]=nx.shortest_path_length(nxGraph, source=sourceID, weight=weight)
    dictB={}
    for sourceID in dfSources.index:
        dictB[sourceID]={}
        if distribution:
            sinkIDs=dfSinks.loc[dfSinks["ClusterID"]==sourceID].index
        for FSID in sinkIDs:
            try:
                dictB[sourceID][FSID]=dictA[sourceID][FSID]
            except:
                dictB[sourceID][FSID]=10000000
    DataFrame=pd.DataFrame(dictB)
    del dictA,dictB
    return DataFrame
#%%
def fillValues(GeoDataFrame, Graph, coords, Source, weight, name="F"):
    '''
    calculate all Source to sink paths
    '''
    import shapely as shp
    
    K_new=nx.Graph()
    K=nx.Graph()

    attributes=[]
    for attr in Graph.attr:
        if "ID" not in attr:
            attributes.append(attr)
    dicAttr={attr:{} for attr in attributes}

    for x, y, z in zip(GeoDataFrame['inputID'],GeoDataFrame['targetID'], GeoDataFrame["capacity"]):
        K_new.add_edge(x, y, weight=z)

    for source in Source.index:
        paths=nx.shortest_path(Graph, source=source, weight=weight)
        for attr in attributes:
            dicAttr[attr][source]={}
        for target in K_new[source]:
            path=paths[target]
            data=K_new[source][target]
            for i in range(len(path)-1):
                try: 
                    for key in data.keys():
                        K[path[i]][path[i+1]][key]=K[path[i]][path[i+1]][key]+data[key]
                except: K.add_edge(path[i], path[i+1], data)
                for attr in attributes:
                    if "ID" in attr:
                        continue
                    try: dicAttr[attr][source][target]=dicAttr[attr][source][target]+Graph[path[i]][path[i+1]][attr]
                    except: dicAttr[attr][source][target]=Graph[path[i]][path[i+1]][attr]

    dicList={attr:[] for attr in attributes}
    for s, t in zip(GeoDataFrame['inputID'],GeoDataFrame['targetID']):
        if name in t:
            for attr in attributes:
                dicList[attr].append(dicAttr[attr][s][t])
        else:
            for attr in attributes:
                dicList[attr].append(dicAttr[attr][t][s])        

    for attr in dicAttr.keys():
        GeoDataFrame[attr]=dicList[attr]    
        
    GeoDataFrame["beeDist"]=GeoDataFrame.length/1000
    GeoDataFrame["detour"]=GeoDataFrame["weightedDistance"]/GeoDataFrame["beeDist"]
    #
    y=np.array(K.edges())
    inputIDarr=y[:,0]
    targetIDarr=y[:,1]
    LinesIn=coords.ix[list(inputIDarr)]
    LinesOut=coords.ix[list(targetIDarr)]

    EdgesCoords=gpd.GeoDataFrame(index=K.edges())
    EdgesCoords["inputCoords"]=LinesIn.geometry.values
    EdgesCoords["outputCoords"]=LinesOut.geometry.values
    EdgesCoords["geometry"]=""

    EdgesCoords["geometry"]=[shp.geometry.LineString([values["inputCoords"], values["outputCoords"]]) for key, values in EdgesCoords.iterrows()]
    EdgesTotalLine=gpd.GeoDataFrame(EdgesCoords["geometry"])
    EdgesTotalLine["capacity"]=pd.Series(nx.get_edge_attributes(K,"weight"))*1000
    
    return GeoDataFrame, EdgesTotalLine
#%%
def truckOptimization(Graph, coords, dfSource, dfFueling, weight="weightedDistance", distribution=False, name="F"):
    '''
    calculate the truck routes
    '''
    
    test=shortest_path_calculation(Graph, dfSource, dfFueling, weight=weight, distribution=distribution)
    test2=pd.DataFrame(test.unstack(level=0)).reset_index()
    test2=test2[~test2[0].isnull()]
    test2.columns=["inputID", "targetID", weight]
    test2.index=[(value["inputID"], value["targetID"]) for key, value in test2.iterrows()]
    GraphTruck2=optiFun.PipeNetWork()
    GraphTruck2.initializeEdges(test2)
    nx.set_node_attributes(GraphTruck2, "productionMax", dfSource.H2ProdCap_kt.to_dict())
    nx.set_node_attributes(GraphTruck2, "demand", dfFueling["H2Demand_kt_F"].to_dict())
    GraphTruck2.initOptiTruck(weight=weight)
    GraphTruck2.optModel()
    prodNodes=GraphTruck2.getProductionNodes()
    test3=GraphTruck2.getEdgesAsGpd(coords, analysisType="truck")
    (test4,test5) = fillValues(test3,Graph, coords, dfSource, weight, name=name)
    return (test4, test5, prodNodes)