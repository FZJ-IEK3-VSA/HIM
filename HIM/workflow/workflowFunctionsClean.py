# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:14:10 2018

@author: m.reuss
"""
from HIM import dataHandling as sFun
from HIM import optiSetup as optiFun
from HIM import hscTotal
from HIM import plotFunctions as pFun
from HIM.workflow import preprocFunc as preFun    
from HIM.utils import *

#%%
def preProcStreets(Streets, speed, crs):
    sFun.simple_gpd(Streets)
    StreetsPoints=sFun.rem_dupl_un(Streets, "Street")
    sFun.as_ID(Streets,GeoDataFrameListUniA=StreetsPoints)
    Streets.index=[(x[1].inputID, x[1].targetID) for x in Streets.iterrows()]
    Streets["speed"]=[speed[x] for x in Streets["streetType"]]
    Streets["time"]=Streets.length/1000/Streets["speed"]
    Streets["weightedDistance"]=Streets["distance"]
    distStreet=Streets.loc[:,["inputID", "targetID", "distance", "weightedDistance", "time"]]       
    Reduction=nx.Graph()
    Reduction.add_edges_from(list(zip(Streets.inputID,Streets.targetID)))
    for subgraph in nx.connected_component_subgraphs(Reduction.copy()):
        if len(subgraph.nodes())<0.02*len(Reduction.nodes()):
            Reduction.remove_nodes_from(subgraph)
    StreetsPoints.index=StreetsPoints["ID"].values
    StreetsPointsNew=StreetsPoints.loc[Reduction.nodes()].copy()
    dictStreets={}
    NewSet=set(Reduction.nodes())
    for key, values in Streets.iterrows():
        if values["inputID"] in NewSet or values["targetID"] in NewSet:
            dictStreets[key]=True
        else:
            dictStreets[key]=False
    Streets["keep"]=pd.Series(dictStreets)
    Streets=Streets[Streets["keep"]]
    Streets.index=[(x[1].inputID, x[1].targetID) for x in Streets.iterrows()]
    distStreet=Streets.loc[:,["inputID", "targetID", "distance", "weightedDistance", "time"]]

    return distStreet, StreetsPointsNew
#%%
def preProcessing(dataPath, penetration, dfTable, country):
    '''
    loads the import data and precalculates the spatial demand as well as the 
    fueling station locations. In addition, creates the distance matrices for 
    the graph calculations afterwards.
    
    Basically, does everything, before the optimization of the transport models 
    start
    ###
    Inputs:
        dataPath=string:
            path where input data is located:
                --> Folder with scenarios/ImportTableTechnologies etc.
        penetration: float
        FCEV penetration,
        dfTable: pandas dataframe :
            all techno-economic as well as scenario specific assumptions, 
        country: string:
            country to investigate
    ___________________________________________________________________________
    Outputs:
        dictionary with 9 objects:
        "Source": GeoDataFrame with Points geometry:
            Important column:
                "H2ProdCap_kt": Hydrogen production capacity in kt per year
        "District": GeoDataFrame with Polygon geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "FuelingNew":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt_F": Hydrogen demand per fueling station in kt per year
        "Cluster":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "pipeCoords":GeoSeries with Points geometry:
            all coordinates for pipeline calculation
        "truckCoords":GeoSeries with Points geometry:
            all coordinates for truck calculation
        "distMatTruck": DataFrame:
            includes all distance matrices for transmission truck graph            
        "distMatTruck2":DataFrame:
            includes all distance matrices for distribution truck graph  
        "distMatPipeline":DataFrame:
            includes all distance matrices for pipeline graph  
    '''
    crs={'ellps': 'GRS80',
         'no_defs': True,
         'proj': 'utm',
         'units': 'm',
         'zone': dfTable["General"].loc["utmZone","General"]}
    dataPathCountry=path.join(dataPath, country)
    #_____________________________________Import_______________________________
    Source = sFun.import_shp(path.join(dataPathCountry, "SourceNodes"), crs=crs, name="S")
    
    District = sFun.import_shp(path.join(dataPathCountry, "AreaPolygons"), crs=crs, name="C")
    
    Fueling = sFun.import_shp(path.join(dataPathCountry, "FuelingNodes"), crs=crs, name="F")
    
    Streets=sFun.import_shp(path.join(dataPathCountry,"StreetLines"), crs=crs, name="W")
    
    NGGridPoint = sFun.import_shp(path.join(dataPathCountry,"GasNodes"), crs=crs, name="G")
    
    distGtoG=pd.read_csv(path.join(dataPathCountry,"distGas.csv"), index_col=0)
    distGtoG.index=[(values["inputID"], values["targetID"]) for key, values in distGtoG.iterrows()]
    #________________________________Loading Scenario Data____________________
    speed={"motorway":dfTable["General"].loc["truckSpeedHighway","General"],
           "urban": dfTable["General"].loc["truckSpeedRural","General"],
           "beeline": dfTable["General"].loc["truckSpeed","General"]} 
    clustering=bool(dfTable["General"].loc["clustering","General"])
    clusterSize=dfTable["General"].loc["clusterSize","General"]
    specificDemand=dfTable["General"].loc["specificDemand","General"]
    mileage=dfTable["General"].loc["mileage","General"]
    targetCapacityFS=dfTable["General"].loc["targetStationSize","General"]
    fuelingMax_kg_d=dfTable["General"].loc["utilization Station","General"]*targetCapacityFS
    detourFactorPipeline=dfTable["General"].loc["detourFactorPipeline","General"]
    detourFactorTruck=dfTable["General"].loc["detourFactorTruck","General"]     
    weightFtoF=detourFactorPipeline*pd.Series([1., 1.25, 1.25, 1.5, 1.5, 1.5, 1.75, 1.75, 2.],
                     index=["1to1","1to2","2to1","2to2","1to3","3to1","2to3","3to2","3to3"])
    #________________________________Preparing dataframes_____________________
 
    
    
    
    spatial_index=Fueling.sindex
    for (areaID, areaValue) in District.iterrows():
        possible_matches_index = list(spatial_index.intersection(areaValue["geometry"].bounds))
        possible_matches = Fueling.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(areaValue["geometry"])]
        Fueling.loc[precise_matches.index, "Name"]=areaValue["Name"]
        Fueling.loc[precise_matches.index,"ID_C"]=areaID
        District.loc[areaID, "nFuelStat"]=len(precise_matches.index) 
    Fueling["BAB"]=0
    Source["H2ProdCap_kt"]=Source["H2ProdCap_"]
    (distStreet, StreetsPointsNew) = preProcStreets(Streets, speed, crs)
    
    Centroid=District.copy()
    Centroid.geometry=Centroid.centroid    
    Centroid["FCEV"]=Centroid["Bestand"]*penetration
    Centroid["H2Demand_kt"]=Centroid["FCEV"]*specificDemand*mileage*1e-6
    fuelingMax_kt_a=fuelingMax_kg_d/1e6*365
    Centroid["minFS"]=np.ceil(Centroid["H2Demand_kt"]/fuelingMax_kt_a)
    Centroid["realFS"]=Centroid["nFuelStat"]
    Centroid["H2Demand_kt_F"]= Centroid["H2Demand_kt"]/Centroid["minFS"]
    Centroid.loc[Centroid["minFS"]==0,["H2Demand_kt_F", "H2Demand_kt"]]=0
    totalH2Demand=Centroid["H2Demand_kt"].sum()
    if country=="France":Source["H2ProdCap_kt"]=Source["p_nom"]/Source["p_nom"].sum()*totalH2Demand*1.1
    Source.loc[Source["H2ProdCap_kt"]>totalH2Demand, "H2ProdCap_kt"]=totalH2Demand
    District["H2Demand_kt"]=Centroid["H2Demand_kt"]

    totalH2Capacity=sum(Source["H2ProdCap_kt"])
    if totalH2Demand>totalH2Capacity:
        print("Production capacity not sufficient for Demand!")    

    # ## Calculate minimum numbers of fueling stations
    try: 
        fuelingMax_kt_a=totalH2Demand/targetFS
        fuelingMax_kg_d=fuelingMax_kt_a*1e6/365
    except:
        fuelingMax_kt_a=fuelingMax_kg_d/1e6*365
    
    Centroid["minFS"]=np.ceil(Centroid["H2Demand_kt"]/fuelingMax_kt_a)
    Centroid.loc[Centroid["realFS"]==0,"minFS"]=0
    
    Centroid["H2Demand_kt_F"]= Centroid["H2Demand_kt"]/Centroid["minFS"]
    Centroid.loc[Centroid["minFS"]==0,"H2Demand_kt_F"]
    Centroid.loc[Centroid["realFS"]==0,"H2Demand_kt"]=0
    #Fueling Station Selection
    
    FuelingNew=preFun.getChosenStations(Fueling=Fueling, Centroid=Centroid, weightFtoF=weightFtoF) 
    #____________________________Clustering____________________________________
    
    if clustering:
        if country=="Japan":
                distFtoStreet=sFun.distMatrix(FuelingNew, StreetsPointsNew, weight=detourFactorTruck, kNN=1)
                ClusterGraph=optiFun.PipeNetWork()
                ClusterGraph.initializeEdges(distFtoStreet.append(distStreet))
                Cluster, FuelingNew=sFun.createCluster(FuelingNew, clusterSize, ClusterGraph)
                FuelingNew["weightedDistance"]=[weightFtoF[values["areaID"]]*values["distToCl"] for key, values in FuelingNew.iterrows()]
        else:
            Cluster=sFun.createCluster(FuelingNew, clusterSize)
            Cluster.crs=Centroid.crs
            FuelingNew["weightedDistance"]=[weightFtoF[values["areaID"]]*values["distToCl"] for key, values in FuelingNew.iterrows()]
    else:
        Cluster=Centroid.copy()
        FuelingNew["distToCl"]=FuelingNew["distToC"]
        FuelingNew["ClusterID"]=FuelingNew["ID_C"]
    #______________________________Distance Matrices__________________________
    pipeCoords=Cluster.geometry.append(Source.geometry).append(NGGridPoint.geometry).append(FuelingNew.geometry)
    truckCoords=Source.geometry.append(FuelingNew.geometry).append(StreetsPointsNew.geometry).append(Cluster.geometry)
   
    distMatTruck, distMatTruck2, distMatPipeline=preFun.getDistanceMatrices(Cluster,
                                                                            Source,
                                                                            FuelingNew,
                                                                            NGGridPoint,
                                                                            StreetsPointsNew,
                                                                            distStreet,
                                                                            distGtoG,
                                                                            weightFtoF,
                                                                            detourFactorTruck,
                                                                            speed=speed,
                                                                            clustering=clustering,
                                                                            clusterSize=clusterSize,
                                                                            beeline=[False, False]) 
  
    return {"Source":Source,
            "District":District,
            "FuelingNew":FuelingNew,
            "Cluster":Cluster,
            "pipeCoords":pipeCoords,
            "truckCoords":truckCoords,
            "distMatTruck":distMatTruck,
            "distMatTruck2":distMatTruck2,
            "distMatPipeline":distMatPipeline}
#%%
def calcTransportSystem(Source,
                        FuelingNew,
                        Cluster,
                        truckCoords,
                        pipeCoords,
                        distMatTruck,
                        distMatTruck2,
                        distMatPipeline,
                        pathResults,
                        beeline=[False, False],
                        weight="time",
                        ):
    '''
    calculates the transport models:
        Pipeline Transmission and distribution
        Truck Transmission
        Truck Distribution
    Inputs:
        "Source": GeoDataFrame with Points geometry:
            Important column:
                "H2ProdCap_kt": Hydrogen production capacity in kt per year
        "District": GeoDataFrame with Polygon geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "FuelingNew":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt_F": Hydrogen demand per fueling station in kt per year
        "Cluster":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "pipeCoords":GeoSeries with Points geometry:
            all coordinates for pipeline calculation
        "truckCoords":GeoSeries with Points geometry:
            all coordinates for truck calculation
        "distMatTruck": DataFrame:
            includes all distance matrices for transmission truck graph            
        "distMatTruck2":DataFrame:
            includes all distance matrices for distribution truck graph  
        "distMatPipeline":DataFrame:
            includes all distance matrices for pipeline graph  
    ___________________________________________________________________________
    Outputs: Results from the transport models:
        resultsEdgesTruck: GeoDataFrame with LineStrings:
            all resulting Edges of the Truck transmission calculation
            Important columns:
                "time": travelled time of each edge
                "weightedDistance": distance of each edge
                "edge": describes if truck section is an "Endpoint" (sink or source)
        resultsEdgesTruck2: GeoDataFrame with LineStrings:
            all resulting Edges of the Truck distribution calculation
            Important columns:
                "time": travelled time of each edge
                "weightedDistance": distance of each edge
                "edge": describes if truck section is an "Endpoint" (sink or source)
        resultsEdgesPipeline: GeoDataFrame with LineStrings:
            all resulting Edges of the Truck calculation
            Important columns:
                "weightedDistance": distance of each edge
                "lineCost": total Cost of each edge
                "diameter": diameter of pipeline section
        
    '''
    #PipelineCalculation
    # ## Import to NetworkX for minimum spanning tree
    GraphPipeline=optiFun.PipeNetWork()
    GraphPipeline.initializeEdges(distMatPipeline)
    
    nx.set_node_attributes(GraphPipeline, "productionMax", Source.H2ProdCap_kt.to_dict())
    nx.set_node_attributes(GraphPipeline, "demand", FuelingNew["H2Demand_kt_F"].to_dict())
    GraphPipeline.useMinSpanTree(weight="weightedDistance")
    #Test
    #GraphPipeline.reduceNetworkSize()
    # init optimization
    GraphPipeline.initOpti(linear=True)
    #Optimization
    GraphPipeline.optModel(logPath=pathResults, tee=False)
    #Extract results
    productionPipeline=GraphPipeline.getProductionNodes()
    resultsEdgesPipeline=GraphPipeline.getEdgesAsGpd(pipeCoords, "pipeline", costCalc="pressureDrop", logPath=pathResults, tee=False)
    Source["pipe_kt_a"]=productionPipeline
    Source["pipe_kg_d"]=Source["pipe_kt_a"]*1e6/365    
    #_____________________________________________________________________
    #Initializing Graph
    GraphTruck=optiFun.PipeNetWork()
    GraphTruck.initializeEdges(distMatTruck)
    
    nx.set_node_attributes(GraphTruck, "productionMax", Source.H2ProdCap_kt.to_dict())
    nx.set_node_attributes(GraphTruck, "demand", FuelingNew["H2Demand_kt_F"].to_dict())
    
    GraphTruck.reduceNetworkSize()
    #Initializing the optimization
    GraphTruck.initOptiTruck(weight=weight)
    
    GraphTruck.optModel(logPath=pathResults, tee=False)
   
    resultsTruckNodes=GraphTruck.getProductionNodes()
    resultsEdgesTruck=GraphTruck.getEdgesAsGpd(truckCoords, "truck")
     #______________________________________________________________________
    Source["truck_kt_a"]=resultsTruckNodes
    Source["truck_kg_d"]=Source["truck_kt_a"]*1e6/365
    resultsEdgesTruck["edge"]=[not ("Street" in x[0] and "Street" in x[1]) for x in resultsEdgesTruck.index]
    
    if not beeline[1]:#Optimization of Distribution trucks
        GraphTruck2=optiFun.PipeNetWork()
        GraphTruck2.initializeEdges(distMatTruck2)
    
        nx.set_node_attributes(GraphTruck2, "productionMax", Cluster.H2Demand_kt.to_dict())
        nx.set_node_attributes(GraphTruck2, "demand", FuelingNew["H2Demand_kt_F"].to_dict())
    
        GraphTruck2.reduceNetworkSize()
    
        GraphTruck2.initOptiTruck(weight=weight)
        
        GraphTruck2.optModel(logPath=pathResults, tee=False)
    
        resultsEdgesTruck2=GraphTruck2.getEdgesAsGpd(truckCoords, "truck")
        resultsEdgesTruck2["H2Demand_kg_d_F"]=resultsEdgesTruck2["capacity"]*1e6/365
    else:
        resultsEdgesTruck2=FuelingNew.loc[:,["distToC","H2Demand_kt_F", "H2Demand_kg_d_F", "EdgesFtoC"]].copy()
        resultsEdgesTruck2.geometry=resultsEdgesTruck2["EdgesFtoC"]
        resultsEdgesTruck2["weightedDistance"]=resultsEdgesTruck2["distToC"]*detourFactorTruck
        resultsEdgesTruck2["time"]=resultsEdgesTruck2["weightedDistance"]/speed["beeline"]
    #Extract results
    resultsEdgesTruck2["edge"]=[not ("Street" in x[0] and "Street" in x[1]) for x in resultsEdgesTruck2.index]
    
    return resultsEdgesTruck, resultsEdgesTruck2, resultsEdgesPipeline
# In[25]:
def calcSpatialHSC(resultsEdgesTruck,
                   resultsEdgesTruck2,
                   resultsEdgesPipeline,
                   hscPathways,
                   Cluster,
                   FuelingNew,
                   Source,
                   District,
                   dfTable,
                   pathResults=False,
                   beeline=[False, False]):
    '''
    calculates the Hydrogen supply chain model based on different pathways
    Inputs:
        "resultsEdgesTruck": GeoDataFrame with LineStrings:
            all resulting Edges of the Truck transmission calculation
            Important columns:
                "time": travelled time of each edge
                "weightedDistance": distance of each edge
                "edge": describes if truck section is an "Endpoint" (sink or source)
        "resultsEdgesTruck2": GeoDataFrame with LineStrings:
            all resulting Edges of the Truck distribution calculation
            Important columns:
                "time": travelled time of each edge
                "weightedDistance": distance of each edge
                "edge": describes if truck section is an "Endpoint" (sink or source)
        "resultsEdgesPipeline": GeoDataFrame with LineStrings:
            all resulting Edges of the Truck calculation
            Important columns:
                "weightedDistance": distance of each edge
                "lineCost": total Cost of each edge
                "diameter": diameter of pipeline section
        "hscPathways": dictionary:
            definition of supply chain pathways --> Which techs to use
        "Source": GeoDataFrame with Points geometry:
            Important column:
                "H2ProdCap_kt": Hydrogen production capacity in kt per year
        "District": GeoDataFrame with Polygon geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "FuelingNew":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt_F": Hydrogen demand per fueling station in kt per year
        "Cluster":GeoDataFrame with Points geometry:
            Important column:
                "H2Demand_kt": Hydrogen demand in kt per year
        "dfTable": pandas dataframe :
            all techno-economic as well as scenario specific assumptions, 

    Output:
        Results: dictionary:
            Collection of Supply Chain Classes of each pathway
    '''
    Results={}
    pipelineDistance=[resultsEdgesPipeline[resultsEdgesPipeline["distribution"]==False],
                      resultsEdgesPipeline[resultsEdgesPipeline["distribution"]]]
    truckDistance=[resultsEdgesTruck,
                   resultsEdgesTruck2]
    sourceDf={"pipeline":Source["pipe_kg_d"],
              "truck":Source["truck_kg_d"]}
    if pathResults:
        pathData=os.path.join(pathResults, "data")
        os.makedirs(pathData)
    i=0
    for hscPathwayType in sorted(hscPathways.keys()):
        listCapacities=[sourceDf[hscPathwayType],
                        sourceDf[hscPathwayType],
                        sourceDf[hscPathwayType],
                        sourceDf[hscPathwayType],
                        sourceDf[hscPathwayType],
                        sourceDf[hscPathwayType],
                        resultsEdgesTruck["capacity"].values*1e6/365,
                        Cluster["H2Demand_kt"]*1e6/365,
                        resultsEdgesTruck2["H2Demand_kg_d_F"],
                        FuelingNew["H2Demand_kg_d_F"]]
        for listHSC in hscPathways[hscPathwayType]:
            cumCost=0
    
            Results[i]=hscTotal.HSC(listHSC,
                                     dfTable,
                                     listCapacities,
                                     FuelingNew["H2Demand_kt_F"].sum()*1e6,
                                     truckDistance=truckDistance,
                                     pipelineDistance=pipelineDistance,
                                     targetCars=0,
                                     beeline=beeline)
    
            Results[i].calcHSC(cumCost=cumCost)
            if pathData:

                Results[i].saveHSC(pathData, i)
            i+=1
    FuelingNew["TOTEXPipe"]=Results[1].hscClasses["Station"].TOTEX
    if pathResults:
        resultsEdgesPipeline.crs=Cluster.crs
        resultsEdgesTruck2.crs=Cluster.crs
        resultsEdgesTruck.crs=Cluster.crs
        pathSHP=path.join(pathResults, "shapefiles")
        os.makedirs(pathSHP)
        testBoolColumns(resultsEdgesTruck).to_file(path.join(pathSHP,"TrucksRoutingTransmission.shp"))
        testBoolColumns(resultsEdgesTruck2).to_file(path.join(pathSHP,"TrucksRoutingDistribution.shp"))
        pipeDistribution=resultsEdgesPipeline.loc[resultsEdgesPipeline.distribution]
        pipeTransmission=resultsEdgesPipeline.loc[resultsEdgesPipeline.distribution==False]
        testBoolColumns(pipeDistribution).to_file(path.join(pathSHP,"PipeDistribution.shp"))
        testBoolColumns(pipeTransmission).to_file(path.join(pathSHP,"PipeTransmission.shp"))
        testBoolColumns(Source).to_file(path.join(pathSHP,"Source.shp"))
        testBoolColumns(FuelingNew).to_file(path.join(pathSHP,"FuelingStation.shp"))
        testBoolColumns(Cluster).to_file(path.join(pathSHP,"Hubs.shp"))
        
        testBoolColumns(District).to_file(path.join(pathSHP,"Area.shp"))
    
    return Results