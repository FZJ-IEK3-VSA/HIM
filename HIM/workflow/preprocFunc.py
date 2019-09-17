# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:00:59 2017

@author: m.reuss
"""
from HIM.utils import *
from HIM import dataHandling as sFun
    #%%
#%%
def importDLMStreets(pathStreet, speed, crs, name="W"):
    '''
    importing the Streetgrid from "Digitales Landschaftsmodell" and selecting the grid with highest nodes
    '''
    Streets = sFun.import_shp(pathStreet, crs=crs, name="W")
    sFun.simple_gpd(Streets)
    StreetsPoints=sFun.rem_dupl_un(Streets, "Street")
    sFun.as_ID(Streets,GeoDataFrameListUniA=StreetsPoints)
    Streets.loc[Streets["BEZ"].isnull(), "BEZ"]=""
    Streets["speed"]=speed["urban"]
    Streets.loc[["A" in bez for bez in Streets["BEZ"]], "speed"]=speed["motorway"]
    Streets["time"]=Streets.length/1000/Streets["speed"]
    Streets["weightedDistance"]=Streets["distance"]
    Reduction=nx.Graph()
    Reduction.add_edges_from(list(zip(Streets.inputID,Streets.targetID)))
    for node in Reduction.nodes():
        G_new=[x for x in nx.node_connected_component(Reduction, node)]
        if len(G_new)>0.9*len(Reduction.nodes()): break
    StreetsPoints.index=StreetsPoints["ID"].values
    StreetsPointsNew=StreetsPoints.loc[G_new].copy()
    Streets.index=[(x[1].inputID, x[1].targetID) for x in Streets.iterrows()]
    distStreet=Streets.loc[:,["inputID", "targetID", "distance", "weightedDistance", "time"]]    
    return distStreet, StreetsPointsNew
#%%
def correlateFuelingData(Fueling, FuelingData, ConvFabMart, Centroid, maxDistPrioFS=25):
    '''
    Converting the Fueling Station DataSet of Robinius and Grüger
    '''
    ConvFabMart.index=[int(val) for val in ConvFabMart["old"]]
    ConvFabMart["newID"]=[int(val) for val in ConvFabMart["new"]]
    FuelingData.index=Fueling.index
    # Copy Fueling Data to Fueling GeoDataFrame
    Fueling=Fueling.join(FuelingData, rsuffix="-")
    # Fine
    Fueling.loc[Fueling["ABST_2BAHN"]<=maxDistPrioFS,"BAB"] = 1
    Fueling["ID_KRS_BKG"]=ConvFabMart.loc[Fueling["ID_KRS_BKG"]]["new"].values
    Centroid.index=Centroid["ID_KRS_BKG"]
    Fueling["ID_C"]=Centroid.ix[Fueling["ID_KRS_BKG"]]["ID"].values
    Centroid.index=Centroid["ID"].values
    # Add the distance to the Centroid in km
    Fueling["CentC"]=Centroid.ix[Fueling["ID_C"]].geometry.values
    Fueling["CoordsFtoC"]=list(zip(Fueling.geometry, Fueling["CentC"]))
    Fueling["EdgesFtoC"]=[LineString(ix) for ix in Fueling["CoordsFtoC"]]
    EdgesFtoC=gpd.GeoDataFrame(Fueling["EdgesFtoC"].values, columns=["geometry"], index=Fueling.index)
    Fueling["distToC"]=EdgesFtoC.length/1000
    return Fueling


#%%
def getDistanceMatrices(Cluster,
                        Source,
                        FuelingNew,
                        NGGridPoint,
                        StreetsPointsNew,
                        distStreet,
                        distGtoG,
                        weightFtoF,
                        detourFactor,
                        beeline,
                        speed,
                        clustering=False,
                        clusterSize="switch",
                        detourPipeline=1.4,
                        kNN=10,
                        kNNGas=10):
        '''
        
        Create Distance Matrices for:
            distMatTruck: Truck Transport from Source to Fueling station
            distMatTruck2: Truck Transport from Hub to Fueling Station
            distMatPipe: Pipeline Transport from Source to Fueling station via hub
            
        '''
        kNN=max(min(len(Cluster)-1, kNN),1)
        #Transmission grid
        distCtoC=sFun.selfDistMatrix(Cluster, weight=detourPipeline, kNN=kNN)
        distStoC=sFun.distMatrix(Source, Cluster, weight=detourPipeline, kNN=kNN)
        distStoS=sFun.selfDistMatrix(Source, weight=detourPipeline, kNN=min(len(Source.index)-1, kNN))
        
        distCtoG=sFun.distMatrix(Cluster, NGGridPoint, weight=detourPipeline, kNN=kNNGas)
        distStoG=sFun.distMatrix(Source, NGGridPoint, weight=detourPipeline, kNN=kNNGas)
        #%%
        listInID=[]
        listTarID=[]
        listDistance=[]
        for key, values in Cluster.iterrows():
            if len(FuelingNew[FuelingNew["ClusterID"]==key]["ID"])>0:
                listF=list(FuelingNew[FuelingNew["ClusterID"]==key]["ID"])
                listCoords=list(sFun.point_array(FuelingNew[FuelingNew["ClusterID"]==key]))
                (inID, outID, distance) = sFun.selfDistMatrixFueling(listF, listCoords)
                listInID.extend(inID)
                listTarID.extend(outID)
                listDistance.extend(distance) 
        ## Distribution grid
        #from fueling station to fueling station
        distFtoF=pd.DataFrame([listInID,
                               listTarID,
                               listDistance],
                               index=["inputID",
                                      "targetID",                              
                                      "distance"]).T
        distFtoF=distFtoF[distFtoF.inputID != distFtoF.targetID]
        distFtoF["inputArea"]=FuelingNew.loc[distFtoF["inputID"].values,"GEBIET"].values
        distFtoF["targetArea"]=FuelingNew.loc[distFtoF["targetID"].values,"GEBIET"].values
        distFtoF["weightID"]=[str(values["inputArea"])+"to"+str(values["targetArea"]) for key, values in distFtoF.iterrows()]
        distFtoF["weightedDistance"]=weightFtoF[distFtoF["weightID"]].values*distFtoF["distance"]
        distFtoF.index=[(x[1]["inputID"], x[1]["targetID"]) for x in distFtoF.iterrows()]
        distFtoF=distFtoF.loc[:,['inputID', 'targetID', 'distance', 'weightedDistance']]
        #From Fueling station to Centroid
        if clustering:
            distFueling=FuelingNew.loc[:,["ID", "ClusterID", "distToCl", "weightedDistance"]]
        else:
            distFueling=FuelingNew.loc[:,["ID", "ID_C", "distToC", "weightedDistance"]]
        distFueling.columns=distCtoC.columns
        distFueling.index=[(x[1]["inputID"], x[1]["targetID"]) for x in distFueling.iterrows()]
        
        
        #Truck distance matrices (nearest points)
        distStoStreet=sFun.distMatrix(Source, StreetsPointsNew, weight=detourFactor, kNN=1)
        distFtoStreet=sFun.distMatrix(FuelingNew, StreetsPointsNew, weight=detourFactor, kNN=1)
        distCtoStreet=sFun.distMatrix(Cluster, StreetsPointsNew, weight=detourFactor, kNN=1)
        
        
        if clusterSize=="noHub":
            FuelingNew.intIndex=range(len(FuelingNew.index))
            distFtoG=sFun.distMatrix(FuelingNew, NGGridPoint, weight=detourPipeline, kNN=1)
            #clusterSize=len(FuelingNew)
            distMatPipeline=distGtoG.append(distStoG).append(distFtoG).append(distStoS)
        else:
            distMatPipeline=distCtoC.append(distGtoG).append(distCtoG).append(distStoG).append(distFtoF).append(distFueling).append(distStoC).append(distStoS)
    
        # In[29]:
        
        if beeline[0]:
            distFtoS=sFun.distMatrix(FuelingNew, Source, weight=detourFactor, kNN=len(Source.geometry))
            distMatTruck=distFtoS
            distMatTruck["time"]=distMatTruck["weightedDistance"]/speed["beeline"]
        else:
            distMatTruck=distStoStreet.append(distFtoStreet)
            distMatTruck["time"]=distMatTruck["weightedDistance"]/speed["urban"]
            distMatTruck=distMatTruck.append(distStreet)
        if beeline[1]:
            distMatTruck2=distFueling.copy()
            distMatTruck2["weightedDistance"]=FuelingNew["distToC"].values
            distMatTruck2["weightedDistance"]=distMatTruck2["weightedDistance"]*detourFactor
            distMatTruck2["time"]=distMatTruck2["weightedDistance"]/speed["beeline"]
        else:
            distMatTruck2=distCtoStreet.append(distFtoStreet)
            distMatTruck2["time"]=distMatTruck2["weightedDistance"]/speed["urban"]
            distMatTruck2=distMatTruck2.append(distStreet)
        

        return distMatTruck, distMatTruck2, distMatPipeline
#%%
def getChosenStations(Fueling, Centroid, weightFtoF):
    '''
    select fueling stations based on demand per centroid
    '''
    if "BAB" not in Fueling.columns:
        Fueling["BAB"]=0
    try: FuelingSorted=Fueling.sort_values(by=["BAB","GEBIET","ID_TS"],ascending=[False,False, True])
    except: FuelingSorted=Fueling.copy()
    FuelingNew=gpd.GeoDataFrame(columns=Fueling.columns, crs=Centroid.crs)
    listInID=[]
    listTarID=[]
    listDistance=[]
    listFull=[]

    FuelingSorted["coords"]=sFun.point_array(FuelingSorted)
#    for key, values in Centroid.iterrows():
#        if values["minFS"]==0:
#            continue
#        listF=list(FuelingSorted[FuelingSorted["ID_C"]==key].head(int(values["minFS"]))["ID"])
#        listCoords=list(FuelingSorted[FuelingSorted["ID_C"]==key].head(int(values["minFS"]))["coords"])
#        listFull.extend(listF)
#        (inID, outID, distance) = sFun.selfDistMatrixFueling(listF, listCoords)
#        listInID.extend(inID)
#        listTarID.extend(outID)
#        listDistance.extend(distance)
    for key, values in Centroid.iterrows():
        if values["minFS"]==0:
            continue
        notEnough=True
        while notEnough:
            if len(FuelingSorted[FuelingSorted["ID_C"]==key].index)>values["minFS"]:
                listF=list(FuelingSorted[FuelingSorted["ID_C"]==key].head(int(values["minFS"]))["ID"])
                listCoords=list(FuelingSorted[FuelingSorted["ID_C"]==key].head(int(values["minFS"]))["coords"])
                listFull.extend(listF)
                notEnough=False
            else:
                FuelingSorted=FuelingSorted.append(FuelingSorted[FuelingSorted["ID_C"]==key], ignore_index=True)   
        (inID, outID, distance) = sFun.selfDistMatrixFueling(listF, listCoords)
        listInID.extend(inID)
        listTarID.extend(outID)
        listDistance.extend(distance)

    FuelingNew=Fueling.loc[listFull].copy()
    FuelingNew["intIndex"]=range(len(FuelingNew.index))
    FuelingNew.index=["F"+str(x) for x in FuelingNew["intIndex"]]
    FuelingNew["ID"]=FuelingNew.index
    FuelingNew["H2Demand_kt_F"]=[Centroid.loc[ID_C, "H2Demand_kt_F"] for ID_C in FuelingNew["ID_C"]]
    FuelingNew["H2Demand_kg_d_F"]=FuelingNew["H2Demand_kt_F"]*1e6/365
    FuelingNew["CentC"]=Centroid.ix[FuelingNew["ID_C"]].geometry.values
    FuelingNew["CoordsFtoC"]=list(zip(FuelingNew.geometry, FuelingNew["CentC"]))
    FuelingNew["EdgesFtoC"]=[LineString(ix) for ix in FuelingNew["CoordsFtoC"]]
    EdgesFtoC=gpd.GeoDataFrame(FuelingNew["EdgesFtoC"].values, columns=["geometry"], index=FuelingNew.index)
    FuelingNew["distToC"]=EdgesFtoC.length/1000
    FuelingNew["areaID"]=[str(int(ix))+"to1" for ix in FuelingNew["GEBIET"]]
    FuelingNew["weightedDistance"]=[weightFtoF[values["areaID"]]*values["distToC"] for key, values in FuelingNew.iterrows()]
    return FuelingNew

#%%
def getGasFrance(NGGridLine, multiple=True):
    NGGridLine["coordsIn"]=NGGridLine.geometry.map(lambda geom: (np.round(geom.coords[0][0]), np.round(geom.coords[0][1])))
    NGGridLine["coordsOut"]=NGGridLine.geometry.map(lambda geom: (np.round(geom.coords[-1][0]), np.round(geom.coords[-1][1])))
    NGGridLine["distance"]=NGGridLine.length/1000
    #NGGridLine.loc[:,["geometry", "f_code"]].to_file(path + "railNG_jpn.shp")
    NGGridPoint=sFun.rem_dupl_un(NGGridLine, name="G")
    NGGridPoint.index=NGGridPoint.coords.values
    NGGridLine["inputID"]=NGGridPoint.loc[NGGridLine["coordsIn"].values, "ID"].values
    NGGridLine["targetID"]=NGGridPoint.loc[NGGridLine["coordsOut"].values, "ID"].values
    NGGridLine["tupleID"]=[(x[1]["inputID"], x[1]["targetID"]) for x in NGGridLine.iterrows()]
    NGGridPoint.index=NGGridPoint["ID"].values
    
    Reduction=nx.Graph()
    Reduction.add_edges_from(list(zip(NGGridLine.inputID,NGGridLine.targetID)))
    for subgraph in nx.connected_component_subgraphs(Reduction.copy()):
        if len(subgraph.nodes())<0.02*len(Reduction.nodes()):
            Reduction.remove_nodes_from(subgraph)
    NGGridPoint.index=NGGridPoint["ID"].values
    NGGridPoint=NGGridPoint.loc[Reduction.nodes()].copy()
    dictStreets={}
    NewSet=set(Reduction.nodes())
    for key, values in NGGridLine.iterrows():
        if values["inputID"] in NewSet or values["targetID"] in NewSet:
            dictStreets[key]=True
        else:
            dictStreets[key]=False
    NGGridLine["keep"]=pd.Series(dictStreets)
    NGGridLine=NGGridLine[NGGridLine["keep"]]
    NGGridLine.index=[(x[1].inputID, x[1].targetID) for x in NGGridLine.iterrows()]
    distGtoG=NGGridLine.loc[:,["inputID", "targetID", "distance"]].copy()
    distGtoG["weightedDistance"]=distGtoG["distance"]
    distGtoG.index=NGGridLine["tupleID"].values
    if multiple:
        NGGridPointTest=NGGridPoint.reset_index(drop=True)
        NGGridPointTest["intIndex"]=NGGridPointTest.index
        NGGridPointTest.index=["G"+ str(x) for x in NGGridPointTest.intIndex]
        NGGridPointTest["ID"]=NGGridPointTest.index
        NGGridPoint=NGGridPointTest
        distGtoG=sFun.selfDistMatrix(NGGridPointTest, weight=1, kNN=10)
    return distGtoG, NGGridPoint
                             #%%
def importFrance(dataPath, crs, speed, fuelingMax_kg_d, penetration, mileage, specificDemand,
                 sourceFile="SourcesFrance.shp",
                 productionMultiplier=1.1):
    '''
    Preprocessing of all French Inputs
    '''
    #df=pd.read_csv(path.join(dataPath,"NuclearPPFrance.csv"))
    #geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    #crsS = {'init': 'epsg:4326'}
    #Source = gpd.GeoDataFrame(df, crs=crsS, geometry=geometry).to_crs(crs)
    #Source.to_file(path.join(dataPath,"SourcesFrance.shp"))
    Source=sFun.import_shp(path.join(dataPath,sourceFile), name="S", crs=crs)
#    ## Adjustments of fueling stations
#    from ast import literal_eval
#    df=pd.read_csv(path.join(dataPath,"fuelstationsFR.csv"))
#    df=df.loc[[isinstance(x, str) for x in df["latlng"]]].copy()
#    df["coords"]=[literal_eval(xy) for xy in df["latlng"]]
#    df["geometry"]=[Point(yx[1],yx[0]) for yx in df["coords"]]
#    crsF = {'init': 'epsg:4326'}
#    Fueling = gpd.GeoDataFrame(df, crs=crsF, geometry=df["geometry"]).to_crs(crs)
#    #GetGEBIETs Value
#    urbanAreas=sFun.import_shp(r"C:\Alles\Sciebo\QGIS\grump-v1-urban-ext-polygons-rev01-shp", crs=crs)
#    urbAreasFr=urbanAreas.loc[urbanAreas["Countryeng"]=="France"]
#    bigUrbAreasFr=urbAreasFr.loc[urbAreasFr["ES95POP"]>100000]
#    Fueling["GEBIET"]=1
#    for (areaID, areaValue) in urbAreasFr.iterrows():
#        Fueling.loc[Fueling.within(areaValue.geometry), "GEBIET"]=2       
#    for (areaID, areaValue) in bigUrbAreasFr.iterrows():
#        Fueling.loc[Fueling.within(areaValue.geometry), "GEBIET"]=3 
#    Fueling.loc[:,["geometry", "typeroute", "GEBIET"]].to_file(path.join(dataPath,"FuelingFrance.shp"))
    
    Fueling=sFun.import_shp(path.join(dataPath,"FuelingFrance.shp"),name="F", crs=crs)
    #gdf=gpd.read_file(path.join(dataPath,"NUTS_RG_01M_2013.shp"))
    #gdf=gdf.loc[gdf.STAT_LEVL_==2]
    #gdf=gdf.loc[["FR" in x for x in gdf.NUTS_ID]].reset_index(drop=True)
    #gdf["intIndex"]=gdf.index
    #gdf.index=["D"+str(x) for x in gdf["intIndex"]]
    #gdf["ID"]=gdf.index
    #District=gdf.copy()
    #District.crs={'init': 'epsg:4326'}
    #District.to_file(path.join(dataPath,"PolyFrance.shp"))
    District=sFun.import_shp(path.join(dataPath,"PolyFrance.shp"), name="D", crs=crs)

    Streets=sFun.import_shp(path.join(dataPath,"FRA_roadsNew.shp"), name="W", crs=crs)
    sFun.simple_gpd(Streets)
    StreetsPoints=sFun.rem_dupl_un(Streets, "Street")
    sFun.as_ID(Streets,GeoDataFrameListUniA=StreetsPoints)
    Streets["speed"]=speed["urban"]
    Streets.loc[Streets.RTT_DESCRI=="Primary Route", "speed"]=speed["motorway"]
    Streets["time"]=Streets.length/1000/Streets["speed"]
    Streets["weightedDistance"]=Streets["distance"]
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
    #StreetsNew=sFun.splitLinesOnMaxDistance(Streets).T
    #StreetsNew.loc[:,["RTT_DESCRI", "geometry", "distance"]].to_file(path+"FRA_roadsNew.shp")
    District=sFun.linePolyIntersectBoolean(Streets,
                                           District)
    District=District.loc[District["boolStreet"]==1]
    District=District.reset_index(drop=True)
    District["intIndex"]=District.index
    District.index=["D"+str(x) for x in District["intIndex"]]
    District["ID"]=District.index

    vehicleDistribution=pd.read_excel(path.join(dataPath,"vehicleStockEurope1112.xls"), header=8)
    vehicleDistribution.index=vehicleDistribution["GEO"].values
    District["nCars"]=vehicleDistribution.loc[District.NUTS_ID][2016].values
    District["name"]=vehicleDistribution.loc[District.NUTS_ID]["GEO(L)/TIME"].values
    District=District.loc[District.nCars>0]

    #Arrondissements=sFun.import_shp(path.join(dataPath,"arrondissements-20131220-5m.shp"), crs=crs, name="A")
    #populationDistribution=pd.read_excel(path.join(dataPath,"Population.xls"), header=7, sheetname="Arrondissements")
    #populationDistribution["insee_ar"]=[(values["Code département"] + str(values["Code arrondissement"])) for key, values in populationDistribution.iterrows()]
    #populationDistribution.index=populationDistribution["insee_ar"].values
    #Arrondissements["population"]=populationDistribution.loc[Arrondissements.insee_ar]["Population totale"].values
    #tic=time.time()
    #dictPref={}
    #dictDist={}
    #dictID={}
    #spatial_index=Arrondissements.sindex
    #
    #for (areaID, areaValue) in District.iterrows():
    #    possible_matches_index = list(spatial_index.intersection(areaValue["geometry"].bounds))
    #    possible_matches = Arrondissements.iloc[possible_matches_index]
    #    precise_matches = possible_matches[possible_matches.intersects(areaValue["geometry"])]
    #    Arrondissements.loc[precise_matches.index,"ID_C"]=areaID
    #    District.loc[areaID, "nArr"]=len(precise_matches.index)                                                                                    
    ##Arrondissements=Arrondissements[[isinstance(ID,str) for ID in Arrondissements["ID_C"]]]
    #Arrondissements=Arrondissements.loc[[isinstance(x, str) for x in Arrondissements["ID_C"]]].copy()
    #Arrondissements.loc[np.invert(Arrondissements["population"]>0), "population"]=0
    ##Arrondissements=Arrondissements.loc[Arrondissements["population"]>0]
    #for id_c in set(Arrondissements["ID_C"]):
    #    Arrondissements.loc[Arrondissements["ID_C"]==id_c,"pop_NUTS2"]=Arrondissements.loc[Arrondissements["ID_C"]==id_c,"population"].sum()
    #    Arrondissements.loc[Arrondissements["ID_C"]==id_c,"cars_NUTS2"]=District.loc[id_c,"nCars"]
    #Arrondissements["share"]=Arrondissements["population"]/Arrondissements["pop_NUTS2"]
    #Arrondissements["nCars"]=Arrondissements["share"]*Arrondissements["cars_NUTS2"]
    #Arrondissements=Arrondissements.reset_index(drop=True)
    #Arrondissements["intIndex"]=Arrondissements.index
    #Arrondissements.index=["A"+str(x) for x in Arrondissements["intIndex"]]
    #Arrondissements["ID"]=Arrondissements.index
    #Arrondissements.to_file(path.join(dataPath,"ArrondFrance_NEW.shp"))
    Arrondissements=sFun.import_shp(path.join(dataPath,"ArrondFrance_NEW.shp"), crs=crs, name="A")

    dictPref={}
    dictDist={}
    dictID={}
    spatial_index=Fueling.sindex

    for (areaID, areaValue) in Arrondissements.iterrows():
        possible_matches_index = list(spatial_index.intersection(areaValue["geometry"].bounds))
        possible_matches = Fueling.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(areaValue["geometry"])]
        Fueling.loc[precise_matches.index,"ID_C"]=areaID
        Arrondissements.loc[areaID, "nFuelStat"]=len(precise_matches.index)                                                                                    
    Fueling=Fueling[[isinstance(ID,str) for ID in Fueling["ID_C"]]]
    Fueling=Fueling.reset_index(drop=True)
    Fueling["intIndex"]=Fueling.index
    Fueling.index=["F"+str(x) for x in Fueling["intIndex"]]
    Fueling["ID"]=Fueling.index
    pathGasP = path.join(dataPath, "pipeFR.shp")
    #pathGasP = path.join(dataPath, "FRA_roadsNew.shp")
    NGGridLine = sFun.import_shp(pathGasP, crs=crs, name="GG")
    distGtoG, NGGridPoint = getGasFrance(NGGridLine)
    Centroid=Arrondissements.copy()
    Centroid.geometry=Centroid.centroid
    Centroid["FCEV"]=Centroid.nCars*penetration
    Centroid["FCEV"].sum()
    Centroid["H2Demand_kt"]=Centroid["FCEV"]*specificDemand*mileage*1e-6
    fuelingMax_kt_a=fuelingMax_kg_d/1e6*365
    Centroid["minFS"]=np.ceil(Centroid["H2Demand_kt"]/fuelingMax_kt_a)
    Centroid["realFS"]=Centroid["nFuelStat"]
    Centroid.loc[Centroid["realFS"]==0,"minFS"]=0
    #lowFS=Centroid[Centroid["minFS"]>Centroid["realFS"]].index
    Fueling["BAB"]=[x == "A" for x in Fueling["typeroute"]]
    Centroid["highwayFS"]=[sum(Fueling[Fueling["ID_C"]==ix]["BAB"]) for ix in Centroid.index]
    #Centroid.loc[lowFS,"minFS"] = Centroid["realFS"][lowFS].values
    Centroid["H2Demand_kt_F"]=Centroid["H2Demand_kt"]/Centroid["minFS"].astype(int)
    Centroid.loc[Centroid["realFS"]==0,"H2Demand_kt_F"]=0
        
    totalH2Demand=Centroid["H2Demand_kt"].sum()

    Source["H2ProdCap_kt"]=Source["p_nom"]/Source["p_nom"].sum()*totalH2Demand*productionMultiplier
    
    return {"Streets":Streets,
            "StreetsPointsNew":StreetsPointsNew,
            "distStreet":distStreet,
            "NGGridLine":NGGridLine,
            "NGGridPoint":NGGridPoint,
            "distGtoG":distGtoG,
            "Fueling":Fueling,
            "Centroid":Centroid,
            "Source":Source,
            "District":District,
            "totalH2Demand":totalH2Demand,
            "Arrondissements":Arrondissements}
                             