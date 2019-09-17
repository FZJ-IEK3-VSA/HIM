# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:14:40 2017

@author: m.reuss
"""
from HIM.utils import *
from scipy import spatial as sp
import shapely as shp
from scipy import sqrt
from shapely.geometry import LineString

#%% import Shapefile as Geopandas dataFrame and change into a common crs
def import_shp(path, crs={'init' :'epsg:4326'}, name=""):
    '''
    input: path --> file source
    crs: coordinate reference system: default WGS84
    name: if you want to have unique indices, you should put there a name (e.g. G)
    -------------------------------
    This function imports a shapefile and gives a GeoDataFrame (geopandas).
    This dataFrame has a unique id as .index
    
    '''
    gpdDataFrame=gpd.read_file(path)
    gpdDataFrame=checkCorrupted(gpdDataFrame)
    gpdDataFrame.reset_index(drop=True)
    gpdDataFrame=gpdDataFrame.to_crs(crs)
    gpdDataFrame["intIndex"]=gpdDataFrame.index.values
    gpdDataFrame.index=[name+str(id1) for id1 in gpdDataFrame.index.values]
    gpdDataFrame["ID"]=gpdDataFrame.index.values
    return gpdDataFrame
#%%
def checkCorrupted(GeoDataFrame):
    NewDataFrame=GeoDataFrame.loc[[x is not None for x in GeoDataFrame.geometry]]
    return NewDataFrame

#%%
def point_array(GeoDataFrame):
    '''
    create Numpy array from GeoDataFrame of Points!!!
    input: GeoSeries of points
    '''
    if GeoDataFrame.geometry.type.all()=="Point":
        x=GeoDataFrame.geometry.map(lambda p:p.x).values
        y=GeoDataFrame.geometry.map(lambda p:p.y).values
        return list(zip(x,y))
    else:
        "GeoDataFrame does not contains Points: point_array is not working!"
        return

#%% Grabs a list of an attribute from NetworkX
def NXtoList(NXGraph, attribute):
    '''
    input:
        NXGraph: NX.Graph()
        attribute: name of the attribute as string
    -------------------
    returns the attributes of a NetworkX Graph as a list
    '''
    dicAttr=nx.get_edge_attributes(NXGraph,attribute)
    return [dicAttr[x] for x in dicAttr], dicAttr


#%%
def distMatrix(gpdIn, gpdOut, weight=1., kNN=10):
    '''
    Build distance Matrix for two geopandas DataFrames
    gpdIn: geopandas dataframe for start
    gpdOut: geopandas dataframe for target
    kNN: number of nearest neighbours
    weight: weighting factor for detouring
    '''
    if len(gpdOut)<kNN:
        kNN=len(gpdOut)
    CoordsOut=point_array(gpdOut)
    CoordsIn=point_array(gpdIn)
    tree=sp.KDTree(CoordsOut,leafsize=3)
    treeDist, treeLoc= tree.query(CoordsIn, k=kNN)
    #idx=(gpdIn.intIndex.values+np.zeros((kNN,1),dtype=np.int)).T.ravel()
    idx=(np.array(range(len(gpdIn.index)))+np.zeros((kNN,1),dtype=np.int)).T.ravel()
    inID=np.array([gpdIn.ID[id1] for id1 in idx])
    outID=np.array([gpdOut.ID[id2] for id2 in treeLoc.ravel()])
    index=[(start, target) for (start, target) in zip(inID, outID)]
    weightedLength=weight*treeDist.ravel()/1000
    distMat=pd.DataFrame([inID,
                          outID,
                          treeDist.ravel()/1000,
                          weightedLength],
                          index=["inputID",
                                "targetID",
                                "distance",
                                "weightedDistance"],
                          columns=index).T
    return distMat

#%%
def selfDistMatrix(gpdDF, weight=1, kNN=1):
    '''
    Build distance Matrix between all coordinates in the dataframe
    gpdDF: geopandas dataframe
    kNN: number of nearest neighbours
    weight: weighting factor for detouring
    '''
    if len(gpdDF)-1<kNN:
        kNN=len(gpdDF)-1
    Coords=point_array(gpdDF)
    tree = sp.KDTree(Coords, leafsize=10)
    treeDist, treeLoc = tree.query(Coords, k=kNN+1)
    idx = (gpdDF.intIndex.values+np.zeros((kNN+1,1),dtype=np.int)).T.ravel()    
    inID=np.array([gpdDF.ID[id1] for id1 in idx])
    tarID=np.array([gpdDF.ID[id2] for id2 in treeLoc.ravel()])
    index=[(start, target) for (start, target) in zip(inID, tarID)]
    weightedLength=weight*treeDist.ravel()/1000
    distMat = pd.DataFrame([inID,
                            tarID,
                            treeDist.ravel()/1000,
                            weightedLength],
                            index=["inputID",
                                  "targetID",                              
                                  "distance",
                                  "weightedDistance"],
                            columns=index).T
    distMat=distMat[distMat.inputID != distMat.targetID]

    return distMat


    #%%
def selfDistMatrixFueling(listF,listC, kNNmax=10):
    '''
    Build distance Matrix for fueling stations inside one cluster
    Coords: Coordinates as list of tuples
    kNNmax: number of nearest neighbours
    '''
    kNN=max(len(listF)-1,1)
    if kNN>kNNmax:
        kNN=kNNmax
    if kNN>0:
        Coords=listC
        tree = sp.KDTree(Coords, leafsize=10)
        treeDist, treeLoc = tree.query(Coords, k=kNN)
        idx = (range(len(listF))+np.zeros((kNN,1),dtype=np.int)).T.ravel() 
        inID=np.array([listF[id1] for id1 in idx])
        tarID=np.array([listF[id2] for id2 in treeLoc.ravel()])
    
        return (inID, tarID, treeDist.ravel()/1000)


#%%
def getDiameterSquare(massflow,
                      H2Density=5.7,
                      vPipeTrans=15):
    '''
    get m² from massflow with density and pipeline velocity
    massflow: kt per year
    H2Density in kg/m³
    output: diameter in m²
    '''
    
    ktPerYear_to_kgPerS=1e6/3600/365/24
    d2=massflow*ktPerYear_to_kgPerS*4/(H2Density*vPipeTrans*np.pi)
    return d2

#%%
def getSpecCost(massflow,
                f_grid=1,                
                H2Density=5.7,
                vPipeTrans=15,
                source="Krieg",
                base="diameter",
                diameter=None,
                **kwargs):
    '''
    massflow: massflow in kt per year
    f_grid: Additional factor for weighting results (just for dijkstra algorithm)
    H2Density: Density of hydrogen
    vPipeTrans: maximum velocity of hydrogen inside the pipeline
    
    Output: specific pipeline invest in Million €

    '''
    if diameter==None:
        diameter=np.sqrt(getDiameterSquare(massflow, H2Density, vPipeTrans))*1000
    if base=="diameter":
        A=2.2e-3
        B=0.86
        C=247.5
        specCost=(A*diameter**2+B*diameter+C)        
    elif base=="throughput":     
        A=474.77
        B=1.3695

        specCost=A*f_grid+B*massflow    
    return specCost*1e-6

#%%
def extractAndCalc(fullDF, minCapacity=0, zeroes=False):
    '''
    standard operations for output
    input: full DataFrame
    minCapacuty= minimum relevant capacity for pipeline design
    
    '''
    if zeroes: x=-1
    else: x=0
    EdgesDist=fullDF[fullDF["capacity"]>x].copy()
    EdgesDist.loc[EdgesDist["capacity"]<minCapacity, "capacity"]=minCapacity  
    EdgesDist["diameter"]=sqrt(getDiameterSquare(EdgesDist["capacity"].values))*1000
    EdgesDist["lineCostSpec"]=getSpecCost(EdgesDist["capacity"].values)     
    EdgesDist["lineCost"]=getSpecCost(EdgesDist["capacity"].values, source="Krieg", base="diameter")*EdgesDist.length.values
    EdgesDist["distance"]=EdgesDist.length.values/1000
    return EdgesDist

#%%
def getGpdFromNXEdges(NXGraph, coordSeries, minCapacity=0, zeroes=True):
    '''
    input:
        NX Graph --> Graph to implement
        coordSeries: Coordinates of all potential Nodes
    return:
        EdgesDist - geopandas Dataframe with extracted values from networkx graph
    '''
    y=np.array(NXGraph.edges())
    (inputIDarr, targetIDarr)=(y[:,0], y[:,1])
    LinesIn=coordSeries.loc[list(inputIDarr)].geometry.values
    LinesOut=coordSeries.loc[list(targetIDarr)].geometry.values
    EdgeCoords=gpd.GeoDataFrame(index=NXGraph.edges())
    EdgeRes=gpd.GeoDataFrame(index=NXGraph.edges())
    EdgeCoords["inputCoords"]=LinesIn
    EdgeCoords["outputCoords"]=LinesOut
    EdgeRes["geometry"]=""
    for key in EdgeCoords.index:
        EdgeRes.loc[key,"geometry"]=shp.geometry.LineString([EdgeCoords["inputCoords"][key], EdgeCoords["outputCoords"][key]]) 
    
    dicCap=nx.get_edge_attributes(NXGraph, "capacity")
    pdCap=pd.DataFrame.from_dict(dicCap, orient="index")
    EdgeRes["capacity"]=pdCap[0]
    
    EdgesDist=extractAndCalc(EdgeRes, minCapacity=minCapacity, zeroes=zeroes)
    return EdgesDist
#%%
def getGpdCapaFromPyomo(pyomoVariable, coordSeries, minCapacity=0, analysisType="pipeline"):
    '''
    input:
        pyomoVariable --> Variable from which to extract the values
        coordSeries: Coordinates of all potential Nodes
    return:
        EdgesDist - geopandas Dataframe with extracted values from networkx graph
    '''
    dicEdges=pyomoVariable.get_values()
    dicEdges={k:v for (k,v) in dicEdges.items() if v > 0}
    EdgesTotal = gpd.GeoDataFrame([(k[0], k[1], v) for (k,v) in dicEdges.items()],
                                   index=[k for k in dicEdges.keys()],
                                   columns=["inputID","targetID", "capacity"])

    LinesIn=coordSeries.ix[EdgesTotal["inputID"].values].geometry.values
    LinesOut=coordSeries.ix[EdgesTotal["targetID"].values].geometry.values
    EdgeCoords=gpd.GeoDataFrame(index=EdgesTotal.index)
    EdgeRes=gpd.GeoDataFrame(index=EdgesTotal.index)
    EdgeRes["capacity"]=EdgesTotal["capacity"]
    EdgeCoords["inputCoords"]=LinesIn
    EdgeCoords["outputCoords"]=LinesOut
    EdgeRes["geometry"]=""
    for key in EdgeCoords.index:
        EdgeRes.loc[key,"geometry"]=shp.geometry.LineString([EdgeCoords["inputCoords"][key], EdgeCoords["outputCoords"][key]])
    if analysisType=="pipeline":    
        EdgesDist=extractAndCalc(EdgeRes, minCapacity=minCapacity)
    elif analysisType=="truck":
        EdgesDist=EdgeRes[EdgeRes["capacity"]>0].copy()
        EdgesDist["distance"]=EdgesDist.length.values/1000
    return EdgesDist

#%%
def getGpdFromPyomoNodes(pyomoVariable, name):
    '''
    input:
        pyomoVariable --> Variable from whcih to extract the values
        coordSeries: Coordinates of all potential Nodes
    '''
    NodesTotal=gpd.GeoDataFrame([(v[1].value) for v in pyomoVariable.iteritems()],
                                 index=[(v[0]) for v in pyomoVariable.iteritems()],
                                 columns=[name])
    
    return NodesTotal


#%%Master student Paris Dimos work!!!

def rem_dupl_un(GeoDataFrame, name="G"):

    '''
    Must first implement simple_gpd
    input: GeoDataFrame 
    output: GeoDataframe with unique Points and ID's
    Need it like that because later I will have issues with distMatrix
    Re-run after the as_ID!!!
    '''
    GeoDataFrameListIn=(list(GeoDataFrame.coordsIn))
    GeoDataFrameListOut=(list(GeoDataFrame.coordsOut))
    num = min(len(GeoDataFrameListIn), len(GeoDataFrameListOut))
    GeoDataFrameListUni = [None]*(num*2)
    GeoDataFrameListUni[::2] = GeoDataFrameListIn[:num]
    GeoDataFrameListUni[1::2] = GeoDataFrameListOut[:num]
    GeoDataFrameListUni.extend(GeoDataFrameListIn[num:])
    GeoDataFrameListUni.extend(GeoDataFrameListOut[num:])
    seen={}
    GeoDataFrameListUni1 = [seen.setdefault(x,x) for x in GeoDataFrameListUni if x not in seen]
    from shapely.geometry import Point
    geometry=[Point(xy) for xy in GeoDataFrameListUni1]
    GeoDataFrameListUniA=gpd.GeoDataFrame()
    GeoDataFrameListUniA['geometry']=geometry
    GeoDataFrameListUniA['intIndex']=range(len(GeoDataFrameListUni1))
    GeoDataFrameListUniA['coords']=point_array(GeoDataFrameListUniA)
    GeoDataFrameListUniA['ID']=[name+str(x) for x in range(len(GeoDataFrameListUni1))]
    GeoDataFrameListUniA.crs=GeoDataFrame.crs
    del GeoDataFrameListUni1, GeoDataFrameListUni
    return GeoDataFrameListUniA

#%%
def as_ID(GeoDataFrame, GeoDataFrameListUniA):
    '''
    Assigns a unique ID to all coordinates of the DataFrame
    Input: GeoDataFrame, GeoDataFrame from rem_dupl_un function 
    Output: GeoDataframe with unique "StrID" and "EndID"
    '''
    GeoDataFrameListUniA.index=GeoDataFrameListUniA['coords'].values
    GeoDataFrame['inputID']=GeoDataFrameListUniA.loc[GeoDataFrame['coordsIn'].values]['ID'].values
    GeoDataFrame['targetID']=GeoDataFrameListUniA.loc[GeoDataFrame['coordsOut'].values]['ID'].values
    #return GeoDataFrame
#%%
def simple_gpd(GeoDataFrame):
    '''
    Creates coords, coordsIn, coordsOut simple_gpd
    Input: GeoDataFrame 
    Output: GeoDataframe with first and last coord at Linestring geometry
    '''
    GeoDataFrame['distance']=GeoDataFrame.length/1000
    GeoDataFrame['coords'] = [ix.coords[::len(ix.coords)-1] for ix in GeoDataFrame.geometry]
    GeoDataFrame['coordsIn'] = [(np.round(x[0][0],3), np.round(x[0][1],3)) for x in GeoDataFrame['coords']]
    GeoDataFrame['coordsOut'] = [(np.round(x[1][0],3), np.round(x[1][1],3)) for x in GeoDataFrame['coords']]

#%%
def splitLinesOnMaxDistance(GeoDataLineString, lMax=1000):
    '''
    split a lots of lines into smaller ones based on the length of the line
    '''
    j=0
    attrDict={}
    
    for key, values in GeoDataLineString.iterrows():
        geom=values["geometry"]
        if geom.length>lMax:
            addPoints=np.ceil(geom.length/lMax)
            start=geom.coords[0]
            for i in range(int(addPoints)+1):
                attrDict[j]={}
                if i>addPoints:
                    end=geom.coords[-1]
                else:
                    newPoint=geom.interpolate(geom.length/(addPoints+1)*(i+1))
                    end=newPoint.coords[0]
                for attr in values.keys():
                    if attr=="geometry": attrDict[j]["geometry"]=LineString([start, end])
                    else: attrDict[j][attr]=values[attr]
                start=newPoint.coords[0]
                j+=1
        else:
            attrDict[j]=values
            j+=1
    NewGrid=gpd.GeoDataFrame().from_dict(attrDict)
    NewGrid.crs=GeoDataLineString.crs
    return NewGrid

#%%
def linePolyIntersectBoolean(lineDataFrame,
                             polyDataFrame,
                             name="boolStreet",
                             save=False,
                             precise=False,
                             savepath=None):
    '''
    checks if Polygon dataframe intersects with a linestring dataframe
    input:
        -lineDataFrame: geopandas dataframe with linestrings
        -polyDataFrame: geopandas dataframe with polygons
        -name: name of new column in dataframe for boolean selection
    return:
        -polyDataFrame: geopandas dataframe with polygons and one additional column
    '''
    dictIntersect={}
    spatial_index = lineDataFrame.sindex
    for (gemIndex, gemValue) in polyDataFrame.iterrows():
        possible_matches_index = list(spatial_index.intersection(gemValue["geometry"].bounds))
        possible_matches = lineDataFrame.iloc[possible_matches_index]
        nMatches=len(possible_matches.index)
        if precise:
            precise_matches = possible_matches[possible_matches.intersects(gemValue["geometry"])]
            nMatches=len(precise_matches.index)
        if nMatches>0:
            dictIntersect[gemIndex]=True
        else:
            dictIntersect[gemIndex]=False
    polyDataFrame[name]=pd.Series(dictIntersect)*1
    if save:
        polyDataFrame.to_file(savepath)
    return polyDataFrame      
#%%

def createCluster(FuelingNew, clusterSize, ClusterGraph=None, name="Cl"):
    '''
    automatic selection of multiple or single cluster selection
    '''
    if isinstance(ClusterGraph, type(None)):
        return createSingleCluster(FuelingNew, clusterSize, name="Cl")
    else:
        return createMultCluster(FuelingNew, clusterSize, ClusterGraph, name="Cl")
#%%
def createSingleCluster(FuelingNew, clusterSize, name="Cl"):
    '''
    workflow for clustering fueling stations based on kmeans algorithm
    to a given mean clustersize
    
    input:
        FuelingNew: Fueling station GeoDataFrame (geopandas)
        clusterSize: average number of fueling stations per cluster
        name: Unique ID-Name for created Cluster
    return:
        GeoDataFrame (geopandas) with Clusterlocations. The Fueling GeoDataFrame
        is extended by respectice ClusterID
    '''
    from scipy.cluster import vq
    from shapely.geometry import Point
    from sklearn.cluster import KMeans
    obs=point_array(FuelingNew)
    nCluster=int(max(np.round(len(FuelingNew)/clusterSize),1))
    #centroids, variance  = vq.kmeans(test, nCluster, iter=100, )
    kmeans=KMeans(n_clusters=nCluster, random_state=42).fit(obs)
    identified, distance = vq.vq(obs, kmeans.cluster_centers_)
    Cluster=gpd.GeoDataFrame(geometry=[Point(x) for x in kmeans.cluster_centers_])
    Cluster["intIndex"]=Cluster.index
    Cluster.index=[name+ str(x) for x in Cluster.intIndex]
    Cluster["ID"]=Cluster.index
    FuelingNew["ClusterID"]=[name+ str(x) for x in identified]
    FuelingNew["distToCl"]=distance/1000
    Cluster["H2Demand_kt"]=FuelingNew.groupby(by="ClusterID")["H2Demand_kt_F"].sum()
    Cluster["numberOfFS"]=FuelingNew.groupby(by="ClusterID").size()
    Cluster.crs=FuelingNew.crs
    return Cluster
#%%
def createMultCluster(FuelingNew, clusterSize, ClusterGraph, name="Cl"):
    '''
    Clustering of fueling stations for multiple separate regions.
    
    input:
        FuelingNew: Fueling station GeoDataFrame (geopandas)
        clusterSize: average number of fueling stations per cluster
        name: Unique ID-Name for created Cluster
    return:
        GeoDataFrame (geopandas) with Clusterlocations. The Fueling GeoDataFrame
        is extended by respectice ClusterID
    '''
    dic={}
    i=0
    for subgraph in nx.connected_components(ClusterGraph):
        dic[i]=subgraph
        i+=1
    dic.keys()
    dicFueling={i:FuelingNew.loc[[x in dic[i] for x in FuelingNew.index]].copy() for i in dic.keys()}
    dicCluster={i:createSingleCluster(dicFueling[i], clusterSize, name=name+str(i)) for i in dicFueling.keys()}
    Cluster=dicCluster[list(dicCluster.keys())[0]]
    FuelingNew=dicFueling[list(dicFueling.keys())[0]]
    for i in list(dicCluster.keys())[1:]:
        Cluster=Cluster.append(dicCluster[i])
        FuelingNew=FuelingNew.append(dicFueling[i])
    FuelingNew=FuelingNew.sort_values(by="intIndex")
    Cluster["intIndex"]=range(len(Cluster.index))
    Cluster.crs=FuelingNew.crs
    return Cluster, FuelingNew

#%%
def cutLineAtPoints(line, points):
    # First coords of line (start + end)
    coords = [line.coords[0], line.coords[-1]]
    # Add the coords from the points
    coords += [list(p.coords)[0] for p in points]
    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]
    # sort the coords based on the distances
    # see http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    # generate the Lines
    lines = [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]
    return lines
def simplifyLinesAndCrossings(gpdLines):
    '''
    input:
    Geopandas dataframe with linestrings
    
    output:
    Geopandas Dataframe with linestrings in separate sections, all points and cat at crossings
    Geopandas Dataframe with  unique points of the linestring to select the coordinates
    '''
    singleLines=[]
    for line in gpdLines.geometry:
        length=len(line.coords)
        for x in range(length-1):
            singleLines.append(LineString([line.coords[x], line.coords[x+1]]))
    SingleLinesGDF=gpd.GeoDataFrame(geometry=singleLines)
    newLines=[]
    for key, values in SingleLinesGDF.iterrows():
        iterSectionsBool=SingleLinesGDF.intersects(values["geometry"])
        iterSections=SingleLinesGDF.intersection(values["geometry"]).loc[iterSectionsBool]
        iterPoints=iterSections.loc[iterSections.index!=key]
        if iterPoints.size>0:
            lines=cutLineAtPoints(values["geometry"],[iterPoints[x] for x in iterPoints.index])
            newLines.extend(lines)
        else:
            newLines.append(values["geometry"])
    
    newGrid=gpd.GeoDataFrame(geometry=newLines)
    newGrid.crs=gpdLines.crs
    newGrid["coordsIn"]=[x.coords[0] for x in newGrid.geometry]
    newGrid["coordsOut"]=[x.coords[-1] for x in newGrid.geometry]
    newGrid["distance"]=newGrid.length/1000
    newGrid["weightedDistance"]=newGrid["distance"]*1
    gridPoints=rem_dupl_un(newGrid)
    gridPoints.index=gridPoints["coords"]
    newGrid["inputID"]=gridPoints.loc[newGrid["coordsIn"].values, "ID"].values
    newGrid["targetID"]=gridPoints.loc[newGrid["coordsOut"].values, "ID"].values
    newGrid=newGrid.loc[[values["inputID"]!=values["targetID"] for key, values in newGrid.iterrows()]].copy()
    newGrid["ID"]=[(values["inputID"],values["targetID"]) for key, values in newGrid.iterrows()]
    newGrid=newGrid.loc[newGrid["ID"].drop_duplicates().index]
    
    gridPoints.index=gridPoints["ID"].values
    
    return newGrid, gridPoints

                                                                                      