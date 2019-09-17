# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:22:38 2017

@author: M.Reuss
"""

from HIM.utils import *
from HIM import dataHandling as sFun
#from HyInfraGis.HyPyOpt.expansion_compute_physical_parameters_adjusted import expansion_compute_physical_param as getPhys 

import pyomo.environ as pyomo
import pyomo.opt as opt
from scipy import sqrt
if __name__ =="__main__":
    logger=setupLogger("logfile.txt")
else:
    try: from __main__ import logger
    except:
        logger=setupLogger("logfile.txt")
        
#%%
class PipeNetWork(nx.Graph):
    '''
    Class derived from networkx.Graph(), that defines the Network, i.e. the pipeline or truck network. 

    It is able to transform the defined pipelineGraph into a pyomo concrete model
    and can optimize the capacities of the Graph.
    '''
    def __init__(self,**kwargs):
        '''
        Initializes an energy system network.
        
        '''
        nx.Graph.__init__(self)
        self.optimized = False
        self.pos = kwargs.get('pos', False)
        self.labelpos = None
        self.M = None
        self.minSpanTree=kwargs.get('minSpanTree', False)
    
    def initializeEdges(self, distMat):
        '''
        Initializes the edges of the network based on the distance matrice given
        '''
        self.add_edges_from(list(zip(distMat["inputID"], distMat["targetID"])),
                                 capacity=0)
        self.attr=[]
        for attr in distMat.columns:
            nx.set_edge_attributes(self, attr, distMat[attr].to_dict())
            self.attr.append(attr)
        
        self.add_nodes_from(nodes = self.nodes(),
                                  productionMax = 0,
                                  production = 0,
                                  demand = 0)

    def useMinSpanTree(self, **kwargs):
        '''
        apply minimum spanning tree on Graph
        kwargs: 
        weight: attribute name to weight on
        '''
        weight=kwargs.get("weight", "weight")
        G=nx.minimum_spanning_tree(self, weight=weight)
        delEdges=[]
        for edge in self.edges_iter():
            if not G.has_edge(edge[0],edge[1]):
                delEdges.append((edge[0],edge[1]))
        self.remove_edges_from(delEdges)
        self.minSpanTree=True
        
    def reduceNetworkSize(self):
        '''
        input
        NetworkXGraph: networkX Graph you want to reduce
        attribute: The attribute of old edges that shall be merged with new edge
        output
        reduced NetworkX Graph
        __________
        Eliminate Nodes with two Neighbors for network reduction    
        '''
        x=self.number_of_nodes()
        y=0
        logger.info("Number of nodes before Reduction: " + str(x))
        while x!=y:
            x=y
            for node in self.nodes():
                if "G" in node:
                    neighbors=self.neighbors(node)
                    if len(neighbors)==2:

                        attrList=self[node][neighbors[0]].keys()
                        attrDic={}
                        for attr in attrList:
                            attrDic[attr]=self[node][neighbors[0]][attr] + self[node][neighbors[1]][attr]

                        self.add_edge(neighbors[0], neighbors[1], attr_dict=attrDic)                        

                        self.remove_node(node)
            y=self.number_of_nodes()
            logger.info("Number of nodes after Reduction: " + str(y))

    def reduceNetworkSizeWater(self):
        '''
        input
        NetworkXGraph: networkX Graph you want to reduce
        attribute: The attribute of old edges that shall be merged with new edge
        output
        reduced NetworkX Graph
        __________
        Eliminate Nodes with two Neighbors for network reduction    
        '''
        x=self.number_of_nodes()
        y=0
        logger.info("Number of nodes before Reduction: " + str(x))
        while x!=y:
            x=y
            for node in self.nodes():
                if "Water" in node:
                    neighbors=self.neighbors(node)
                    if len(neighbors)==2:

                        attrList=self[node][neighbors[0]].keys()
                        attrDic={}
                        for attr in attrList:
                            attrDic[attr]=self[node][neighbors[0]][attr] + self[node][neighbors[1]][attr]

                        self.add_edge(neighbors[0], neighbors[1], attr_dict=attrDic)                        

                        self.remove_node(node)
            y=self.number_of_nodes()
            logger.info("Number of nodes after Reduction: " + str(y))
            
    def checkFeasibility(self):
        '''
        simple comparison between Sources and Sinks
        '''
        prod=np.ceil(sum(nx.get_node_attributes(self, "productionMax").values())*1000)/1000
        dem=sum(nx.get_node_attributes(self, "demand").values())
        if prod<dem:
            logger.info("Infeasible problem! The selected production is not able to supply the demand.\nPlease add capacity or cut demand.")
            raise SystemExit
#_____________________________________________________________________________        
    def initOpti(self,**kwargs):
        '''
        Transforms the defined Graph into Concrete Model of Pyomo
        
        Initialize the OptiSystem class for the pipeline:
        Keyword arguments:
            linear --> shall the optimization solve linear?
            cistBinary --> binary cost element
            costLinear --> Linear cost element
            BigM --> maximal capacity
            SmallM --> minimal capacity
    
        Optional keyword arguments:
        '''

        self.checkFeasibility()
        self.M=pyomo.ConcreteModel()
        self.M.isLP=kwargs.get('linear', False)
        self.M.isInt=kwargs.get('integer', False)
        self.M.A=kwargs.get('costBinary', 474.77)
        self.M.B=kwargs.get('costLinear', 1.3695)
        self.M.QPa=kwargs.get('QPa', -0.0002)
        self.M.QPb=kwargs.get('QPb', 2.1487)     
        self.M.BigM=kwargs.get('BigM', 3000)
        self.M.SmallM=kwargs.get('SmallM', 0)
        self.M.weight=kwargs.get('weight', 'weightedDistance')
        self.M.treeStructure=kwargs.get("tree", False)
        
        self.M.edgeIndex=self.edges()                     
        self.M.edgeIndexForw=[(node1,node2) for (node1,node2) in self.M.edgeIndex]
        self.M.edgeIndexBack=[(node2,node1) for (node1,node2) in self.M.edgeIndex]
        
        self.M.edgeIndexFull = self.M.edgeIndexForw
        self.M.edgeIndexFull.extend(self.M.edgeIndexBack)
        
        self.M.edgeLength=nx.get_edge_attributes(self, self.M.weight)
        
        self.M.edgeCapacity=pyomo.Var(self.M.edgeIndex)
        if not self.M.isLP:
            self.M.edgeCapacityInt=pyomo.Var(self.M.edgeIndex, within=pyomo.Binary)
        self.M.edgeFlow=pyomo.Var(self.M.edgeIndexFull, within = pyomo.NonNegativeReals)
        self.M.nodeIndex=self.nodes()
        
        self.M.nodeProductionMax=nx.get_node_attributes(self,"productionMax")
        self.M.nodeDemand=nx.get_node_attributes(self,"demand")
        
        self.M.nodeNeighbour = {self.M.nodeIndex[i]: neighbours for i,neighbours in enumerate(self.adjacency_list()) }

        self.M.nodeProduction=pyomo.Var(self.M.nodeIndex, within = pyomo.NonNegativeReals)

        #Constraints
        def massRule(M, n_index):
                return (sum(M.edgeFlow[(n_neighbour,n_index)] - M.edgeFlow[(n_index,n_neighbour)] for n_neighbour in M.nodeNeighbour[n_index])+M.nodeProduction[n_index]-M.nodeDemand[n_index])==0
        self.M.massCon = pyomo.Constraint(self.M.nodeIndex, rule=massRule)  
           
        if not self.M.isLP:
            def maxRule(M, e_index0, e_index1):
                return M.edgeCapacity[(e_index0,e_index1)]<= M.BigM*M.edgeCapacityInt[(e_index0, e_index1)]     
            self.M.maxCon = pyomo.Constraint(self.M.edgeIndex, rule=maxRule)  
            
        def capacityRule(M, e_index0, e_index1):
                return M.edgeFlow[(e_index0,e_index1)] + M.edgeFlow[(e_index1,e_index0)] <= M.edgeCapacity[(e_index0,e_index1)]        
        self.M.capacityCon = pyomo.Constraint(self.M.edgeIndex, rule=capacityRule) 
        
        def prodRule(M, n_index):
                return M.nodeProduction[n_index]<=M.nodeProductionMax[n_index]        
        self.M.prodCon=pyomo.Constraint(self.M.nodeIndex, rule=prodRule)

        if self.M.treeStructure: 
            self.M.nodeInt=pyomo.Var(self.M.nodeIndex, within=pyomo.Binary)
        
            def nodeRule1(M, n_index):
                return (sum(M.edgeFlow[(n_neighbour,n_index)] + M.edgeFlow[(n_index,n_neighbour)] for n_neighbour in M.nodeNeighbour[n_index]))<=2*M.BigM*M.nodeInt[n_index]
            def nodeRule2(M, n_index):
                return (sum(M.edgeFlow[(n_neighbour,n_index)] + M.edgeFlow[(n_index,n_neighbour)] for n_neighbour in M.nodeNeighbour[n_index]))>=M.nodeInt[n_index]
            
            self.M.nodeCon1 = pyomo.Constraint(self.M.nodeIndex, rule=nodeRule1)  
            self.M.nodeCon2 = pyomo.Constraint(self.M.nodeIndex, rule=nodeRule2)
            
            
            
            def treeRule(M):
                    return sum(intVal for intVal in M.edgeCapacityInt.values()) <=  sum(intVal for intVal in M.nodeInt.values())-1    
            self.M.treeCon=pyomo.Constraint(rule=treeRule)
        
        #Objective Function
        if self.M.isLP:
            def objRule(M): 
                return (sum(M.edgeCapacity[e_index]*M.edgeLength[e_index] for e_index in M.edgeIndex))     

        elif self.M.isInt:
            def objRule(M): 
                return (sum(M.edgeCapacityInt[e_index]*M.edgeLength[e_index] for e_index in M.edgeIndex))         
        else:
            def objRule(M): 
                return (sum((M.edgeCapacityInt[e_index]*M.A + M.edgeCapacity[e_index]*M.B)*M.edgeLength[e_index] for e_index in M.edgeIndex))     
        self.M.obj=pyomo.Objective(rule=objRule)
#_____________________________________________________________________________
    def initOptiTruck(self,**kwargs):
        '''
        Transforms the defined Graph into Concrete Model of Pyomo
        
        Initialize the OptiSystem class.
    
        '''
        self.checkFeasibility()
        self.M=pyomo.ConcreteModel()
        self.M.isLP=True
        self.M.weight=kwargs.get('weight', 'weightedDistance')
        
        self.M.edgeIndex=self.edges()                     
        self.M.edgeIndexForw=[(node1,node2) for (node1,node2) in self.M.edgeIndex]
        self.M.edgeIndexBack=[(node2,node1) for (node1,node2) in self.M.edgeIndex]
        
        self.M.edgeIndexFull = self.M.edgeIndexForw
        self.M.edgeIndexFull.extend(self.M.edgeIndexBack)
        
        self.M.edgeCapacity=pyomo.Var(self.M.edgeIndex)

        self.M.edgeFlow=pyomo.Var(self.M.edgeIndexFull, within = pyomo.NonNegativeReals)
        
        self.M.edgeLength=nx.get_edge_attributes(self, self.M.weight)
        
        self.M.nodeIndex=self.nodes()
        
        self.M.nodeProductionMax=nx.get_node_attributes(self,"productionMax")
        
        self.M.nodeDemand=nx.get_node_attributes(self,"demand")
        
        self.M.nodeNeighbour = {self.M.nodeIndex[i]: neighbours for i,neighbours in enumerate(self.adjacency_list()) }
        
        self.M.nodeProduction=pyomo.Var(self.M.nodeIndex, within = pyomo.NonNegativeReals)
        
        #Constraints
        def massRule(M, n_index):
                return (sum(M.edgeFlow[(n_neighbour,n_index)] - M.edgeFlow[(n_index,n_neighbour)] for n_neighbour in M.nodeNeighbour[n_index])+M.nodeProduction[n_index]-M.nodeDemand[n_index])==0
        self.M.massCon = pyomo.Constraint(self.M.nodeIndex, rule=massRule)  
                                     
        def capacityRule(M, e_index0, e_index1):
                return M.edgeFlow[(e_index0,e_index1)] + M.edgeFlow[(e_index1,e_index0)] <= M.edgeCapacity[(e_index0,e_index1)]        
        self.M.capacityCon = pyomo.Constraint(self.M.edgeIndex, rule=capacityRule)  
        
        def prodRule(M, n_index):
                return M.nodeProduction[n_index]<=M.nodeProductionMax[n_index]        
        self.M.prodCon=pyomo.Constraint(self.M.nodeIndex, rule=prodRule)         
        
        #Objective Function
        if self.M.isLP:
            def objRule(M): 
                return (sum(M.edgeCapacity[e_index]*M.edgeLength[e_index] for e_index in M.edgeIndex))     
        
        self.M.obj=pyomo.Objective(rule=objRule)

    # Optimization
    def optModel(self, **kwargs):

        
        self.solver = kwargs.get('solver','gurobi')
        self.optprob = opt.SolverFactory(self.solver)
        self.optprob.options["timeLimit"]=kwargs.get('timeLimit',2000)
        self.optprob.options["threads"]=kwargs.get('threads',7)
        self.optprob.options["MIPgap"]=kwargs.get('gap',0.005)
        self.optprob.options["Heuristics"]=0.5
                            
        logfile=os.path.join(kwargs.get('logPath',""),"GurobiLog.txt")
        self.optprob.options["logfile"]=kwargs.get('logfile',logfile)
        self.optiRes = self.optprob.solve(self.M,tee=kwargs.get('tee',True))
    def optNLModel(self):
        self.solver = 'ipopt'
        self.solver_io = 'nl'
        self.optprob = opt.SolverFactory(self.solver,solver_io=self.solver_io)
        self.optiRes = self.optprob.solve(self.M,tee=kwargs.get('tee',True))#, warmstart=True)
#______________________________________________________
    def getEdgesAsGpd(self, coordSeries, analysisType, minCapacity=20, weighted=True, weightedTransmission=True, costCalc="Krieg",lbExport=1e-6, **kwargs):
        '''
        input:
            NX Graph --> Graph to implement
            coordSeries: Coordinates of all potential Nodes
        '''
        '''
    input:
        pyomoVariable --> Variable from whcih to extract the values
        coordSeries: Coordinates of all potential Nodes
    '''

        dicEdges=self.M.edgeFlow.get_values()
        nx.set_edge_attributes(self, "capacity", dicEdges)
        dicEdges={k:v for (k,v) in dicEdges.items() if v > lbExport}
        EdgesTotal = gpd.GeoDataFrame([(k[0], k[1], v) for (k,v) in dicEdges.items()],
                                       index=[k for k in dicEdges.keys()],
                                       columns=["inputID","targetID", "capacity"])
        
        LinesIn=coordSeries.loc[EdgesTotal["inputID"].values].geometry.values
        LinesOut=coordSeries.loc[EdgesTotal["targetID"].values].geometry.values
        EdgeCoords=gpd.GeoDataFrame(index=EdgesTotal.index)
        EdgeCoords["inputCoords"]=LinesIn
        EdgeCoords["outputCoords"]=LinesOut
        EdgesTotal["geometry"]=""
        EdgesTotal["distribution"]=False
        EdgesTotal.loc[["F" in tup[0] or "F" in tup[1] for tup in EdgesTotal.index ], "distribution"]=True
        
        geodict={}              
        for key, values in EdgeCoords.iterrows():
            geodict[key]=LineString([values["inputCoords"], EdgeCoords["outputCoords"][key]])
        EdgesTotal["geometry"]=gpd.GeoSeries(geodict)
        for attr in self.attr:
            EdgesTotal[attr]=[self[key[0]][key[1]][attr] for key in dicEdges.keys()]
        
        EdgesTotal["capacityMax"]=EdgesTotal.capacity
        EdgesTotal.loc[EdgesTotal.capacityMax<minCapacity, "capacityMax"]=minCapacity
        if costCalc=="Krieg":    
            EdgesTotal["diameter"]=sqrt(sFun.getDiameterSquare(EdgesTotal["capacityMax"].values))
            EdgesTotal["lineCostSpec"]=sFun.getSpecCost(EdgesTotal["capacityMax"], source="Krieg", base="diameter", **kwargs)  
            
            if not weightedTransmission:
                try:
                    EdgesTotal.loc[EdgesTotal["distribution"]==False,"weightedDistance"]=EdgesTotal.loc[EdgesTotal["distribution"]==False, "distance"]
                except:
                    logger.info("weighted Transmission not possible")
            if weighted:
                EdgesTotal["lineCost"]=EdgesTotal["lineCostSpec"]*EdgesTotal["weightedDistance"]*1000
            else:
                try:
                    EdgesTotal["lineCost"]=EdgesTotal["lineCostSpec"]*EdgesTotal["distance"]*1000
                except:
                    EdgesTotal["lineCost"]=EdgesTotal["lineCostSpec"]*EdgesTotal["weightedDistance"]*1000
                 
        return EdgesTotal

       
#______________________________________________________________________________
    def getProductionNodes(self):
        '''
        input:
            pyomoVariable --> Variable from whcih to extract the values
            coordSeries: Coordinates of all potential Nodes
        '''
        NodesTotal=gpd.GeoDataFrame([(v[1].value) for v in self.M.nodeProduction.iteritems()],
                                     index=[(v[0]) for v in self.M.nodeProduction.iteritems()],
                                     columns=["production"])
        
        return NodesTotal

