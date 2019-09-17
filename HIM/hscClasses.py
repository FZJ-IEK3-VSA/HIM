# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:44:44 2017

@author: m.reuss
"""

from HIM.utils import *
import HIM.dataHandling as sFun
import CoolProp.CoolProp as CP
#%% Motherclass
class Module(object):
    '''
    Generic mother class of all defined modules. 
    Inputs:
    form: Storage form like gaseous, liquid, lohc, metalhydride...
    pressure: in bar
    electricityCost: €/kWh prize for bought electricity
    NGCost: €/kWh prize for bought heat
    WACC: interestrate for the whole system
    demand: daily demand for hydrogen (normally vector)
    distance: distance between storage and distribution
    dfTable: Dataframe for Parameter
    '''
    
    def __init__(self, demand, dfTable, totalH2Demand=None):
        #Initialize the Class with General Parameters
        self.GeneralTab=dfTable['General']['General']
        self.LHVDiesel=self.GeneralTab["LHVDiesel"]
        self.electricityCostRES = self.GeneralTab['electricityCostRES']
        self.NGCost = self.GeneralTab['NGCost']
        self.heatGain = self.GeneralTab['heatGain']
        self.WACC = self.GeneralTab['WACC']
        self.storageDays=self.GeneralTab['storageDays']
        self.driverCost=self.GeneralTab['driverCost']
        self.demand=demand
        self.waterCost=self.GeneralTab['waterCost']
        self.dieselCost=self.GeneralTab['dieselCost']
        self.hours=self.GeneralTab['op. Hours RES']
        self.overDimension=1/(self.hours/8760)
        self.utilization=self.GeneralTab['utilization Station']     
        self.storagePart=self.GeneralTab['storagePart']  
        self.residentialTime=self.storageDays/self.storagePart
        self.speed=self.GeneralTab['truckSpeed']
        self.eMultiplier=self.GeneralTab['eMultiplier']
        self.numberOfTrucks=0
        self.pipelineLength=0
        self.CAPEX=0
        self.fixOPEX=0
        self.varOPEX=0
        self.totalH2Demand=totalH2Demand
        self.CO2Emissions=0
        self.primary=0
    def getAnnuity(self,WACC,lifetime):
        '''
        Function to return the Annuity
        '''
        annuity=(WACC*(1+WACC)**lifetime)/((1+WACC)**lifetime-1)
        return annuity
        
        
    def dayToSec(self,demand):
        '''
        mutation from kg/day to kg/s
        '''
        massflow=demand/24/3600
        return massflow

    def getElectricityCost(self,annualDemand):
        '''
        get electricity Price based on annual production
        values from eurostat 2016 http://appsso.eurostat.ec.europa.eu/nui/show.do
        '''

        eCost=annualDemand/1000
        try:
            eCost[eCost>70000.]=70001
 
        except:
            eCost=np.array([eCost])


        eCost[eCost>70000.]=self.GeneralTab.loc["BandIF"]
        eCost[eCost>20000.]=self.GeneralTab.loc["BandIE"]
        eCost[eCost>2000.]=self.GeneralTab.loc["BandID"]
        eCost[eCost>500.]=self.GeneralTab.loc["BandIC"]
        eCost[eCost>20.]=self.GeneralTab.loc["BandIB"]
        eCost[eCost>1.]=self.GeneralTab.loc["BandIA"]
                 
        return eCost*self.eMultiplier
    
    def getMeanValue(self, variable):
        if self.totalH2Demand is None:
            return np.sum(self.demand*variable)/np.sum(self.demand)
        else:
            return np.sum(self.demand*variable)/(self.totalH2Demand/365)
        
#%%Production Instance
class Import(Module):
    def __init__(self, demand, name, dfTable):
        '''
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen per day
        basic module for starting costs of supply chain
        '''
         #Initialize of Motherclass
        Module.__init__(self, demand, dfTable)
        self.costIn =dfTable["Import"][name]['costIn']
        self.CO2Emissions=dfTable["Import"][name]['emissionsIn']
        self.primary=dfTable["Import"][name]['primaryIn']
    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''
        self.TOTEX=self.costIn
        return self.TOTEX
        
    def getTotalCost(self):
        return self.getTOTEX()
    
    def getInvest(self):
        return 0

    def getTotalInvest(self):
        return 0
    
    def getMeanTOTEX(self):
        return self.getTOTEX()
        
    def getDemand(self):
        return 0., 0., 0., 0., 0.
    
    def getExpenditures(self):
        return self.TOTEX, 0, 0           

#%%Production Instance
class Production(Module):
    def __init__(self, demand, name, dfTable):
        '''
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen per day
        distance: distance between hydrogen source and sink
        name: Kind of Transport Technology
        
        calculates the specific costs of hydrogen production
        '''
         #Initialize of Motherclass
        Module.__init__(self, demand, dfTable)
        self.electricityCost=self.electricityCostRES
        #Import Transport Table
        self.ProductionTab=dfTable['Production']
        
        self.form =self.ProductionTab[name]['form']
        self.pressureOut =self.ProductionTab[name]['pressureOut']
        self.investBase =self.ProductionTab[name]['investBase']
        self.investCompare =self.ProductionTab[name]['investCompare']
        self.investScale =self.ProductionTab[name]['investScale']
        self.investLifetime =self.ProductionTab[name]['investLifetime']
        self.boilOff =self.ProductionTab[name]['boilOff']        
        self.investOM =self.ProductionTab[name]['investOM']
        self.electricityDemand =self.ProductionTab[name]['electricityDemand']
        self.waterDemand =self.ProductionTab[name]['waterDemand']
        
    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''        
        self.annuity=self.getAnnuity(self.WACC, self.investLifetime)
        self.invest=self.investBase*(self.demand*self.overDimension/self.investCompare)**self.investScale
        self.CAPEX=(self.annuity*self.invest)/(self.demand*365)
        self.electricityPower=self.demand*self.electricityDemand/24*self.overDimension
        self.fixOPEX=(self.investOM*self.invest)/(self.demand*365)
        self.varOPEX=self.electricityDemand*self.electricityCost+self.waterCost*self.waterDemand
        self.OPEX=self.fixOPEX+self.varOPEX
        self.TOTEX=self.CAPEX+self.OPEX
        self.electricityDemandAnnual=self.electricityDemand*self.demand*365
        
        return self.TOTEX
        
    def getTotalCost(self):
        return self.getTOTEX()
    
    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest 

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        except:
            self.getTOTEX()
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        
    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 0., self.getMeanValue(self.electricityDemand), 0., 0., 0.
    def getDemandAbstract(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 0., self.electricityDemand, 0., 0., 0.    
    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX

#%% Storage Instance
class Storage(Module):
    def __init__(self, demand, name, dfTable, costH2In=0, days=False):
        '''
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen per day
        distance: distance between hydrogen source and sink
        name: Kind of Storage Technology
        
        calculates the specific costs of hydrogen storage
        '''
        
        #Initialize of Motherclass
            #Getting the General Informations
        Module.__init__(self, demand, dfTable)
        if days:
            self.storageDays=days
        #Import Storage Table
        self.name=name
        self.StorageTab=dfTable['Storage']
        #Initialize of Parameter
        self.pressureOut =self.StorageTab[name]['pressureOut']
        self.pressureIn =self.StorageTab[name]['pressureIn']
        self.form = self.StorageTab[name]['form']
        self.investBase= self.StorageTab[name]['investBase']
        self.investCompare= self.StorageTab[name]['investCompare']
        self.investScale= self.StorageTab[name]['investScale']
        self.investLifetime= self.StorageTab[name]['investLifetime']
        self.boilOff=self.StorageTab[name]['boilOff']
        self.investOM=self.StorageTab[name]['investOM']
        self.costH2In=costH2In
        self.cavBackUp=self.StorageTab["GH2-Tank"]['investBase']
        self.investLinear=self.StorageTab[name]['investLinear']
        self.pressureBased=bool(self.StorageTab[name]['pressureBased'])
        
    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''
        self.annuity=self.getAnnuity(self.WACC,self.investLifetime)
        self.capacity=self.storageDays*self.demand
        if self.pressureBased:
            self.density = CP.PropsSI('D','T',298,'P',self.pressureIn*1e5,'hydrogen')-CP.PropsSI('D','T',298,'P',self.pressureOut*1e5,'hydrogen'); # [T]=K, [P]=kPa, [h]=J/kg
            self.volume=self.capacity/self.density
            # Here, we calculate integer based caverns!
            self.numberOfCaverns=np.ceil(self.volume/self.investCompare)
            #self.numberOfCaverns=self.volume/self.investCompare
            self.invest=(self.numberOfCaverns>0)*self.investBase+self.numberOfCaverns*self.investLinear
            
            #self.invest=(self.investBase+self.investLinear)*(self.numberOfCaverns)**self.investScale
            #Backup: If cavern volume < 500.000 m³ == capacity < 4,000,000 kg --> Local storage in aboveground storage
            #self.invest.loc[self.capacity<4e6]=self.cavBackUp*self.capacity.loc[self.capacity<4e6]
            #self.numberOfCaverns.loc[self.capacity<4e6]=0
            
        else:
            self.invest=self.investBase*(self.capacity/self.investCompare)**self.investScale
        self.CAPEX=self.invest*self.annuity/(self.demand*365)
        self.fixOPEX=self.invest*self.investOM/(self.demand*365)
        self.varOPEX=self.storageDays*self.boilOff*self.costH2In
        self.OPEX=self.fixOPEX+self.varOPEX
        self.TOTEX=self.CAPEX+self.OPEX
        
        return self.TOTEX
        
    def getTotalCost(self):
        
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out        
    
    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest 

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        except:
            self.getTOTEX()
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        
    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen Electricity, Heat, Diesel
        return self.getMeanValue(1+self.storagePart*self.boilOff*self.storageDays), 0., 0., 0., 0.
    def getDemandAbstract(self):
        self.getTOTEX()
        #Output: Hydrogen Electricity, Heat, Diesel
        return 1+self.storagePart*self.boilOff*self.storageDays, 0., 0., 0., 0.
        
    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX        
        
#%%Storage Instance        
class Pipeline(Module):
    def __init__(self, totalH2Demand, totalLineCost, totalLength, dfTable, costH2In=0):
        '''
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen in kg per year
        distance: distance between hydrogen source and sink
        
        this module just calculates the specific pipelinecosts based on precalculated pipeline lengths and investments
        '''
         #Initialize of Motherclass
            #Getting the General Informations
        Module.__init__(self, None, dfTable, totalH2Demand)
        
        #Import Transport Table
        self.TransportTab=dfTable['Transport']       
        self.form =self.TransportTab["Pipeline"]['form']
        self.lifetime=self.TransportTab["Pipeline"]['pipeLifetime']
        self.OM=self.TransportTab["Pipeline"]['pipeOM']
        self.costH2In=costH2In
        self.invest=totalLineCost
        self.annualDemand=totalH2Demand
        self.length=totalLength
        self.demand=self.annualDemand/365

    def getTOTEX(self):        
        '''
        calculates the specific hydrogen costs
        '''        
        #annuity calculation for pipeline
        self.annuity=self.getAnnuity(self.WACC,self.lifetime)
        self.CAPEX=self.annuity*self.invest/self.annualDemand                     
        self.fixOPEX=self.OM*self.length/self.annualDemand
        self.varOPEX=0
        self.OPEX=self.fixOPEX+self.varOPEX
        #CAPEX
        self.TOTEX=self.CAPEX+self.OPEX
        
        ### Total Cost H2
        self.costH2Out=self.costH2In+self.TOTEX
        return self.TOTEX

    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out

    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return self.TOTEX
        except:
            self.getTOTEX()
            return self.TOTEX

    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1., 0., 0., 0., 0.#kWh/l       
    def getDemandAbstract(self):
        return self.getDemand()
    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX
#%%
class PipelineSingle(Module):
    '''
    This module is explicitly for the utilization in the abstract calculation
    '''
    def __init__(self,demand, distance, dfTable, kind="Transmission",costH2In=0, **kwargs):
                
        Module.__init__(self,
                        demand,
                        dfTable,                  
                        **kwargs)

        self.distance=distance
        self.TransportTab = dfTable['Transport']["Pipeline"]
        self.form = self.TransportTab['form']
        self.pressureIn = self.TransportTab['pressureIn']
        self.pipeInvestA = self.TransportTab['pipeInvestA']
        self.pipeInvestB = self.TransportTab['pipeInvestB']
        #self.pipeInvestC = self.TransportTab['pipeInvestC'] -->   Veraltet
        self.pipeLifetime = self.TransportTab['pipeLifetime']
        self.pipeHours = self.TransportTab['pipeHours']
        self.pipeOM = self.TransportTab['pipeOM']
        self.pipeSystem = self.TransportTab['pipeSystem']
        self.loadingtime = self.TransportTab['loadingtime']
        self.pressureStationMin = self.TransportTab['pipePressureStation'] 
        self.pressureHubMin = self.TransportTab['pipePressureHub']
        self.kind=kind
        if self.kind=="Transmission":
            self.pOut=self.pressureHubMin
            self.pIn=self.pressureIn
        else:
            self.pOut=self.pressureStationMin
            self.pIn=self.pressureHubMin
        #minimal diameter 100
        self.capacity=self.demand*self.overDimension#*self.overCapacity
        self.costH2In=costH2In
        
    def getDiameterAndCosts(self, H2Density=5.7, vPipeTrans=15):
        
        diameter=np.sqrt(sFun.getDiameterSquare(self.capacity*365/1e6, H2Density, vPipeTrans))*1000
        costs=sFun.getSpecCost(self.capacity*365/1e6, H2Density, vPipeTrans)*1e9
        return diameter, costs
    
    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''
        self.diameter, self.pipeInvestSpecific=self.getDiameterAndCosts()           
        self.invest = self.distance * self.pipeInvestSpecific
        self.pipeAnnuity = self.getAnnuity(self.WACC, self.pipeLifetime)
        self.annualProduction=self.demand*365
        self.CAPEX = self.pipeAnnuity * self.invest / (self.annualProduction)                         
        self.fixOPEX = self.pipeOM * self.distance *1000/(self.annualProduction)    
        self.varOPEX = 0
        self.boilOff = 0
        self.TOTEX =self.CAPEX + self.fixOPEX + self.varOPEX
        self.costH2Out=self.costH2In+self.TOTEX
        return self.TOTEX

    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out

    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return self.TOTEX
        except:
            self.getTOTEX()
            return self.TOTEX

    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1., 0., 0., 0., 0.#kWh/l       
    def getDemandAbstract(self):
        return self.getDemand()   
    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX


#%%Truck Instance        
class Truck(Module):
    def __init__(self, demand, distance, name, dfTable, costH2In=0, beeline=True, time=None, end=None, totalH2Demand=None):
        '''
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen per day
        distance: distance between hydrogen source and sink
        name: Kind of Transport Technology
        '''
         #Initialize of Motherclass
            #Getting the General Informations
        Module.__init__(self, demand, dfTable, totalH2Demand)
        
        #Import Transport Table
        self.distance = distance
        self.TransportTab=dfTable['Transport']       
        self.form =self.TransportTab[name]['form']
        self.truckInvest=self.TransportTab[name]['truckInvest']
        self.truckLifetime=self.TransportTab[name]['truckLifetime']
        self.truckHours=self.TransportTab[name]['truckHours']
        self.truckOMfix=self.TransportTab[name]['truckOMfix']
        self.truckDriver=self.TransportTab[name]['truckDriver']
        self.truckFuelDemandDiesel=self.TransportTab[name]['truckFuelDemandDiesel']
        self.truckFuelDemandH2=self.TransportTab[name]['truckFuelDemandH2']
        self.trailerInvest=self.TransportTab[name]['trailerInvest']
        self.trailerLifetime=self.TransportTab[name]['trailerLifetime']
        self.trailerHours=self.TransportTab[name]['trailerHours']
        self.trailerOM=self.TransportTab[name]['trailerOM']
        self.trailerCapacity=self.TransportTab[name]['trailerCapacity']
        self.loadingtime=self.TransportTab[name]['loadingtime']
        self.costH2In=costH2In
        self.boilOffHourly=self.TransportTab[name]['boilOffHourly']  
        self.toll=self.TransportTab[name]['truckToll'] 
        #self.stationDistance=np.ones((len(distance)))*self.distributionDistance
        self.beeline=beeline
        self.time=time
        self.end=end
        self.totalH2Demand=totalH2Demand
        
    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''
        #Annuity for the trailer truck
        self.truckAnnuity=self.getAnnuity(self.WACC,self.truckLifetime)
        #annuity calculation for the trailer
        self.trailerAnnuity=self.getAnnuity(self.WACC,self.trailerLifetime)        
        #time for traveling to destination and back with loading
        if self.beeline:
            self.truckTime=(self.distance/self.speed+self.loadingtime)*2
        else:
            self.truckTime=self.time*2+self.end*self.loadingtime


        #number of trucks needed for the transportation
        self.nTruck=self.truckTime/self.trailerCapacity/self.truckHours*self.demand*365
        self.nTruckPerDay=self.demand/self.trailerCapacity
        self.nTrailer=self.truckTime/self.trailerCapacity/self.trailerHours*self.demand*365
        self.invest=self.nTruck*self.truckInvest+self.nTrailer*self.trailerInvest
        #distance to Station and back
        self.truckDistance=self.distance*2
        #Used fuel for traveling to destination and back
        self.truckFuelUseD=self.truckFuelDemandDiesel*self.truckDistance/100
        self.truckFuelUseH2=self.truckFuelDemandH2*self.truckDistance/100
        #Additional Calculations
        self.truckHourly=(self.truckAnnuity+self.truckOMfix)*self.truckInvest/self.truckHours
        self.trailerHourly=(self.trailerAnnuity+self.trailerOM)*self.trailerInvest/self.trailerHours
        #OPEX calculation in €/kg
        self.truckCAPEX=self.truckAnnuity*self.truckInvest/self.truckHours*self.truckTime/self.trailerCapacity
        self.trailerCAPEX=self.trailerAnnuity*self.trailerInvest/self.trailerHours*self.truckTime/self.trailerCapacity
        #OPEX calculation in €/kg        
        self.truckOPEXHours=(self.truckOMfix*self.truckInvest/self.truckHours+self.driverCost*self.truckDriver)*self.truckTime/self.trailerCapacity
        self.truckOPEXFuel=(self.truckFuelUseD*self.dieselCost+self.truckFuelUseH2*self.costH2In)/self.trailerCapacity
        self.truckOPEXToll=self.toll*self.truckDistance/self.trailerCapacity
        self.truckOPEX=self.truckOPEXHours+self.truckOPEXFuel+self.truckOPEXToll
        self.H2Consumption=self.truckFuelUseH2/self.trailerCapacity
        
        self.trailerOPEX=self.trailerOM*self.trailerInvest/self.trailerHours*self.truckTime/self.trailerCapacity
        self.boilOff=self.boilOffHourly*self.truckTime/2
    
        #Total CAPEX OPEX and TOTEX
        self.varOPEX=self.truckOPEXFuel+ self.truckOPEXToll+self.costH2In*self.boilOff
        self.CAPEX=self.truckCAPEX+self.trailerCAPEX
        self.fixOPEX=self.truckOPEXHours
        self.OPEX=self.truckOPEX+self.trailerOPEX+self.costH2In*self.boilOff
        self.TOTEX=self.CAPEX+self.fixOPEX+self.varOPEX
        self.electricityDemandAnnual=0
        ### Total Cost H2
        self.costH2Out=self.costH2In+self.TOTEX
        self.numberOfTrucks=self.nTruck.sum()
        
        return self.TOTEX

    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out
    
    def getInvest(self):
        self.getTOTEX()
        return self.invest

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return self.getMeanValue(self.TOTEX)
        except:
            self.getTOTEX()
            return self.getMeanValue(self.TOTEX)
        
    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return (1+ self.getMeanValue(self.boilOff))*(1+self.getMeanValue(self.H2Consumption)), 0, 0., 0., self.getMeanValue(self.truckFuelUseD/self.trailerCapacity*self.LHVDiesel) #kWh/l
    def getDemandAbstract(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return (1+ self.boilOff)*(1+self.H2Consumption), 0, 0., 0., self.truckFuelUseD/self.trailerCapacity*self.LHVDiesel #kWh/l  

    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX
#%% Station Instance
class Station(Module):
    def __init__(self, demand, name, dfTable, costH2In=0):
        '''
        
        transport Module:
        Inputs (like Motherclass):
        
        demand: demand of hydrogen in kg per day
        name: Kind of Station Technology
        '''
        #Implementation of Motherclass
            #Getting the General Informations
        Module.__init__(self, demand, dfTable)
        #Import Transport Table
        
        self.maxCapacity=dfTable["General"].loc["targetStationSize","General"]
        self.stationInteger=np.ceil(self.demand/self.maxCapacity)
        self.StationTab=dfTable['Station']
        self.form = self.StationTab[name]['form']
        #self.pressureIn = self.StationTab[name]['pressureIn']
        #self.stationDemand=self.StationTab[name]['stationDemand']
        
        self.stationInvest=self.StationTab[name]['stationInvest']*self.stationInteger
        self.stationLifetime=self.StationTab[name]['stationLifetime']
        
        self.stationOM=self.StationTab[name]['stationOM']
        self.electricityDemand=self.StationTab[name]['electricityDemand']
        self.heatDemand=self.StationTab[name]['heatDemand']
        self.boilOffEff=self.StationTab[name]['boilOffEff']
        self.dieselDemand=self.StationTab[name]['dieselDemand']        
        self.costH2In=costH2In
        self.boolTrailer=self.StationTab[name]['boolTrailer'] 
        if "boilOff" in self.StationTab[name].index:
            self.boilOffEff=self.StationTab[name]["boilOff"]

        self.trailerInvest=self.boolTrailer*dfTable['Transport']['GH2-Truck']['trailerInvest']*self.stationInteger
        self.trailerLifetime=dfTable['Transport']['GH2-Truck']['trailerLifetime']
        self.trailerOM=self.boolTrailer*dfTable['Transport']['GH2-Truck']['trailerOM']
        self.trailerAnnuity=self.getAnnuity(self.WACC,self.trailerLifetime)

    def getTOTEX(self):
        '''
        calculates the specific hydrogen costs
        '''
        #Annuity Calculation
        self.annuity=self.getAnnuity(self.WACC,self.stationLifetime)
        
        self.annualDemand=self.demand*365
        #CAPEX calculation
        self.CAPEXStation=self.annuity*self.stationInvest/self.annualDemand
        self.CAPEXTrailer=self.trailerAnnuity*self.trailerInvest/self.annualDemand
        self.CAPEX=self.CAPEXTrailer+self.CAPEXStation
        #OPEX calculation
        self.fixOPEXTrailer=self.trailerOM*self.trailerInvest/self.annualDemand
        self.fixOPEXStation=self.stationOM*self.stationInvest/self.annualDemand
        self.fixOPEX=self.fixOPEXTrailer+self.fixOPEXStation
        
        
        self.electricityDemandAnnual=self.electricityDemand*self.annualDemand
        #variable electricity price - uncomment for predefined
        self.electricityCost=self.getElectricityCost(self.electricityDemandAnnual)
        
        self.varOPEX=self.electricityDemand*self.electricityCost+self.heatDemand*self.NGCost+self.boilOffEff*self.costH2In+self.dieselDemand*self.dieselCost/self.LHVDiesel
        #TOTEX calculation
        self.TOTEX=self.CAPEX+self.fixOPEX+self.varOPEX
        self.invest=(self.stationInvest+self.trailerInvest)*(self.demand**0)
        self.electricityPower=self.electricityDemand*self.demand/24
        self.numberOfTrucks=self.boolTrailer*self.stationInteger.sum()
        
        return self.TOTEX
        
    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out   
    
    def getInvest(self):
        self.getTOTEX()
        return self.invest

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        except:
            self.getTOTEX()
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
    
    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1+self.boilOffEff, 0. , self.getMeanValue(self.electricityDemand), self.getMeanValue(self.heatDemand), self.getMeanValue(self.dieselDemand)     
    def getDemandAbstract(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1+self.boilOffEff, 0. , self.electricityDemand, self.heatDemand, self.dieselDemand 

    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX
#%%Connector Class        
class Connector(Module):
    def __init__(self, demand, name,  dfTable, costH2In=0, pressureIn=30, pressureOut=200, nextStep="Pipeline", storagePartition=False):
        '''
        transport Module:
        Inputs (like Motherclass):

        demand: demand of hydrogen per day
        distance: distance between hydrogen source and sink
        name: Kind of Connection Technology
        position: Connected Modules
        '''
        #Implementation of Motherclass
        #Getting the General Informations
        Module.__init__(self, demand, dfTable)
        self.costH2In=costH2In
        self.dfTable=dfTable
        self.name=name
        self.pressureIn=pressureIn
        self.pressureOut=pressureOut        
        self.nextStep=nextStep
        if not storagePartition:
            self.storagePart=1
    def getValues(self, name):
        '''
        Import the Values of the specific Connector
        '''
                
        #Import Connector Table
        self.ConnectorTab=self.dfTable['Connector'][name]
                #Parameter Import
        self.investBase=self.ConnectorTab['investBase']
        self.investCompare=self.ConnectorTab['investCompare']
        self.investScale=self.ConnectorTab['investScale']
        self.investLifetime=self.ConnectorTab['investLifetime']
        self.investOM=self.ConnectorTab['investOM']
        self.electricityDemandBase=self.ConnectorTab['electricityDemandBase']
        self.electricityDemandScale=self.ConnectorTab['electricityDemandScale']
        self.electricityDemandCompare=self.ConnectorTab['electricityDemandCompare']
        self.heatDemand=self.ConnectorTab['heatDemand']
        self.heatSupply=self.ConnectorTab['heatSupply']
        self.boilOffEff=self.ConnectorTab['boilOffEff']
        self.system=self.ConnectorTab['system']
        self.maxCapacity=self.ConnectorTab['capacityMax']
        self.multiplier=self.demand//self.maxCapacity+1
        if self.nextStep=="Pipeline":
            self.fInst=self.ConnectorTab['installationFactorPipe']
        elif self.nextStep=="GH2-Truck":
            self.fInst=self.ConnectorTab['installationFactorTruck']
        else:
            self.fInst=self.ConnectorTab['installationFactor']
        self.heatToNG=self.dfTable["General"].loc["heatNGEquivalent","General"]
    def getTOTEX(self):
      
        self.getValues(self.name)
        
        self.overCapacity=1
        self.capacity=self.overCapacity*self.demand*self.overDimension
        #annual production
        self.annualProduction=self.demand*self.overCapacity*365
        #annuity calculation
        self.annuity=self.getAnnuity(self.WACC, self.investLifetime)                
        
        # If Connector == Compressor then the compression systematic is used
        if self.name=='Compressor':
            self.electricityDemand, self.electricityPower=getCompressionEnergy(self.pressureIn, self.pressureOut, self.capacity)            
            #Regarding maximum Capacity: If demand>maxCapacity--> use more sites
            #invest
            self.invest=self.fInst*self.investBase*(self.electricityPower/self.investCompare)**self.investScale
            #CAPEX
            self.CAPEX=(self.annuity*self.invest/self.annualProduction)
            #OPEX
            self.fixOPEX=(self.investOM*self.invest/self.annualProduction)
        else:#if no compression --> liquid or LOHC is used
            self.electricityDemand=self.electricityDemandBase*(self.demand/self.electricityDemandCompare)**self.electricityDemandScale
            
            #invest
            self.invest=self.investBase*(self.capacity/self.investCompare)**self.investScale
            #CAPEX
            self.CAPEX=(self.annuity*self.invest/self.annualProduction)
            #OPEX
            self.fixOPEX=(self.investOM*self.invest/self.annualProduction)
            
        self.electricityPower=self.electricityDemand*self.demand/24*self.overCapacity*self.overDimension
        self.varOPEX=self.storagePart*(self.electricityDemand*self.electricityCostRES
                                       +self.heatDemand*self.NGCost
                                       -self.heatSupply*self.heatGain
                                       +self.boilOffEff*self.costH2In)
        self.OPEX=self.fixOPEX+self.varOPEX
        #TOTEX
        self.TOTEX=self.CAPEX+self.OPEX
        self.costH2Out=self.costH2In+self.TOTEX
        if self.heatSupply*self.heatGain>0: self.heatDemand=-self.heatSupply/self.heatToNG
        return self.TOTEX
        
    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out        

    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest

    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
        try:
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
        except:
            self.getTOTEX()
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)    

    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1+self.storagePart*self.boilOffEff, self.storagePart*self.getMeanValue(self.electricityDemand), 0. , self.storagePart*self.getMeanValue(self.heatDemand), 0.
    def getDemandAbstract(self):
            self.getTOTEX()
            #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
            return 1+self.storagePart*self.boilOffEff, self.storagePart*self.electricityDemand, 0. , self.storagePart*self.heatDemand, 0.             
     
    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX
        
#%%
class Connector2(Module):
    def __init__(self, demand, name, dfTable, costH2In=0, pressureIn=30, pressureOut=200, nextStep="Pipeline", storagePartition=False):
        '''
        transport Module:
        Inputs (like Motherclass):

        demand: demand of hydrogen per day
        distance: distance between hydrogen source and sink
        name: Kind of Connection Technology
        position: Connected Modules
        '''
        #Implementation of Motherclass
        #Getting the General Informations
        Module.__init__(self, demand, dfTable)
        self.costH2In=costH2In
        self.dfTable=dfTable
        self.name=name
        self.pressureIn=pressureIn
        self.pressureOut=pressureOut
        self.nextStep=nextStep
        if not storagePartition:
            self.storagePart=1
        
    def getValues(self, name, kind):
        '''
        Import the Values of the specific Connector
        '''
        #Import Connector Table
        
        self.ConnectorTab=self.dfTable['Connector'][name]
                #Parameter Import
        self.investBase=self.ConnectorTab['investBase']
        self.investCompare=self.ConnectorTab['investCompare']
        self.investScale=self.ConnectorTab['investScale']
        self.investLifetime=self.ConnectorTab['investLifetime']
        self.investOM=self.ConnectorTab['investOM']
        self.electricityDemandBase=self.ConnectorTab['electricityDemandBase']
        self.electricityDemandScale=self.ConnectorTab['electricityDemandScale']
        self.electricityDemandCompare=self.ConnectorTab['electricityDemandCompare']
        self.system=self.ConnectorTab['system']
        if kind==0:        
            self.heatDemand=self.ConnectorTab['heatDemand']
            self.heatSupply=self.ConnectorTab['heatSupply']
            self.boilOffEff=self.ConnectorTab['boilOffEff']
        if self.nextStep!="GH2-Truck":
            self.fInst=self.ConnectorTab['installationFactorPipe']
        else:
            self.fInst=self.ConnectorTab['installationFactorTruck']
        self.heatToNG=self.dfTable["General"].loc["heatNGEquivalent","General"]
        
    def getTOTEX(self):
        self.getValues(self.name, kind=0)
        self.overCapacity=1
        self.capacity=self.overCapacity*self.demand
        #annual production
        self.annualProduction=self.demand*self.overCapacity*365
        
        #annuity calculation
        self.annuity=self.getAnnuity(self.WACC, self.investLifetime)                
        
        if self.name=='Compressor':
            self.electricityDemand, self.electricityPower=getCompressionEnergy(self.pressureIn, self.pressureOut, self.capacity)            
            #invest
            self.invest=self.fInst*self.investBase*(self.electricityPower/self.investCompare)**self.investScale
            #CAPEX
            self.CAPEX=(self.annuity*self.invest/self.annualProduction)
            #OPEX
            self.fixOPEX=(self.investOM*self.invest/self.annualProduction)
        else:
            self.electricityDemand=self.electricityDemandBase*(self.demand/self.electricityDemandCompare)**self.electricityDemandScale
            
            self.invest=self.investBase*(self.capacity/self.investCompare)**self.investScale
            #CAPEX
            self.CAPEX=(self.annuity*self.invest/self.annualProduction)
            #OPEX
            self.fixOPEX=(self.investOM*self.invest/self.annualProduction)
        
        #Variable electricity price industry - uncomment to work with predefined
        self.electricityPower=self.electricityDemand*self.demand/24
        self.annualElectricityDemand=self.electricityDemand*365*self.demand
        self.electricityCost=self.getElectricityCost(self.annualElectricityDemand)
        
        
        
        self.varOPEX=self.storagePart*(self.electricityDemand*self.electricityCost
                                       +self.heatDemand*self.NGCost
                                       -self.heatSupply*self.heatGain
                                       +self.boilOffEff*self.costH2In)
        self.OPEX=self.fixOPEX+self.varOPEX
        #TOTEX
        self.TOTEX=self.CAPEX+self.OPEX
        self.costH2Out=self.costH2In+self.TOTEX
        if self.heatSupply*self.heatGain>0: self.heatDemand=-self.heatSupply/self.heatToNG
        return self.TOTEX
        
    def getTotalCost(self):
        self.costH2Out=self.costH2In+self.getTOTEX()
        return self.costH2Out        
    
    def getInvest(self):
        try:
            return self.invest
        except:
            self.getTOTEX()
            return self.invest
    
    def getTotalInvest(self):
        try:
            return np.sum(self.invest)
        except:
            self.getTOTEX()
            return np.sum(self.invest)
    
    def getMeanTOTEX(self):
            self.getTOTEX()
            return np.sum(self.TOTEX*self.demand)/np.sum(self.demand)
    
    def getDemand(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return self.getMeanValue(1+self.storagePart*self.boilOffEff), 0, self.storagePart*self.getMeanValue(self.electricityDemand), self.storagePart*self.getMeanValue(self.heatDemand), 0

    def getDemandAbstract(self):
        self.getTOTEX()
        #Output: Hydrogen, ElectricityRES, electricityGrid, Heat, Diesel
        return 1+self.storagePart*self.boilOffEff, 0, self.storagePart*self.electricityDemand, self.storagePart*self.heatDemand, 0

    def getExpenditures(self):
        return self.CAPEX, self.fixOPEX, self.varOPEX