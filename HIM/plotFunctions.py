# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:52:20 2016

@author: m.reuss
"""
from HIM.utils import *
from mpl_toolkits.mplot3d.art3d import Text3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import numpy as np
import xlsxwriter
import matplotlib.image as mpimg
from descartes.patch import PolygonPatch
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from HIM import dataHandling as sFun
import pandas as pd
from six import next
from six.moves import xrange
try:
    from PIL import Image, ImageChops
except:
    print("please install PIL for full functionality")
from io import BytesIO
IMAGEPATH=path.join(path.dirname(STYLEPATH), "icons")


from mpl_toolkits.mplot3d import proj3d
np.set_printoptions(suppress=True)
#%%
def trim(im, border):
  bg = Image.new(im.mode, im.size, border)
  diff = ImageChops.difference(im, bg)
  bbox = diff.getbbox()
  if bbox:
      return im.crop(bbox)
#%%

def orthogonalProj(zfront, zback):
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -0.0001, zback]])

#%%


def trisurfplotMin(
        demandArray,
        distArray,
        z_array,
        min_array,
        line_array,
        dfHSC,
        zmax=8,
        zmin=5,
        figSize=0,
        saveFig=False,
        savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
        saveName='FigureComparison'):
    '''
    Function for plotting the costs as surface based on minimal costs + lines

    input parameter:
    X: Demand Array
    Y: Distance Array
    z_array: Array of solutions for different HSC (cutted)
    min_array: Array of minimal Costs
    line_Array: Array of intersection line
    dfHSC: DataFrame for the HSC
    '''
#    demandArray = demandArray / 1000
    x, y = np.meshgrid(demandArray, distArray)
    x = x.ravel()
    y = y.ravel()
    ymin = np.amin(min_array)
    ymax = np.amax(min_array)
   
    if figSize==0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figSize)
        
    fig.patch.set_facecolor('white')
    ax = fig.gca(projection='3d')
    ax.set_alpha(None)
    line = line_array[:, :].ravel()
    for i in range(len(dfHSC)):
        z = z_array[:, :, i].ravel()
        trip = np.array([x, y, z])
        tripx = (trip.T[~np.isnan(trip.T).any(axis=1)]).T
        if len(tripx.T) > 3:
            xx = tripx[0, :]
            yy = tripx[1, :]
            zz = tripx[2, :]
            xm = np.average(xx)
            ym = np.average(yy)
            zm = np.average(zz)
            ax.text(
                xm,
                ym,
                zm,
                dfHSC['General'][i],
                zorder=len(dfHSC) + i,
                ha='center',
                size=9,
                bbox=dict(
                    facecolor=(
                        1,
                        1,
                        1),
                    alpha=0.6,
                    edgecolor='none',
                    pad=1))

    trip = np.array([x, y, line])
    tripx = (trip.T[~np.isnan(trip.T).any(axis=1)]).T
    xx = tripx[0, :]
    yy = tripx[1, :]
    zz = tripx[2, :]

    surf = ax.plot_surface(
        demandArray,
        distArray,
        min_array,
        rstride=1,
        cstride=1,
        linewidth=0,
        cmap=cm.jet,
        antialiased=False,
        vmin=ymin,
        vmax=ymax)
    ax.plot_trisurf(x, y, line, color='black', linewidth=0, zorder=i + 1)
    surf.set_alpha(None)

    ax.set_alpha(None)
    ax.set_xlabel(r'Wasserstoffbedarf [t$_{H2}$/Tag]', labelpad=5, size=11)
    ax.set_ylabel(r'$\varnothing$ Transportentfernung [km]',labelpad=10, size=11)
    ax.set_zlabel(r'Wasserstoffkosten [EUR/kg$_{H2}$]', labelpad=-2, size=11)
    ax.set_zlim(zmin, ymax)
    proj3d.persp_transformation = orthogonalProj
#    ax.xaxis.set_major_formatter(FormatStrFormatter(r'%d $\frac{t}{day}$'))
#    ax.yaxis.set_major_formatter(FormatStrFormatter('%d km'))
#    ax.zaxis.set_major_formatter(FormatStrFormatter(r'%d $\frac{€}{kg}$'))



# Change Grid Color for better vizualization
    ax.xaxis.set_pane_color((0, 91 / 255, 130 / 255, 0.1))
    ax.zaxis.set_pane_color((0, 91 / 255, 130 / 255, 0.1))
    ax.yaxis.set_pane_color((0, 91 / 255, 130 / 255, 0.1))

#    ax.w_xaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.9)}})
#    ax.w_yaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.9)}})
#    ax.w_zaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.9)}})

    ax.w_xaxis.gridlines.set_lw(0.5)
    ax.w_yaxis.gridlines.set_lw(0.5)
    ax.w_zaxis.gridlines.set_lw(0.5)

    #fig.suptitle('Wasserstoffgesamtkosten nach Elektrolyseproduktion')

    # Insert Colorbar
    cbaxes = fig.add_axes([0.05, 0.25, 0.02, 0.5])
    cb = plt.colorbar(surf, cax=cbaxes)#, format=FormatStrFormatter(r'%.1f $\frac{€}{kg}$'))
    cb.set_label("Wasserstoffkosten [EUR/kg$_{H2}$]", labelpad=-50, size=11)
    plt.tight_layout()
#    
    if saveFig==True:
        
        plt.savefig(savePath+saveName)
        plt.savefig(savePath+saveName+".svg")
#        png1=BytesIO()
#        fig.savefig(png1,format='png')
#        png2 = Image.open(png1)
#        png2.save(savePath+saveName+'.tiff')
#        png1.close()
    plt.show()
#%%


def trisurfplotCompress(p_1_array, p_2_array, dem_array):
    '''
    Function for plotting the costs as surface based on minimal costs + lines

    input parameter:
    X: Demand Array
    Y: Distance Array
    z_array: Array of solutions for different HSC (cutted)
    min_array: Array of minimal Costs
    line_Array: Array of intersection line
    dfHSC: DataFrame for the HSC
    '''
    max = np.max(dem_array)
    x, y = np.meshgrid(p_1_array, p_2_array)
    x = x.ravel()
    y = y.ravel()
    fig = plt.figure()
    # fig.autolayout=True
    fig.patch.set_facecolor('white')
    #ax = plt.subplot2grid((1,6), (0, 0), colspan=6, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_alpha(None)

    surf = ax.plot_surface(
        p_1_array,
        p_2_array,
        dem_array,
        rstride=1,
        cstride=1,
        linewidth=0,
        cmap=cm.jet,
        antialiased=False,
        vmin=0,
        vmax=max)
    surf.set_alpha(None)

    ax.set_alpha(None)
    ax.set_xlabel('demand', size=14)
    ax.set_ylabel('Distance', size=14)
    ax.set_zlabel('Energy demand in kWh/kg')
    ax.set_zlim(0, max)
    proj3d.persp_transformation = orthogonalProj
    fig.suptitle('Energy demand for hydrogen compression')

    # Insert Colorbar
    cbaxes = fig.add_axes([0.05, 0.25, 0.02, 0.5])
    cb = plt.colorbar(surf, cax=cbaxes)
    # fig.colorbar(surf,shrink=0.5, aspect=15,pad=0.2, location='left')#orientation='vertical'
#    plt.tight_layout()

    plt.show()

#%%
# tornado chart example


def tornado(
        data,
        namesX,
        namesY,
        x,
        dist,
        dem,
        varValues,
        varUnits,
        dfHSC,
        imagepath,
        best,
        figSize=0,
        saveFig=False,
        savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
        saveName='FigureTornado'):
    '''
    '''
    dataMin = np.min(data) - 0.5
    dataMax = np.max(data) + 0.5
    maxValues=best
    names = namesY
    names[names == 'electricityCostLow'] = 'electricityCost\nRES'
    names[names == 'electricityCostHigh'] = 'electricityCost\navg'
    # title=namesX[x]
    bases = np.array(data[:, x, 1])
    base = bases[0]

    lows = np.array(data[:, x, 0])

    highs = np.array(data[:, x, 2])

    changeLows = (lows / bases - 1) * 100
    changeHighs = (highs / bases - 1) * 100
    sorts = np.absolute(highs - lows)
    sorts, lows, highs, names, bases, changeLows, changeHighs, varValues, varUnits = zip(
        *sorted(zip(sorts, lows, highs, names, bases, changeLows, changeHighs, varValues, varUnits), reverse=True))
    sorts, lows, highs, names, bases, changeLows, changeHighs, varValues, varUnits = sorts[:maxValues], lows[:maxValues], highs[:maxValues], names[:maxValues], bases[:maxValues], changeLows[:maxValues], changeHighs[:maxValues], varValues[:maxValues], varUnits[:maxValues]
    ###########################################################################
    # The actual drawing part

    # The y position for each variable
    ys = range(len(highs))[::-1]  # top to bottom
    
    if figSize==0:
        fig = plt.figure()
    else:
        fig=plt.figure(figsize=figSize)
     
    ###########################################################################
    # Images of the Supplychain
    imageDf = {}
    steps = [
        'Production',
        'Connector1',
        'Storage',
        'Connector2',
        'Transport',
        'Station']

    for i, step in zip(range(len(steps)), steps):
        imageDf[step] = mpimg.imread(imagepath + dfHSC[step][x] + '.png')
        technology = dfHSC[step][x]
        # imageDf[step]=mpimg.imread(path+system+'.png')
        sp = fig.add_subplot(6, 6, i + 1)
        imgplot = sp.imshow(imageDf[step])

        sp.get_xaxis().set_visible(False)
        sp.get_yaxis().set_visible(False)
        if technology == "Nothing2" or technology == "Nothing3":
            continue
#        if styleUsed == "\matplotlibrcEES.mplstyle":
#            sp.set_title(technology)
#        elif styleUsed == "\matplotlibrc.mplstyle":
#            sp.set_title(technology)     
    ###########################################################################
    
    axes = plt.subplot2grid((20, 20), (4, 1), colspan=17, rowspan=16)
    # Plot the bars, one by one
    for y, low, high, base, changeLow, changeHigh, varValue, varUnit in zip(
            ys, lows, highs, bases, changeLows, changeHighs, varValues, varUnits):
        # The width of the 'low' and 'high' pieces
        low_width = base - low
        high_width = high - base

        # Each bar is a "broken" horizontal bar chart
        axes.broken_barh(
            [(low, low_width), (base, high_width)],
            (y - 0.4, 0.8),
            # Try different colors if you like
            facecolors=[(81 / 255, 83 / 255, 86 / 255),
                        (0 / 255, 91 / 255, 130 / 255)],
            edgecolors=['black', 'black'],  # 'none',
            linewidth=1)

        strValue = str(varValue) + ' ' + varUnit
        plt.text(dataMax + 0.1, y, strValue,
                 va='center',
                 ha='left',
                 color='black')

        # Display the change as text
        z = 0.6
        if low_width > 0:
            if low_width > z:
                plt.text(base -
                         low_width /
                         2, y, str(np.around(changeLow, 1)) +
                         '%', va='center', ha='center', color='white')
            else:
                plt.text(low - z / 2, y, str(np.around(changeLow, 1)) + '%',
                         va='center',
                         ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',pad=0.1))
        else:
            if low_width < -z:
                plt.text(base -
                         low_width /
                         2, y, str(np.around(changeLow, 1)) +
                         '%', va='center', ha='center', color='white')
            else:
                plt.text(low + z / 2, y, str(np.around(changeLow, 1)) + '%',
                         va='center',
                         ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',pad=0.1))

        if high_width > 0:
            if high_width > z:
                plt.text(base +
                         high_width /
                         2, y, str(np.around(changeHigh, 1)) +
                         '%', va='center', ha='center', color='white')
            else:
                plt.text(high + z / 2, y, str(np.around(changeHigh, 1)) + '%',
                         va='center',
                         ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',pad=0.1))
        else:
            if high_width < -x:
                plt.text(base +
                         high_width /
                         2, y, str(np.around(changeHigh, 1)) +
                         '%', va='center', ha='center', color='white')
            else:
                plt.text(high - z / 2, y, str(np.around(changeHigh, 1)) + '%',
                         va='center',
                         ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',pad=0.1))

    # Draw a vertical line down the middle
    axes.axvline(base, color='black',linewidth=1)

    # Additional Information
    info = str(dist) + ' km distance\n' + \
        str(dem / 1000).rstrip('0').rstrip('.') + ' t/day demand'

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    # axes.xaxis.set_ticks_position('top')
    axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f €'))
    axes.set_xlabel('Total Cost of Hydrogen [€/kg]')
    #axes.xaxis.set_label_coords(0.5, 1.1)
    # Make the y-axis display the variables
    plt.grid()
    plt.yticks(ys, names)
    lowPatch = mpatches.Patch(
        color=(
            81 / 255,
            83 / 255,
            86 / 255),
        label='-20%')
    highPatch = mpatches.Patch(
        color=(
            0 / 255,
            91 / 255,
            130 / 255),
        label='+20%')
    infoPatch = mpatches.Patch(color='w', label=info)
    if styleUsed == "\matplotlibrcEES.mplstyle":
        if(dataMin + dataMax) / 2 > base:
            plt.legend(handles=[lowPatch, highPatch], loc=4)
        else:
            plt.legend(handles=[lowPatch, highPatch], loc=3)
    elif styleUsed == "\matplotlibrc.mplstyle":        
        if(dataMin + dataMax) / 2 > base:
            plt.legend(handles=[infoPatch, lowPatch, highPatch], loc=4)
        else:
            plt.legend(handles=[infoPatch, lowPatch, highPatch], loc=3)
    
    # Set the portion of the x- and y-axes to show
    axes.set_xlim([dataMin, dataMax])

    
    if saveFig==True:
        #plt.savefig(savePath+'Tornados\\'+saveName+str(x))
        png1=BytesIO()
        fig.savefig(png1,format='png')
        png2 = Image.open(png1)       
        png2.save(savePath+'Tornados\\'+saveName+str(x)+'.tiff')
        png1.close()
    plt.show()
    return names[0], names[1], names[2]
# %% horizontal stacking of energy demand


def stackedBarChart(
        data,
        listTechnologies,
        listEnergies,
        dist,
        dem,
        title='Energy demand of all investigated pathways',
        titleShow=False,
        labelTitle='Energy demand [kWh/kg]',
        legendCols=2,
        spaceLeft=1,
        spaceRight=0,
        spaceTop=1,
        spaceBottom=0,
        spaceCol=7,
        spaceRow=6,
        figSize=0,
        saveFig=False,
        savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
        saveName='FigureStackedBarChart'):
    '''
    Energy demand chart for different technologies
        data --> numpy array of to stacking data
        listTechnologies --> list of Technology/pathway names (y-axis)
        listEnergies --> list of energies/dates (legend)
        dist --> distance for transportation
        dem --> Demand of hydrogen
        title --> title of the plot
        titleShow --> Show title or not
        labelTitle --> x-axis label
        publicationType: 'journal', 'presentation', etc
    
    '''

    colors = ((0, 0.32549, 0.5098), (0.4196, 0.4196, 0.8117), (0.4039, 0.8196, 1),
              (0.318, 0.325, 0.333), (0.176, 0.176, 0.541), (0.25, 0.25, 0.25), (0, 0, 0))
    colorList = [(0, 0.32549, 0.5098), (0.4196, 0.4196, 0.8117), (0.4039, 0.8196, 1),
                 (0.318, 0.325, 0.333), (0.176, 0.176, 0.541), (0.25, 0.25, 0.25), (0, 0, 0)]
    nEnergies = len(listEnergies)
    
    colorsUsed = colors[:(nEnergies - 1)]
    ypos = np.arange(len(listTechnologies))
    barwidth = 0.8
    i_add=0
    
    
    if figSize==0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figSize)
        if figSize[1]<3.:
            i_add=1
    
    if styleUsed=="\matplotlibrcEES.mplstyle":    
        barChart = plt.subplot2grid((spaceRow+i_add, spaceCol),
                                    (spaceTop+i_add, spaceLeft),
                                     rowspan=spaceRow-spaceTop-spaceBottom,
                                     colspan=spaceCol-spaceLeft-spaceRight)
    elif styleUsed == "\matplotlibrc.mplstyle":
        barChart = plt.subplot2grid((5, 8), (0, 1), rowspan=5, colspan=5)

    axes = fig.gca()
    for i in range(len(listEnergies)):
        if i == 0:
            barChart.barh(-ypos,
                          data[:, i],
                          barwidth,
                          color=colorList[i],
                          edgecolor='none',
                          align='center',
                          label=listEnergies[i])
            dataLeft = data[:, i]
        else:
            barChart.barh(-ypos,
                          data[:, i],
                          barwidth,
                          left=dataLeft,
                          color=colorList[i],
                          edgecolor='none',
                          align='center',
                          label=listEnergies[i])
            dataLeft = dataLeft + data[:, i]

    dataMax = np.max(dataLeft) * 1.1
    ###########################################################################
    # Text and stuff
    plt.yticks(-ypos, listTechnologies)

    plt.xlabel(labelTitle)
    
    if titleShow==True:
        plt.title(title)
    
    if styleUsed=="\matplotlibrcEES.mplstyle":
    
        firstLeg = barChart.legend(bbox_to_anchor=(0, 1.01, 1, .0125),
                                   loc=3,
                                   ncol=legendCols,
                                   borderaxespad=0.,
                                   mode="expand")
    
    elif styleUsed == "\matplotlibrc.mplstyle":
        firstLeg = barChart.legend(bbox_to_anchor=(1.01,1),
                                   loc=legendCols,
                                   borderaxespad=0.)
        info = str(dist) + ' km distance\n' + \
            str(dem / 1000).rstrip('0').rstrip('.') + ' t/day demand'
        infoPatch = mpatches.Patch(color='w', label=info)
        axes.text(dataMax*1.05, -len(listTechnologies)+0.75,
                  info,
                  color='w',
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  bbox={'facecolor':(0/255, 91/255, 130/255),
                        'edgecolor':'None',
                        'alpha':1,
                        'pad':5})
                        
    axes.add_artist(firstLeg)
     
    axes.set_xlim([0, dataMax])
    axes.set_ylim([-len(listTechnologies) + 0.5, +0.5])
#    if styleUsed=="\matplotlibrcEES.mplstyle":
#        fig.tight_layout()
    
    if saveFig==True:
        #plt.savefig(savePath+saveName)
        png1=BytesIO()
        fig.savefig(png1,format='png', bbox_extra_artists=(firstLeg,))
        png2 = Image.open(png1)
#        trx=trim(png2,"white")
#        trx.save(savePath+saveName+'.tiff')    
        png2.save(savePath+saveName+'.tiff')
        png1.close()
    plt.show()
# %% horizontal stacking of energy demand


def wellToWheelTotal(
                dataCO2Own,
                dataTotalEnergyOwn,
                listTechnologiesOwn,
                listNamesOwn,
                dataCO2Other,
                dataTotalEnergyOther,
                listTechnologiesOther,
                listNamesOther,
                dist,
                dem,
                figSize=0,
                saveFig=False,
                savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
                saveName='FigureWellToWheel'):
    '''
    Energy demand chart for different technologies
        data --> numpy array of to stacking data
        listTechnologies --> list of Technology/pathway names (y-axis)
        listEnergies --> list of energies/dates (legend)
        dist --> distance for transportation
        dem --> Demand of hydrogen
        title --> title of the plot
        titleShow --> Show title or not
        labelTitle --> x-axis label
        info --> show some additional information or not 
    
    '''
    colorList = [(0., 0.359, 0.510),        # darkbluegreen
                 (0.4196, 0.4196, 0.8117),  # blue
                 (0.4039, 0.8196, 1),       # lightblue
                 (0.8196 ,0.4039, 1),       # lightblue?
                 (0.318, 0.325, 0.333),     # darkgrey
                 (0.619, 0.619, 0.619),     # lightgrey
                 (0.176, 0.176, 0.541),     # violet
                 (0.25, 0.25, 0.25),        # grey
                 (1., 0.3, 0.3),            # light red
                 (0.1, 0.510, 0.359),        # darkgreenblue
                 (0, 0, 0)]                 # black

    yposOwn = np.arange(len(listTechnologiesOwn))
    yposOther = np.arange(len(listTechnologiesOther))

    dataMaxCO2 = np.max([np.max(dataCO2Own),np.max(dataCO2Other)]) * 1.1
    dataMaxTotalEnergy = np.max([np.max(dataTotalEnergyOwn),np.max(dataTotalEnergyOther)]) * 1.1    
    if figSize==0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figSize)
    ###########################################################################
                # barplot left top #
    barChart = plt.subplot2grid((8, 8), (0, 0), rowspan=6, colspan=4)
    barChart.barh(-yposOwn,
                  dataTotalEnergyOwn[:, 0],
                  0.8,
                  color=colorList[2],
                  edgecolor='none',
                  align='center',
                  label=listNamesOwn[0])
    
    barChart.barh(-yposOwn,
                  dataTotalEnergyOwn[:, 1],
                  0.5,
                  color=colorList[7],
                  edgecolor='none',
                  align='center',
                  label=listNamesOwn[1])

    plt.yticks(-yposOwn, listTechnologiesOwn)
    barChart.set_xlim([0, dataMaxTotalEnergy])
    barChart.set_ylim([-len(listTechnologiesOwn)+0.5, +0.5])
    plt.setp(barChart.get_xticklabels(),visible=False)
    barChart.set_ylabel('Results of this study', size=9)
    barChart.set_title('Well-to-Wheels Analysis Total Energy Consumption')
    ###########################################################################
            # barplot left bot #
    barChart2 = plt.subplot2grid((8, 8), (6, 0), rowspan=2, colspan=4)
    barChart2.barh(-yposOther,
                   dataTotalEnergyOther[:, 0],
                   0.8,
                   color=colorList[9],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[0])
    
    barChart2.barh(-yposOther,
                   dataTotalEnergyOther[:, 1],
                   0.5,
                   color=colorList[7],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[1])
    
    plt.yticks(-yposOther, listTechnologiesOther)   
    barChart2.set_xlim([0, dataMaxTotalEnergy])
    barChart2.set_ylim([-len(listTechnologiesOther)+0.5, +0.5])
    barChart2.set_ylabel('Results JEC', size=9)                  
    barChart2.set_xlabel(r'Total Energy Consumption $\frac{MJ}{100 km}$')
    ###########################################################################
    # barplot rignt top #
    barChart3 = plt.subplot2grid((8, 8), (0, 4), rowspan=6, colspan=4)
    barChart3.barh(-yposOwn,
                   dataCO2Own[:, 0],
                   0.8,
                   color=colorList[2],
                   edgecolor='none',
                   align='center',
                   label=listNamesOwn[0])

    barChart3.barh(-yposOwn,
                   dataCO2Own[:, 1],
                   0.5,
                   color=colorList[7],
                   edgecolor='none',
                   align='center',
                   label=listNamesOwn[1])


    barChart3.set_xlim([0, dataMaxCO2])
    barChart3.set_ylim([-len(listTechnologiesOwn)+0.5, +0.5])
    barChart3.set_title('Well-to-Wheels Analysis CO2-Emissions')
    plt.setp(barChart3.get_xticklabels(),visible=False)
    plt.yticks(-yposOwn, listTechnologiesOwn)
    
    plt.legend()
    ###########################################################################
    # barplot rignt bot
    barChart4 = plt.subplot2grid((8, 8), (6, 4), rowspan=2, colspan=8)
    barChart4.barh(-yposOther,
                   dataCO2Other[:, 0],
                   0.8,
                   color=colorList[9],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[0])
    
    barChart4.barh(-yposOther,
                   dataCO2Other[:, 1],
                   0.5,
                   color=colorList[7],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[1])

    plt.yticks(-yposOther, listTechnologiesOther)
    barChart4.set_xlim([0, dataMaxCO2])
    barChart4.set_ylim([-len(listTechnologiesOther)+0.5, +0.5])
    barChart4.set_xlabel(r'CO2-Emissions in $\frac{kg_{CO2}}{km}$')
    plt.legend(loc=4, ncol=2)

    fig.tight_layout()
    if saveFig==True:
        #plt.savefig(savePath+saveName)
        png1=BytesIO()
        fig.savefig(png1, format='png')
        png2 = Image.open(png1)
        png2.save(savePath+saveName+'.tiff')
        png1.close()
    plt.show()
# %% horizontal stacking of energy demand


def wellToWheel(
                dataOwn,
                listTechnologiesOwn,
                listNamesOwn,
                dataOther,
                listTechnologiesOther,
                listNamesOther,
                dist,
                dem,
                title='Energy demand of all investigated pathways',
                titleShow=False,
                labelTitle='Energy demand [kWh/kg]',
                publicationType='journal',
                saveFig=False,
                savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
                saveName='FigureStackedBarChart'):
    '''
    Energy demand chart for different technologies
        data --> numpy array of to stacking data
        listTechnologies --> list of Technology/pathway names (y-axis)
        listEnergies --> list of energies/dates (legend)
        dist --> distance for transportation
        dem --> Demand of hydrogen
        title --> title of the plot
        titleShow --> Show title or not
        labelTitle --> x-axis label
        info --> show some additional information or not 
        publicationType: 'journal', 'presentation', etc
    
    '''
    colorList = [(0., 0.359, 0.510),        # darkbluegreen
                 (0.4196, 0.4196, 0.8117),  # blue
                 (0.4039, 0.8196, 1),       # lightblue
                 (0.8196 ,0.4039, 1),       # lightblue?
                 (0.318, 0.325, 0.333),     # darkgrey
                 (0.619, 0.619, 0.619),     # lightgrey
                 (0.176, 0.176, 0.541),     # violet
                 (0.25, 0.25, 0.25),        # grey
                 (1., 0.3, 0.3),            # light red
                 (0.1, 0.510, 0.359),        # darkgreenblue
                 (0, 0, 0)]                 # black

    yposOwn = np.arange(len(listTechnologiesOwn))
    yposOther = np.arange(len(listTechnologiesOther))

    dataMax = np.max([np.max(dataOwn),np.max(dataOther)]) * 1.1
    info = str(dist) + ' km distance\n' + \
            str(dem / 1000).rstrip('0').rstrip('.') + ' t/day demand'
    fig = plt.figure()
    if styleUsed=="\matplotlibrcEES.mplstyle":
        barChart = plt.subplot2grid((8, 7), (0, 0), rowspan=6, colspan=7)
        barChart2 = plt.subplot2grid((8, 7), (6, 0), rowspan=2, colspan=7)
    elif styleUsed=="\matplotlibrc.mplstyle":
        barChart = plt.subplot2grid((9, 7), (0, 0), rowspan=7, colspan=5)
        barChart2 = plt.subplot2grid((9, 7), (7, 0), rowspan=2, colspan=5)
    
    barChart.barh(-yposOwn,
                  dataOwn[:, 0],
                  0.8,
                  color=colorList[2],
                  edgecolor='none',
                  align='center',
                  label=listNamesOwn[0])

    barChart.barh(-yposOwn,
                  dataOwn[:, 1],
                  0.5,
                  color=colorList[7],
                  edgecolor='none',
                  align='center',
                  label=listNamesOwn[1])

    
    barChart.set_xlim([0, dataMax])
    barChart.set_ylim([-len(listTechnologiesOwn)+0.5, +0.5])
    barChart.set_yticks(-yposOwn)
    barChart.set_yticklabels(listTechnologiesOwn)
    plt.setp(barChart.get_xticklabels(),visible=False)
    
    if styleUsed == "\matplotlibrcEES.mplstyle":
        plt.legend()
    elif styleUsed == "\matplotlibrc.mplstyle":
        secLeg = barChart.legend(bbox_to_anchor=(1.01,1),
                                 loc=2,
                                 borderaxespad=0.)
        barChart.add_artist(secLeg)
        barChart.text(dataMax*1.05, -len(listTechnologiesOwn)+0.75,
               info,
               color='w',
               horizontalalignment='left',
               verticalalignment='bottom',
               bbox={'facecolor': (0/255, 91/255, 130/255),
                     'edgecolor': 'None',
                     'alpha': 1,
                     'pad': 5})
    ###########################################################################
    barChart2.barh(-yposOther,
                   dataOther[:, 0],
                   0.8,
                   color=colorList[9],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[0])

    barChart2.barh(-yposOther,
                   dataOther[:, 1],
                   0.5,
                   color=colorList[7],
                   edgecolor='none',
                   align='center',
                   label=listNamesOther[1])

    barChart2.set_xlim([0, dataMax])
    barChart2.set_ylim([-len(listTechnologiesOther)+0.5, +0.5])
    barChart2.set_yticks(-yposOther)
    barChart2.set_yticklabels(listTechnologiesOther)
    barChart2.set_xlabel(labelTitle)
    
    if styleUsed == "\matplotlibrcEES.mplstyle":
        plt.legend(loc=4)
    elif styleUsed == "\matplotlibrc.mplstyle":
        secLeg = barChart2.legend(bbox_to_anchor=(1.01,1),
                                   loc=2,
                                   borderaxespad=0.)
        barChart2.add_artist(secLeg)                           


    if titleShow==True:
        plt.title(title)
    fig.tight_layout()

    if saveFig==True:
        #plt.savefig(savePath+saveName)
        png1=BytesIO()
        fig.savefig(png1,format='png')
        png2 = Image.open(png1)
        png2.save(savePath+saveName+'.tiff')
        png1.close()    
    plt.show()

# %%
# horizontal bar chart emissions


def chartEmissions(
        data,
        listTechnologies,
        dist,
        dem,
        title='operative CO2 Emissions for different hydrogen supply chains',
        lableTitle='CO2-Emissions [kg_CO2/kg_H2]',
        saveFig=False,
        savePath='C:\\Alles\\Sciebo\\Python\\Trunk\\Images\\default\\',
        saveName='FigureStackedBarChart'):
    '''
    Emission chart for different technologies
    '''
    colors = [(0, 0.32549, 0.5098), (0.4196, 0.4196, 0.8117),
              (0.4039, 0.8196, 1), (0.318, 0.325, 0.333), (0.176, 0.176, 0.541)]
    ypos = np.arange(len(listTechnologies))
    barwidth = 0.8

    fig = plt.figure()
    barChart = plt.subplot2grid((1, 20), (0, 4), colspan=15)
    barChart.barh(ypos,
                  data[:],
                  barwidth,
                  color=colors[0],
                  edgecolor='none',
                  align='center')

    plt.yticks(ypos, listTechnologies)
    # plt.title(title)
    plt.xlabel(lableTitle)
    barChart.set_ylim([len(listTechnologies) - 0.5, -0.5])
    
    if saveFig==True:
        #plt.savefig(savePath+saveName)
        png1=BytesIO()
        fig.savefig(png1,format='png')
        png2 = Image.open(png1)
        png2.save(savePath+saveName+'.tiff')
        png1.close()
        
    plt.show()

#%% Plotting the HSC


def plotHSC(dfHSC, path):
    '''
    Plotting the Hydrogen Supply Chain with different pictograms
    '''

    steps = [
        'Production',
        'Connector1',
        'Storage',
        'Connector2',
        'Transport',
        'Station']
    for x in range(len(dfHSC)):
        fig = plt.figure()
        imageDf = {}
        ###Initialize and load the different images###

        #fig.subplots_adjust(hspace = .5, wspace=.001)
        # axs=axs.ravel()
        for i, step in zip(range(len(steps)), steps):
            imageDf[step] = mpimg.imread(path + dfHSC[step][x] + '.png')
            technology = dfHSC[step][x]
            # imageDf[step]=mpimg.imread(path+system+'.png')
            sp = fig.add_subplot(6, 6, i + 1)
            imgplot = plt.imshow(imageDf[step])
            sp.get_xaxis().set_visible(False)
            sp.get_yaxis().set_visible(False)
            if technology == "Nothing2" or technology == "Nothing3":
                continue
            sp.set_title(technology, fontsize=14)

    plt.show()
    # fig.tight_layout()
    
#%%
def _flatten_multi_geoms(geoms, colors):
    """
    Returns Series like geoms and colors, except that any Multi geometries
    are split into their components and colors are repeated for all component
    in the same Multi geometry.  Maintains 1:1 matching of geometry to color.

    "Colors" are treated opaquely and so can actually contain any values.

    Returns
    -------

    components : list of geometry

    component_colors : list of whatever type `colors` contains
    """
    components, component_colors = [], []

    # precondition, so zip can't short-circuit
    assert len(geoms) == len(colors)
    for geom, color in zip(geoms, colors):
        if geom.type.startswith('Multi'):
            for poly in geom:
                components.append(poly)
                # repeat same color for all components
                component_colors.append(color)
        else:
            components.append(geom)
            component_colors.append(color)
    return components, component_colors
#%%
def plot_polygon_collection(ax, geoms, colors_or_values, plot_values=False,
                            vmin=None, vmax=None, cmap=None,
                            edgecolor='black', alpha=0.5, linewidth=1.0,label="Area",norm=None,  **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    colors_or_values : a sequence of `N` values or RGBA tuples
        It should have 1:1 correspondence with the geometries (not their components).

    plot_values : bool
        If True, `colors_or_values` is interpreted as a list of values, and will
        be mapped to colors using vmin/vmax/cmap (which become required).
        Otherwise `colors_or_values` is interpreted as a list of colors.

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted
    """


    
    components, component_colors_or_values = _flatten_multi_geoms(
        geoms, colors_or_values)

    # PatchCollection does not accept some kwargs.
    collection = PatchCollection([PolygonPatch(poly) for poly in components],
                                 linewidth=linewidth, edgecolor=edgecolor,
                                 alpha=alpha,label=label, **kwargs)

    if plot_values:
        collection.set_array(np.array(component_colors_or_values))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
    else:
        # set_color magically sets the correct combination of facecolor and
        # edgecolor, based on collection type.
        collection.set_color(component_colors_or_values)

        # If the user set facecolor and/or edgecolor explicitly, the previous
        # call to set_color might have overridden it (remember, the 'color' may
        # have come from plot_series, not from the user). The user should be
        # able to override matplotlib's default behavior, by setting them again
        # after set_color.
        if 'facecolor' in kwargs:
            collection.set_facecolor(kwargs['facecolor'])
        if edgecolor:
            collection.set_edgecolor(edgecolor)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection
#%%
def plotLines(edges, ax, linewidth=1, color="black", linestyle="solid", zorder=1, alpha=1., label=False):
    seg=np.array([[(x,y) for x,y in line.coords] for line in edges.geometry])
    if label:
        ax.add_collection(LineCollection(seg,
                                     linewidths=linewidth,
                                     colors=color,
                                     linestyle=linestyle,
                                     zorder=zorder,
                                     alpha=alpha,
                                     label=label,
                                     antialiaseds=1, pickradius=10)
                          )
    else:
        ax.add_collection(LineCollection(seg,
                                     linewidths=linewidth,
                                     colors=color,
                                     linestyle=linestyle,
                                     zorder=zorder,
                                     alpha=alpha,
                                     antialiaseds=1, pickradius=10)
                          )
    x=[line.coords[0][0] for line in edges.geometry]
    y=[line.coords[0][1] for line in edges.geometry]
    ax.scatter(x,y,c=[color],s=linewidth**2,edgecolor='none', zorder=zorder,alpha=alpha)
#%%
def plotGpdLinesVar(gpd, colName, ax, rangeMax=10, maxLineWidth=5, minLineWidth=1, alpha=1, zorder=3, name="redToWhite",labelling=True, label="Lines"):
    '''
    plots the geopandas lines
    '''
    def getColor(iNorm, name):
        if name=="redToWhite":
            return (i_norm, 1-i_norm, 1-i_norm,0)
        elif name=="black":
            return (0,0,0)
        elif name=="redToBlack":
            return (i_norm, 0, 0)
        else:
            return name

    base=np.ceil(gpd[colName].max())/rangeMax
    if colName=="capacity":
        unit=" kt/a"
    elif colName=="diameter" or colName=="diameterMM":
        unit = " mm"
    elif colName=="nTruckPerDay":
        unit= "Tr per day"
    else:
        unit=""
    for i in range(rangeMax):
        i_norm=(i+1)/rangeMax
        if labelling:
            if i==0:
                labelI=label +"\n %.0f" % (base*i) +" - "+ "%.0f" % (base*(i+1))
            else:
                labelI="%.0f" % (base*i) +" - "+ "%.0f" % (base*(i+1))
            labelI=labelI+unit
            plotLines(gpd[gpd[colName].between(base*i, base*(i+1))],
                      ax=ax,
                      linewidth=(maxLineWidth-minLineWidth)*i_norm+minLineWidth,
                      color=getColor(i_norm, name),
                      #color=(i_norm, 0, 0),
                      alpha = alpha,
                      zorder=zorder,
                      label=labelI)
        else:
            plotLines(gpd[gpd[colName].between(base*i, base*(i+1))],
                      ax=ax,
                      linewidth=(maxLineWidth-minLineWidth)*i_norm+minLineWidth,
                      color=getColor(i_norm, name),
                      #color=(i_norm, 0, 0),
                      alpha = alpha,
                      zorder=zorder)

#%%
def plotGPDPoints(gpd, ax, colName=None, color="yellow", label="FuelingStation", marker=".", markersize=1, zorder=1, alpha=1, edgecolor="black", linewidth=0.5):
    if not isinstance(colName,type( None)):
        markersize=gpd[colName]
    testarray=np.array(sFun.point_array(gpd)).T
    ax.scatter(testarray[0],
            testarray[1],
            marker=marker,
            s=markersize,
            c=[color],
           label=label,
           zorder=zorder,
           alpha=alpha,
           edgecolor=edgecolor,
           linewidth=linewidth)  
#%%
def getSummedConnectorDF(df):
    dfCon=df.loc[["Con" in x for x in df.index]]
    df=df.loc[["Con" not in x for x in df.index]]
    df.loc["Connectors", :]=dfCon.sum()
    return df

def createPlotsPostProcessing(Results=False, savePath=False, addProduction=False, show=False):
    '''
    creating all Plots from HSCResults File
    
    '''
    from collections import OrderedDict
    postProcResults=OrderedDict()
    for col in Results[0].dfHSCRes.columns:
        postProcResults[col]=pd.DataFrame(index=Results[0].dfHSCRes.index)
    
    
    for key, pathHSC in Results.items():
        for col in pathHSC.dfHSCRes.columns:
            postProcResults[col][key]=pathHSC.dfHSCRes[col].values

    technologyDF=pd.DataFrame(index=Results[0].dfHSCRes.index)
    for key, HSC in Results.items():
        technologyDF[key]=HSC.dfHSCRes["technology"].values
    if addProduction: p=0
    else: p=1
    techShort=technologyDF.loc[["Con" not in x for x in technologyDF.index]].iloc[p:-1]
    plotBarWithIcons(techShort, postProcResults["TOTEX"],
                     savePath=savePath,
                     saveName="CostResultsPathways", show=show)
    plotBarWithIcons(techShort, postProcResults["CO2Emissions[kg/kg]"],
                     xlabel=r"CO$_2$ Emissions [kg$_{CO2}$/kg$_{H2}$]",
                     savePath=savePath,
                     saveName="CO2ResultsPathways", show=show)
    plotBarWithIcons(techShort, postProcResults["primaryEnergy[MJ/kg]"],
                     xlabel=r"Primary Energy Demand [MJ/kg$_{H2}$]",
                     savePath=savePath,
                     saveName="PEResultsPathways", show=show)
    

#%%
       
def plotIconsBottom(fig, imagepath, techdF, height, width, rows):
    imageDf={}
    for col in techdF.columns:
        steps = techdF[col].values
        totNum=len(steps)
        totTech=len(techdF.index)
        for i, step in zip(range(totNum), steps):
            imageDf[step] = mpimg.imread(path.join(imagepath, step + '.png'))
            if step=="None":
                continue
            sp = fig.add_subplot(height, width, width*(i+rows*totTech)+int(float(col)) + 1)
            imgplot = sp.imshow(imageDf[step])

            sp.get_xaxis().set_visible(False)
            sp.get_yaxis().set_visible(False)

def plotIconsLeft(fig, imagepath, techdF, height, width, rows):
    imageDf={}
    for col in techdF.columns:
        steps = techdF[col].values
        totNum=len(steps)
        totTech=len(techdF.index)
        for i, step in zip(range(totNum), steps):
            imageDf[step] = mpimg.imread(path.join(imagepath, step + '.png'))
            if step=="None":
                continue
            sp = fig.add_subplot(height, width, height*(i+rows*totTech)+int(float(col)) + 1)
            imgplot = sp.imshow(imageDf[step])

            sp.get_xaxis().set_visible(False)
            sp.get_yaxis().set_visible(False)  
#%%            
def plotBarWithIcons(techDf, plotDf,
                     xlabel="Hydrogen costs at dispenser [EUR/kg$_{H2}$]",
                     savePath=False,
                     saveName="ResultsPathways",
                     stacked=True,
                     ylim=False,
                     show=False):
    '''
    plotting barchart with icons as legend
    '''    
    rows=1
    width=len(techDf.columns)
    height=len(techDf.index)*(rows+1)
    fig=plt.figure(figsize=(max(width, 10),height))
    ax=plt.subplot2grid((rows+1, 1), (0, 0), rowspan=rows, colspan=1)
    
    plotDf.T.plot(kind="bar", stacked=stacked, ax=ax)

    ax.set_ylabel(xlabel)
    ax.get_xaxis().set_visible(False)
    if ylim:
        ax.set_ylim([0, ylim])
    elif stacked:
        ax.set_ylim([0, plotDf.sum().max()*1.1])
    bars = ax.patches
    hatches = ''.join(h*len(plotDf.index) for h in '      //////******OOOOOO......')
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(bbox_to_anchor=(-0., 1.02, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.)    
    if stacked:
        for x,y in plotDf.sum().items():
            ax.annotate("%1.2f" %(y), (x, y), size=12, ha="center", va="bottom")
    
    plotIconsBottom(fig, IMAGEPATH, techDf, height, width, rows)
    
    if savePath:
        fig.savefig(path.join(savePath,saveName), bbox_inches="tight")
    if show:   
        plt.show()
    else:
        plt.close()
#%%
def plotGISResultsTriplePlot(
    area, 
    pipeT,
    pipeD,
    truckT,
    truckD,
    hub,
    source,
    fuel,
    truckColor=colorList[0],
    pipeColor="black",
    colorList=colorList):
    '''
    Plot 
    '''
    fig, axes=plt.subplots(1,4, figsize=(12,4))
    #axes=[axes[0][0],axes[0][1],axes[1][0],axes[1][1]]
    testarray=np.array(sFun.point_array(fuel)).T
    unary=gpd.GeoSeries(area.unary_union)
    for n in range(3):
        unary.plot(color=bg_area, edgecolor=bg_lines, ax=axes[n], alpha=1)
        axes[n].set_ylim([area.total_bounds[1], area.total_bounds[3]])
        axes[n].set_xlim([area.total_bounds[0], area.total_bounds[2]])
        axes[n].set_yticklabels(["" for x in axes[n].get_yticks()])
        axes[n].axis("off")
        axes[n].set_aspect("equal")

    sourceMarkerMax=80
    sourceMarkerMin=10

    source["markersizePipe"]=((source["pipe_kt_a"]-source["pipe_kt_a"].min())/source["pipe_kt_a"].max()*(sourceMarkerMax-sourceMarkerMin)+sourceMarkerMin)
    source["markersizeTruck"]=((source["truck_kt_a"]-source["truck_kt_a"].min())/source["truck_kt_a"].max()*(sourceMarkerMax-sourceMarkerMin)+sourceMarkerMin)
    plotGPDPoints(source,axes[0],
                           colName="markersizePipe",
                           color=(178/255, 223/255, 138/255),
                           zorder=5,
                           marker="D",
                           label="Elektrolyseur\n %.1f" % source["pipe_kt_a"].min() + " - %.1f" % source["pipe_kt_a"].max() + "kt/a")
    plotGPDPoints(source,axes[1],
                           colName="markersizePipe",
                           color=(178/255, 223/255, 138/255),
                           zorder=5,
                           marker="D",
                           label="Electrolyzer\n %.1f" % source["pipe_kt_a"].min() + " - %.1f" % source["pipe_kt_a"].max() + "kt/a")

    plotGPDPoints(source,axes[2],
                       colName="markersizeTruck",
                       color=(178/255, 223/255, 138/255),
                       zorder=5,
                       marker="D",
                       label="Electrolyzer\n %.1f" % source["truck_kt_a"].min() + " - %.1f" % source["truck_kt_a"].max() + "kt/a")
    plotGPDPoints(hub,
                       axes[1],
                       color="white",
                       zorder=6,
                       markersize=8,
                       marker="^",
                       label="Hubs")  

    lineWidthMax=[5,0.5]
    lineWidthMin=[0.5,0.5]
    colorStyle=["redToWhite", "black"]
    maxRange=[5,1]
    pipeLabel=["Transmission Pipeline", "Distribution Pipeline"]
    truckLabel=["Truck Routes", "Truck Routes (Distribution)"]
    pipeT["diameterMM"]=pipeT["diameter"].values*1000
    pipeD["diameterMM"]=pipeD["diameter"].values*1000
    
    plotGpdLinesVar(pipeT,
                         "diameterMM",
                         axes[0],
                         zorder=4,
                         alpha=1,
                         name=pipeColor,
                         rangeMax=maxRange[0],
                         maxLineWidth=3,
                         minLineWidth=0.2,
                         label="Transmissionspipeline")

    plotGpdLinesVar(pipeT,
                         "diameterMM",
                         axes[1],
                         zorder=4,
                         alpha=1,
                         name=pipeColor,
                         rangeMax=maxRange[0],
                         maxLineWidth=3,
                         minLineWidth=0.2,
                         labelling=False)

    plotGpdLinesVar(pipeD,
                         "diameterMM",
                         axes[0],
                         zorder=3,
                         alpha=1,
                         name=pipeColor,
                         rangeMax=maxRange[1],
                         maxLineWidth=0.2,
                         minLineWidth=0.1,
                         label="Distributionspipeline")

    plotGpdLinesVar(truckD,
                         "capacity",
                         axes[1],
                         zorder=3,
                         alpha=1,
                         name=truckColor,
                         rangeMax=maxRange[1],
                         maxLineWidth=0.2,
                         minLineWidth=0.1,
                         label="Truck Routen Distribution")

    plotGpdLinesVar(truckT,
                         "capacity",
                         axes[2],
                         zorder=4,
                         alpha=1,
                         name=truckColor,
                         rangeMax=5,
                         maxLineWidth=3,
                         minLineWidth=0.2,
                         label="Truck Routen")#

    fig.subplots_adjust(top=1,
                        right = 1,
                        bottom = 0.,
                        wspace = 0.0 ,
                        hspace = 0.0)

    handles, labels=axes[0].get_legend_handles_labels()
    handles1, labels1=axes[1].get_legend_handles_labels()
    handles2, labels2=axes[2].get_legend_handles_labels()
    handlesLeg=handles+handles1[1:]+handles2[1:]
    labelsLeg=labels+labels1[1:]+labels2[1:]

    axes[3].legend(handlesLeg, labelsLeg, bbox_to_anchor=(0,0,1,1),loc=5, mode="expand", fontsize=10, frameon=False)

    axes[3].axis("off")
    return axes