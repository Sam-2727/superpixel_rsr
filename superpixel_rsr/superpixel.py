import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import boruvka_superpixel
import matplotlib.pyplot as plt
import rsr
import alphashape
from shapely.geometry import Point
from shapely import vectorized
from joblib import Parallel, delayed
import os
class SuperPixel():
    def __init__(self,lon,lat,surf_amp):
        self.lon = lon
        self.lat = lat
        #stored as log surfAmp
        self.surfAmp = surf_amp
        self.resultPlotGridLoaded = False
        self.resultPlotDataLoaded = False
    """
    Function to grid surface amplitude based on user specified spacing. Stores gridded longitude in xgrid,
    gridded latitued in ygrid, and gridded surface amplitude in zgrid.

    Parameters
    ----------
    numlon : int
        number of grid points along longitude direction
    numlat : int
        number of grid points along latitude direction
    """
    def gridAmp(self,numlon,numlat):
        # Interpolation (non-spherical, for now)
        xgrid, ygrid = np.meshgrid(np.linspace(np.min(self.lon), np.max(self.lon), numlon)\
                    , np.linspace(np.min(self.lat),np.max(self.lat), numlat))
        zgrid = (griddata(np.vstack((self.lon, self.lat)).T, np.log(self.surfAmp), \
                          (xgrid, ygrid), method="linear"))
        #setting nan values as 0 for now. Is there a better way to do this?
        zgrid[np.isnan(zgrid)]=0
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.zgrid = zgrid
    """
    Function to calculate superpixels over region. Results can be plotted with plotSuperpixel() function.

    Parameters
    ----------
    nuSuperpixels : int
        number of superpixels to use in calculation.
    """        
    def calcSuperpixel(self,nSuperpixels):
        superpixel = boruvka_superpixel.BoruvkaSuperpixel()
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        img_edge = np.zeros((self.zgrid.shape[:2]), dtype=self.zgrid.dtype)
        bosupix.build_2d(self.zgrid, img_edge)
        self.supixObject = bosupix
        out = bosupix.average(nSuperpixels, 1, self.zgrid)
        uniqueRegions = np.unique(out.copy())
        for i in range(0,nSuperpixels):
            out[out==uniqueRegions[i]]=i
        self.superpixels = out
        self.nsupix = nSuperpixels
    #aim for 10000 per superpixel
    # def in_hull(self,p, hull):
    #     if not isinstance(hull,Delaunay):
    #         hull = Delaunay(hull)

        # return hull.find_simplex(p)>=0
    """
    Function to apply RSR method over each superpixel. Warning: this is the most computationally intensive step.
    superpixel grids can be accessed with <object name>.supixPointsReturn and points in each superpixel can be accessed with
    <object name>.lonPointsReturn and <object name>.latPointsReturns

    Parameters
    ----------
    parallelUnits : int
        number of threads to create for parallel execution of code.
    Returns
    ----------
    length 2 list. First component is an array of longitude and latitude points, sorted by superpixel.
    These can be accessed through an array call: returnValue[0][i][x] where i is the ith superpixel,
    x=0 is longitude, and x=1 is latitude
    
    Second component is the rsrReturns corresponding to each superpixel
    """         
    def calcRSR(self,filename="rsrResultObjects",startingSupix=0):
        self.rsrReturns = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        try:
            os.mkdir(filename)
        except:
            print("Assuming that folder {0} exists in current directory".format(filename))
            pass
        self.filename = filename
        # self.supixPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        # self.lonPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        # self.latPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        self.totalCount = 0
        for i in range(startingSupix,int(np.max(self.superpixels)+1)):
            self.calcRSRInternal(i)
        #_ = Parallel(parallelUnits)(delayed(self.calcRSRInternal)(i) for i in range(0,int(np.max(self.superpixels))+1))
        return None
    """
    Internal method for parallel execution of RSR on superpixels, shouldn't be called.
    """           
    def calcRSRInternal(self,i):
        print(i)
        lonRavel = np.ravel(self.xgrid)
        latRavel = np.ravel(self.ygrid)
        supixRavel  = np.ravel(self.superpixels)
        supixCondition  = (supixRavel==i)
        possiblePointsCondition = (self.lon<np.max(lonRavel[supixCondition]))&(self.lon>np.min(lonRavel[supixCondition]))\
            &(self.lat<np.max(latRavel[supixCondition]))&(self.lat>np.min(latRavel[supixCondition]))
        pointsPossible = np.vstack([self.lon[possiblePointsCondition],self.lat[possiblePointsCondition]]).T
        if(len(pointsPossible)/len(self.lon)>0.5):
            print("skipped superpixel {0} -- too large".format(i))
            return np.nan
        supixPoints = np.vstack([lonRavel[supixCondition],latRavel[supixCondition]]).T
        alpha_shape = alphashape.alphashape(supixPoints, 100)
        boolContains = vectorized.contains(alpha_shape,pointsPossible[:,0],pointsPossible[:,1])
        #self.supixPointsReturn[i] = supixPoints
        np.save(self.filename + "/supixPoints{0}".format(i),supixPoints)
        lenSurfAmp = len(np.array(self.surfAmp)[possiblePointsCondition][boolContains])
        if(lenSurfAmp<100):
            print("only {0} points in superpixel {1},skipping!".format(lenSurfAmp,i))
            return np.nan
        np.save(self.filename + "/rsrReturns{0}".format(i),(rsr.run.processor(self.surfAmp[possiblePointsCondition][boolContains], fit_model='hk')))
        np.save(self.filename + "/lonPointsReturn{0}".format(i),np.array(self.lon[possiblePointsCondition][boolContains]))
        np.save(self.filename + "/latPointsReturn{0}".format(i),np.array(self.lat[possiblePointsCondition][boolContains]))
        self.totalCount+=1
        return None
        #self.rsrReturns[i] = 
        #self.lonPointsReturn[i] = 
        #self.latPointsReturn[i] = 
    """
    Plot superpixels over region (region is gridded power amplitude).

    Parameters
    ----------
    nSuperpixels : int
        Number of superpixels to plot.
    xlim: length-2 list
        xlimit of plot, optional. Ex: [0,10]
    ylim: length-2 list
        ylimit of plot, optional. Ex: [0,10]        
    """           
    def plotSuperpixelsPower(self,nSuperpixels,xlim=False,ylim=False):
        plt.imshow(self.supixObject.average(nSuperpixels, 1, self.zgrid),origin="lower")
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)
    """
    Plot results of applying RSR method over superpixels onto grid.

    Parameters
    ----------
    observable: str
        RSR parameter to plot (ex: "pc-pn")
    size : int
        size of points that are plotted, optional.
    xlim: length-2 list
        xlimit of plot, optional. Ex: [0,10]
    ylim: length-2 list
        ylimit of plot, optional. Ex: [0,10]
    forceReload:
        results are loaded and stored so replotting is faster, 
        but the data can be reloaded if data is changed (or plotting from a different folder)
        by setting this to True (default False), optional.
    filename:
        folder to load data from (default is folder written to by 
        most recent calcRSR call), optional.
    """            
    def resultPlotGrid(self,observable,size=False,xlim=False,ylim=False,forceReload=False,filename=False):
        if(filename):
            self.filename = filename
        if((not self.resultPlotGridLoaded ) or forceReload):
            plotx = np.array(0)
            ploty = np.array(0)
            plotLoad = np.array(0)
            totalCount = 0
            for i in range(1,self.nsupix):
                try:
                    supixPointsReturn = np.load(self.filename + "/supixPoints{0}.npy".format(i),allow_pickle=True)
                    rsrReturns = np.load(self.filename + "/rsrReturns{0}.npy".format(i),allow_pickle=True).item()
                    plotx = np.concatenate((plotx, supixPointsReturn[:,0]),axis=None)
                    ploty = np.concatenate((ploty, supixPointsReturn[:,1]),axis=None)    
                    plotLoad = np.concatenate((plotLoad,np.repeat(rsrReturns.power(),len(supixPointsReturn[:,0]))),axis=None)
                except(FileNotFoundError):
                    print("Following superpixel index doesn't exist (likely doesn't contain enough points -- consider lowering number of superpixels?):", i)
            plotx = plotx[1:]
            ploty = ploty[1:]
            plotLoad = plotLoad[1:]
            self.gridPlot = plotLoad
            self.gridX = plotx
            self.gridY = ploty

        plotz = []
        for i in self.gridPlot:
            plotz.append(i[observable])
        if(size):
            plot = plt.scatter(self.gridX,self.gridY,c=plotz,s=size)
        else:
            plot = plt.scatter(self.gridX,self.gridY,c=plotz)
        plt.colorbar(plot)
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)
    """
    Clears data used for making of grid plot. Useful if you want 
    to clear up memory and don't need to replot grid soon.
    """        
    def clearGrid(self):
        self.gridX = None
        self.gridY = None
        self.gridPlot = None
    def resultPlotData(self,observable,size=False,xlim=False,ylim=False,forceReload=False,filename=False):
        if(filename):
            self.filename = filename
        if((not self.resultPlotDataLoaded) or forceReload):
            plotx = np.array(0)
            ploty = np.array(0)
            plotLoad = np.array(0)
            for i in range(1,self.nsupix):
                try:
                    supixPointsReturn = np.load(self.filename + "/supixPoints{0}.npy".format(i),allow_pickle=True)
                    lonPointsReturn = np.load(self.filename + "/lonPointsReturn{0}.npy".format(i),allow_pickle=True)
                    latPointsReturn = np.load(self.filename + "/latPointsReturn{0}.npy".format(i),allow_pickle=True)
                    rsrReturns = rsrReturns = np.load(self.filename + "/rsrReturns{0}.npy".format(i),allow_pickle=True).item()
                    plotx = np.concatenate((plotx, lonPointsReturn),axis=None)
                    ploty = np.concatenate((ploty, latPointsReturn),axis=None)    
                    plotLoad = np.concatenate((plotLoad,np.repeat(rsrReturns.power(),len(lonPointsReturn))),axis=None)
                except(FileNotFoundError):
                    print("Following superpixel index doesn't exist (likely doesn't contain enough points -- consider lowering number of superpixels?):", i)
            plotx = plotx[1:]
            ploty = ploty[1:]
            plotLoad = plotLoad[1:]
            self.dataPlot = plotLoad
            self.dataX = plotx
            self.dataY = ploty
        plotz = []
        for i in self.dataPlot:
            plotz.append(i[observable])

        if(size):
            plot = plt.scatter(self.dataX,self.dataY,c=plotz,s=size)
        else:
            plot = plt.scatter(self.dataX,self.dataY,c=plotz,s=1)
        plt.colorbar(plot)
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)
    """
    Clears data used for making of data plot. Useful if you want 
    to clear up memory and don't need to replot data soon.
    """ 
    def clearData(self):
        self.dataX = None
        self.dataY = None
        self.dataPlot = None