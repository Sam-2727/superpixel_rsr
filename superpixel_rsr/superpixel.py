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
class SuperPixel():
    def __init__(self,lon,lat,surf_amp):
        self.lon = lon
        self.lat = lat
        #stored as log surfAmp
        self.surfAmp = surf_amp
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
    """         
    def calcRSR(self,parallelUnits):
        self.rsrReturns = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        self.supixPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        self.lonPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        self.latPointsReturn = np.zeros(int(np.max(self.superpixels))+1,dtype=object)
        _ = Parallel(parallelUnits)(delayed(self.calcRSRInternal)(i) for i in range(0,int(np.max(self.superpixels))+1))
        return [np.vstack((self.lonPointsReturn,self.latPointsReturn)),self.rsrReturns]
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
        self.supixPointsReturn[i] = supixPoints
        lenSurfAmp = len(np.array(self.surfAmp)[possiblePointsCondition][boolContains])
        Parallel()
        if(lenSurfAmp<100):
            print("only {0} points in superpixel {1},skipping!".format(lenSurfAmp,i))
            return np.nan
        self.rsrReturns[i] = (rsr.run.processor(self.surfAmp[possiblePointsCondition][boolContains], fit_model='hk'))
        self.lonPointsReturn[i] = self.lon[possiblePointsCondition][boolContains]
        self.latPointsReturn[i] = self.lat[possiblePointsCondition][boolContains]
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
    """            
    def resultPlotGrid(self,observable,size=False,xlim=False,ylim=False):
        plotx = np.array(0)
        ploty = np.array(0)
        plotz = np.array(0)
        for i in range(0,len(self.supixPointsReturn)):
            if(type(self.supixPointsReturn[i])!=int):
                plotx = np.concatenate((plotx, self.supixPointsReturn[i][:,0]),axis=None)
                ploty = np.concatenate((ploty, self.supixPointsReturn[i][:,1]),axis=None)    
                plotz = np.concatenate((plotz,np.repeat(self.rsrReturns[i].power()[observable],len(self.supixPointsReturn[i][:,0]))),axis=None)
        plotx = plotx[1:]
        ploty = ploty[1:]
        plotz = plotz[1:]
        if(size):
            plot = plt.scatter(plotx,ploty,c=plotz,s=size)
        else:
            plot = plt.scatter(plotx,ploty,c=plotz)
        plt.colorbar(plot)
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)
    def resultPlotData(self,observable,size=False,xlim=False,ylim=False):
        plotx = np.array(0)
        ploty = np.array(0)
        plotz = np.array(0)
        for i in range(0,len(self.supixPointsReturn)):
            if(type(self.supixPointsReturn[i])!=int):
                plotx = np.concatenate((plotx, self.lonPointsReturn[i]),axis=None)
                ploty = np.concatenate((ploty, self.latPointsReturn[i]),axis=None)    
                plotz = np.concatenate((plotz,np.repeat(self.rsrReturns[i].power()[observable],len(self.lonPointsReturn[i]))),axis=None)
        plotx = plotx[1:]
        ploty = ploty[1:]
        plotz = plotz[1:]
        if(size):
            plot = plt.scatter(plotx,ploty,c=plotz,s=size)
        else:
            plot = plt.scatter(plotx,ploty,c=plotz)
        plt.colorbar(plot)
        if(xlim):
            plt.xlim(xlim)
        if(ylim):
            plt.ylim(ylim)            