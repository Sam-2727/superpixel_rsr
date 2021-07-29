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
class SuperPixel():
    def __init__(self,lon,lat,surf_amp):
        self.lon = lon
        self.lat = lat
        #stored as log surfAmp
        self.surfAmp = surf_amp
    '''
    numlon and numlat are the number of grid points along the longitude and latitude directions, respectively
    '''
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
    def in_hull(self,p, hull):
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0
    def calcRSR(self):
        lonRavel = np.ravel(self.xgrid)
        latRavel = np.ravel(self.ygrid)
        supixRavel  = np.ravel(self.superpixels)
        for i in range(0,int(np.max(self.superpixels))+1):
            supixCondition  = (supixRavel==i)
            possiblePointsCondition = (self.lon<np.max(lonRavel[supixCondition]))&(self.lon>np.min(lonRavel[supixCondition]))\
                &(self.lat<np.max(latRavel[supixCondition]))&(self.lat>np.min(latRavel[supixCondition]))
            pointsPossible = np.vstack([self.lon[possiblePointsCondition],self.lat[possiblePointsCondition]]).T
            alpha_shape = alphashape.alphashape(supixPoints, 100)
            boolContains = vectorized.conatains(alpha_shape,pointsPossible[:,0],pointsPossible[:,1])
            if(len(pointsPossible)/len(self.lon)>0.5):
                print("skipped superpixel {0} -- too large".format(i))
                continue
            boolContains = vectorized.contains(alpha_shape,pointsPossible[:,0],pointsPossible[:,1])
            supixPoints = np.vstack([lonRavel[supixCondition],latRavel[supixCondition]]).T
            lenSurfAmp = len(np.array(self.surfAmp)[possiblePointsCondition][boolContains])
            if(lenSurfAmp<100):
                print("only {0} points in superpixel {1},skipping!".format(lenSurfAmp,i))
                continue
            rsrReturn = rsr.run.processor(self.surfAmp[possiblePointsCondition][boolContains], fit_model='hk')
            return rsrReturn
    def plotSuperpixelsPower(self,nSuperpixels,xlim,ylim):
        plt.imshow(self.supixObject.average(nSuperpixels, 1, self.zgrid),origin="lower")
