## Packages Required
You will need the following non-standard packages:
* Shapely (pip install shapely)
* boruvka superpixel (download here: https://github.com/semiquark1/boruvka-superpixel)
* alphashape (pip install alphashape)

## Sample code to get you started:

```python

import pandas as pd
powerReturns = pd.read_hdf("/west_alba_mons.srf.h5")
booleanCriteria= (powerReturns["SUB_SC_EAST_LONGITUDE"]<200)&(powerReturns["SUB_SC_PLANETOCENTRIC_LATITUDE"]<40)\
                &(powerReturns["SUB_SC_PLANETOCENTRIC_LATITUDE"]>30)&(powerReturns["SUB_SC_EAST_LONGITUDE"]>190)
sub = powerReturns[booleanCriteria]
from superpixel_rsr.superpixel import SuperPixel
superpixel = SuperPixel(sub["SUB_SC_EAST_LONGITUDE"],sub["SUB_SC_PLANETOCENTRIC_LATITUDE"],sub['surf_amp'])
superpixel.gridAmp(1000,1000)
superpixel.calcSuperpixel(200)
superpixel.calcRSR()

```
