import timeit
import numpy as np
from multiprocessing import Pool
#import multiprocessing.pool as mpp
#from tqdm import tqdm

def numpy2asc(map_id, xll=10000.000000, yll=300000.000000, cs=100.000000, ndv=-32768
              , nan2num=False, delimiter=' '):
    """This function converts a numpy array into an asc file with header data"""
    dataname = 'C:/LUMOS/MCK/csvmaps/' + 'map500exp_' + str(map_id) + '.csv'
    ndarray = np.genfromtxt(dataname, delimiter=',')
    filename = 'map' + str(map_id) + '.asc'
    savedir = "C:/LUMOS/MCK/ascmaps2/"
    with open(savedir + '\\' + filename, 'w') as fh:

        # write header
        nrows, ncols = ndarray.shape
        header = '\n'.join(["ncols " + str(ncols),
                            "nrows " + str(nrows),
                            'xllcorner ' + str(xll),
                            'yllcorner ' + str(yll),
                            'cellsize ' + str(cs),
                            'NODATA_value ' + str(ndv),
                            ''])
        fh.write(header)

        # write each row in ndarray
        fmt = "%.6e"
        fmti = "%i"
        for row in ndarray:
            if nan2num:
                row[np.isnan(row)] = ndv
            line = delimiter.join(str(fmti % value) if value % 1 == 0 else str(fmt % value) for value in row)
            fh.write(line + '\n')

        return map_id

if __name__ == '__main__':
    start = timeit.default_timer()
    iterable = [i for i in range(100)]
    with Pool(16) as pool:
        pool.map(numpy2asc, iterable)

    stop = timeit.default_timer()
    print(stop - start)