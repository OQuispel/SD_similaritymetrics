import pontiPy as ppy
import pandas as pd
import multiprocessing.pool as mpp
import pontius as pont


def calc_tquant(cm):
    """This function checks if the passed confusion matrix has any quantity disagreement"""
    if cm.quantity() > 0:
        return 1
    else:
        return 0
    
def calc_tquant2(cm):
    """This function checks if the passed confusion matrix has any quantity disagreement
    This function can run with the stored ponti objects by returning them to regular DFs"""
    cm = redf(cm)
    if cm.quantity() > 0:
        return 1
    else:
        return 0
    
def calc_maxrow(cm):
    """This functions extracts the max row/col disagreement for a map comparison and returns the respective classes"""
    df = cm.matrix()
    df['Row Disagreement'] = pd.to_numeric(df['Row Disagreement'])
    c1 = df['Row Disagreement'][0:26].idxmax()
    c2 = df.loc['Column Disagreement'][0:26].idxmax()
    return c1, c2

def calc_classexchange(cm, cat):
    """This function requires a confusion matrix of 2 maps and a land-use class then returns the land-use class with which it exchanges cells the most"""
    cm = cm.exchange(cat)
    #Remove total value to make max run only on classes
    del cm['Total Exchange']
    exch_max = max(cm, key=cm.get)
    exch_val = cm[exch_max]
    return exch_max, exch_val

def calc_exchangetwo(cm):
    """Calculate exchange disagreement between 'Nature' and 'Pasturage' land use classes."""
    val = cm.exchange(22,19)
    return val

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def redf(class_ppy):
    """Turns a pontiPy DF back into a DF using """
    df = class_ppy.matrix()
    df =df.drop(index=['Sum', 'Column Disagreement'], columns=['Sum', 'Row Disagreement'])
    df = pont.pontiPy(df)
    return df