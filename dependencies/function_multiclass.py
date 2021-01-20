import numpy as np
import pandas as pd
import pontius as pont
import multiprocessing.pool as mpp
import scipy.spatial.distance as ssd
"""Import extra functions for the sklearn sections of the code"""
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn .preprocessing import LabelBinarizer
from sklearn .preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
#from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning

def gen_cms(pair_id):
    """Test function to generate some CMs to analyze"""
    map1 = pontius_pairs[pair_id][0]
    map2 = pontius_pairs[pair_id][1]
    m1_dir = abs_dir + 'ascmaps/' + map1 + '.asc'
    m2_dir = abs_dir + 'ascmaps/' + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    
    rows_cols = ['class' + str(i + 1) for i in range(28)]
    rows_cols.remove('class24')
    rows_cols.remove('class28')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    df_pont = pont.pontiPy(df)
    return df_pont

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def calc_multi(pair_id, class1=22):
    """This function is the basis for calculating Pontius' metrics (2014), starting off by computing the confusion matrix for two provided maps and 
    ending with the Total Disagreement if it is assigned to be calculated."""
    map1 = pontius_pairs[pair_id][0]
    map2 = pontius_pairs[pair_id][1]
    m1_dir = abs_dir + 'ascmaps/' + map1 + '.asc'
    m2_dir = abs_dir + 'ascmaps/' + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    # turn array into df so that ppy can work with it
    rows_cols = ['class' + str(i) for i in range(28)]
    rows_cols.remove('class23')
    rows_cols.remove('class27')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    #create object to run ppy code
    df_pont = pont.pontiPy(df)
    #calculate overall accuracy
    oa = calc_oa(df_pont)
    #calculate pontius metrics
    od, oqd, oad = calc_pontius(df_pont)
    #calculate categorical overall quantity difference
    oqd_n = calc_oqd_n(df_pont, class1)
    return map1, map2, oa, od, oqd, oad, oqd_n

def calc_classes(pair_id, class1=22):
    """This function goes through all steps to turn a pair of maps into metric values for PLAND, TCA, Allocation Disagreement and Exchange Disagreement"""
    map1 = pontius_list[pair_id][0]
    map2 = pontius_list[pair_id][1]
    m1_dir = map_dir + map1 + '.asc'
    m2_dir = map_dir + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    # turn array into df so that ppy can work with it
    rows_cols = ['class' + str(i) for i in range(28)]
    rows_cols.remove('class23')
    rows_cols.remove('class27')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    """Class 4"""
    #create object to run ppy code
    df_pont = pont.pontiPy(df)
    #calc PLAND
    pland1_4 = df_pont.size(4, 'X') / df_pont.size() * 100
    pland2_4 = df_pont.size(4, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_4 = df_pont.size(4, 'X')
    tca2_4 = df_pont.size(4, 'Y')
    
    """Class 8"""
    #calc PLAND
    pland1_8 = df_pont.size(8, 'X') / df_pont.size() * 100
    pland2_8 = df_pont.size(8, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_8 = df_pont.size(8, 'X')
    tca2_8 = df_pont.size(8, 'Y')
    
    """Class 9"""
    #calc PLAND
    pland1_9 = df_pont.size(9, 'X') / df_pont.size() * 100
    pland2_9 = df_pont.size(9, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_9 = df_pont.size(9, 'X')
    tca2_9 = df_pont.size(9, 'Y')
    
    """Class 12"""
    #calc PLAND
    pland1_12 = df_pont.size(12, 'X') / df_pont.size() * 100
    pland2_12 = df_pont.size(12, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_12 = df_pont.size(12, 'X')
    tca2_12 = df_pont.size(12, 'Y')
    
    """Class 14"""
    #calc PLAND
    pland1_14 = df_pont.size(14, 'X') / df_pont.size() * 100
    pland2_14 = df_pont.size(14, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_14 = df_pont.size(14, 'X')
    tca2_14 = df_pont.size(14, 'Y')
    
    """Class 16"""
    #calc PLAND
    pland1_16 = df_pont.size(16, 'X') / df_pont.size() * 100
    pland2_16 = df_pont.size(16, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_16 = df_pont.size(16, 'X')
    tca2_16 = df_pont.size(16, 'Y')
    
    """Class 19"""
    #calc PLAND
    pland1_19 = df_pont.size(19, 'X') / df_pont.size() * 100
    pland2_19 = df_pont.size(19, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_19 = df_pont.size(19, 'X')
    tca2_19 = df_pont.size(19, 'Y')

    """Class 21"""
    #calc PLAND
    pland1_21 = df_pont.size(21, 'X') / df_pont.size() * 100
    pland2_21 = df_pont.size(21, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1_21 = df_pont.size(21, 'X')
    tca2_21 = df_pont.size(21, 'Y')
    return pland1_4, pland2_4, tca1_4, tca2_4, pland1_8, pland2_8, tca1_8, tca2_8, pland1_9, pland2_9, tca1_9, tca2_9, pland1_12, pland2_12, tca1_12, tca2_12, pland1_14 \
    , pland2_14, tca1_14, tca2_14, pland1_16, pland2_16, tca1_16, tca2_16, pland1_19, pland2_19, tca1_19, tca2_19, pland1_21, pland2_21, tca1_21, tca2_21

def conf_mat(map1, map2):
    """This function computes the confusion matrix of two maps."""
    # change array to lists as confusion matrix library doesn't work on arrays
    arr1 = map1.astype('int32')
    arr2 = map2.astype('int32')
    arr1 = np.concatenate(map1, axis=0)
    arr2 = np.concatenate(map2, axis=0)
    # compute confusion matrix
    cm = custom_cm(arr1, arr2)
    # delete the rows and columns for out of bound cells
    cm = np.delete(np.delete(cm, [24,27], 0), [24,27], 1)
    return cm

def calc_oa(cm):
    """This function calculates Overall Accuracy from the confusion Matrix
    OA = diagonal of matrix/total nr cells"""  
    #Calculate OA
    oa = cm.agreement() / cm.size()
    return oa

def calc_pontius(cm):
    """This function calculates the overall values of the Pontius metrics and returns these. Note that total allocation is returned and not 
    the newer components of Shift and Exchange."""
    #Calculate overall quantity difference
    oqd = cm.quantity()
    #Calculate total allocation difference as the components of shift and exchange together
    oad = cm.allocation()
    #Calculate overall difference from its components
    od = cm.difference()
    return od, oqd, oad

def calc_oqd_n(cm, n):
    """This function calculates the overall quantity difference for a provided class n"""
    oqd_n = cm.quantity(n)
    return oqd_n

def single_df(maps, stats, metric):
    """This function takes the output from the run comparisons function for single map comparison metrics and returns the distance matrix"""
    #create base df to form basis for distance matrix
    df = pd.DataFrame(index=maps)
    df[metric] = stats
    #calculate euclidean distance between all values then change to matrix form
    matrix = ssd.squareform(ssd.pdist(df))
    df_clean = pd.DataFrame(matrix, index=maps, columns=maps)
    
    # save values to disk
    csv_val = csv_dir + metric + '_values.csv'
    df_vals = pd.DataFrame(index=maps)
    df_vals[metric] = stats
    df_vals.to_csv(csv_val)
    #save df to disk
    csv_name = csv_dir + metric + '_df.csv'
    df_clean.to_csv(csv_name, index=False)
    
def multi_df(map1, map2, stats, metric):
    """This function takes the output from the run calc_poontius function for multi map comparison metrics and stores the data to disk"""
    #Create two dataframes with swapped map columns
    df = pd.DataFrame()
    df['map1'] = [x for x in map1]
    df['map2'] = [x for x in map2]
    df[metric] = stats
    df2 = df
    df2 = df2[[ 'map2', 'map1', metric]]
    df2.columns = ['map1', 'map2', metric]      
    df_concat = pd.concat([df, df2])
    df_pivot = df_concat.pivot(index='map2', columns='map1', values=metric)   
    ## clean up the presentation
    #Remove unecessary labeling
    index = df_pivot.index.union(df_pivot.columns)
    df_clean = df_pivot.reindex(index=index, columns=index)
    #reindex to correct numerical order
    ordered = df_clean.index.to_series().str.rsplit('p').str[-1].astype(int).sort_values()
    df_clean = df_clean.reindex(index=ordered.index, columns=ordered.index).fillna(1).round(decimals=3) 
    #save df to disk
    csv_name = csv_dir + metric + '_df.csv'
    df_clean.to_csv(csv_name, index=False)

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

def custom_cm(y_true, y_pred, *, labels=None, sample_weight=None,
                     normalize=None):
    """Confusion matrix function from the sklearn library 
    adjusted to speed up the generation of matrices
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
        n_labels = labels.size
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        elif y_true.size == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        elif np.all([l not in y_true for l in labels]):
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    check_consistent_length(y_true, y_pred, sample_weight)

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    # convert yt, yp into index
    #y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    #y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
    else:
        dtype = np.float64

    cm = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels), dtype=dtype,
                    ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm

def _check_targets(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            try:
                unique_values = np.union1d(y_true, y_pred)
            except TypeError as e:
                # We expect y_true and y_pred to be of the same data type.
                # If `y_true` was provided to the classifier as strings,
                # `y_pred` given by the classifier will also be encoded with
                # strings. So we raise a meaningful error
                raise TypeError(
                    f"Labels in y_true and y_pred should be of the same type. "
                    f"Got y_true={np.unique(y_true)} and "
                    f"y_pred={np.unique(y_pred)}. Make sure that the "
                    f"predictions provided by the classifier coincides with "
                    f"the true labels."
                ) from e
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred

mpp.Pool.istarmap = istarmap
    
### VARIABLES ###
# max map comparisons to run
nr_maps = 2000

## ABSOLUTE PATH ##
# changing this variable to where the main folder with all files is is the only path adjustment that is necessary
#abs_dir = 'C:/LUMOS/clusterfiles/'
#use for 100 map test
abs_dir = 'C:/LUMOS/MCK/'
map_dir = 'E:/thesis_data/asc_2k/'
#directory where the dataframes containing output are stored
csv_dir = abs_dir + 'Output_DFs/'
# Generate list containing all possible map comparisons
pontius_pairs=[]
for i in range(nr_maps):
    for j in range(nr_maps):
        if i < j:
            map1 = 'map' + str(i)
            map2 = 'map' + str(j)
            pontius_pairs.append((map1,map2))

maps = ['map' + str(i) for i in range(nr_maps)]
# Generate list of map inputs for metrics that have a single map statistic
pontius_list= []
for i in np.arange(0, nr_maps, 2):
        try:
            map1 = maps[i]
            map2 = maps[i + 1]
            pontius_list.append((map1,map2))
        except IndexError:
            # If there is only one map left send it in with itself
            map1 = maps[i]
            map2 = maps[i]
            pontius_list.append((map1,map2)) 
            
#Variable that stores number of iterations required for the assigned number of maps  
multi_its = len(pontius_pairs)
single_its = len(pontius_list)