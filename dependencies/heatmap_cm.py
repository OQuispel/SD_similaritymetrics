#---------------------------------------------------------------------------------------------
# Code Source: https://github.com/verma-priyanka/pontiPy
# Code Acquisition Day: 20/10/2020
# Code was adjusted from adjusted version on 27/11/2020 to create cm matrices with heatmaps
# Python Version: Python 3.7
#---------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn .preprocessing import LabelBinarizer
from sklearn .preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

class pontiPy(object):
    def __init__(self, dataframe):
        """Return a new pandas dataframe object."""
        self.dataframe = dataframe
        self.df_row_col_sums = dataframe.copy(deep=True)
        column_names = []
        for col in self.dataframe.columns:
            column_names.append(col)
        self.df_row_col_sums['Col Sum'] = self.df_row_col_sums.sum(axis=1)
        self.df_row_col_sums.loc['Row Sum'] = self.df_row_col_sums.sum(axis=0)
        
        # Normalize the df and convert fraction to percentage
        df_sum = self.size()
        self.df_row_col_sums = self.df_row_col_sums.loc[:'class27', :'class27'] / df_sum * 100
        # Calculate normalized sums
        self.df_row_col_sums['Col Sum'] = self.df_row_col_sums.sum(axis=1)
        self.df_row_col_sums.loc['Row Sum'] = self.df_row_col_sums.sum(axis=0)

    # Function to compute false alarms
    def row_disagreement(self, category = None):
        _false_alarm = []
        # subtract one to get # of categories
        # removes row sum from the length
        df_length = (len(self.df_row_col_sums) - 1)
        for i in range(len(self.df_row_col_sums)):
            # False alarms = Column Sum - Hits for each category
            _false_alarm.append(self.df_row_col_sums.iloc[i][df_length]-self.df_row_col_sums.iloc[i][i])
        # if no category is specified
        # len-1 because total false alarm sum is included in list
        if category is None:
            return sum(_false_alarm[0:len(_false_alarm)-1])
        # List to build contingency table in the contingency() function
        elif category == 'CONTINGENCY':
            # add sum of false alarms to list as last item
            _false_alarm[len(_false_alarm)-1] = sum(_false_alarm[0:len(_false_alarm)-1])
            return _false_alarm
        # if category is specified, return false alarm for that category
        else:
            return _false_alarm[category]

    # Function to compute miss
    def column_disagreement(self, category = None):
        _miss = []
        df_length = len(self.df_row_col_sums)-1
        for i in range(len(self.df_row_col_sums)):
            # miss = Row Sum - Hits for each category
            _miss.append(self.df_row_col_sums.iloc[df_length][i]- self.df_row_col_sums.iloc[i][i])
        # if no category is specified
        if category is None:
            return sum(_miss[0:len(_miss)-1])
        # List to build contingency table in the contingency() function
        # add sum of misses to list as last item
        elif category == 'CONTINGENCY':
            _miss[len(_miss)-1] = sum(_miss[0:len(_miss)-1])
            return _miss
        # if category is specified, return miss for that category
        else:
            return _miss[category]
        
    def size(self, category = None, axis= None, Total = False):
        # size of the data frame is returned
        if category is None and axis is None:
            return self.df_row_col_sums.at['Row Sum', 'Col Sum']
        # return col or row sum for category depending on axis
        # An axis (x or y) must be provided with a category
        elif category is not None and axis is not None:
            if Total is False:
                # If x is specified, return col sum for category
                if axis.lower() == 'x':
                    _col_sum = self.df_row_col_sums['Col Sum'][category]
                    return _col_sum
                # If y is specified, return row sum for category
                elif axis.lower() == 'y':
                    _row_sum = self.df_row_col_sums.loc['Row Sum'][category]
                    return _row_sum
                # if axis isn't specified, return the sum of col and row sum for category
            else:
                # Get row
                if axis.lower() == 'x':
                    _col_sum = self.df_row_col_sums.iloc[category]
                    x_dict = _col_sum.to_dict()
                    # Remove col sum and False Alarms if they exist
                    x_dict.pop('Col Sum', None)
                    x_dict.pop('False Alarms', None)
                    return x_dict
                # If y is specified, return col for category
                elif axis.lower() == 'y':
                    # list of i
                    index_list = self.df_row_col_sums.index
                    for col in index_list:
                        pos = (self.df_row_col_sums.index.get_loc(col))
                        if pos == category:
                            y_dict = self.df_row_col_sums.get(col).to_dict()
                            # Remove Row Sum and Misses if they exist in dictionary
                            y_dict.pop('Row Sum', None)
                            y_dict.pop('Misses', None)
                            return y_dict
        else:
            return(self.size(category, axis = 'x') + self.size(category, axis = 'y'))
 
    # Function to compute quantity between all or one category
    # Requires at least 1 category in parameter
    def quantity(self, category = None, label = False):
        if category is not None:
            # Returned as a dictionary
            # If no category is specified, return total quantity
            _quantity = {}
            # Calculate quantity by subtracting false alarms from misses
            _q_by_category = self.column_disagreement(category) - self.row_disagreement(category)
            # Quantity Labels
            # If greater than 0, it is a miss quantity
            if _q_by_category > 0:
                _quantity['Miss'] = abs(_q_by_category)
            # If greater than 1, it is a false alarm quantity
            elif _q_by_category < 0:
                _quantity['False Alarm'] = abs(_q_by_category)
            # If 0, quantity is 0
            else:
            # if it isn't a miss or false alarm quantity
                _quantity['Blank'] = abs(_q_by_category)

         # If no category is specified: return the absolute sum of all quantity
        # Divide sum quantity by 2
        # Label is off by default
        if category is None:
            _categories = range(len(self.df_row_col_sums) - 1)
            _quantity_sum = 0
            for i in _categories:
                _quantity_sum += abs(self.column_disagreement(i)-self.row_disagreement(i))
            return int(_quantity_sum/2)
        # If True: it returns the quantity value for that category
        elif label is True:
            return _quantity
        # If False (Default): it returns a dictionary
        # Dictionary Key = Miss/False Alarm/Blank label
        # Dictionary Value = Quantity value for the key
        elif label is False:
            return list(_quantity.values())[0]

    # Generate final matrix
    # This function will call previous functions
    def matrix(self):
        _matrix = self.df_row_col_sums.copy(deep=True)
        miss_row = self.column_disagreement('CONTINGENCY')
        # Add a blank item since the False Alarm column will not have misses
        # This is required since the list size will differ from matrix size
        miss_row.append('')
        # Add False alarm to matrix
        _matrix["RowDisagreement"] = self.row_disagreement('CONTINGENCY')
        # Add Misses to matrix
        _matrix.loc['ColumnDisagreement'] = miss_row
        # Rename columns for display
        _matrix = _matrix.rename({'Col Sum': 'Sum'}, axis=1)
        _matrix = _matrix.rename({'Row Sum': 'Sum'}, axis=0)
        return _matrix
    
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