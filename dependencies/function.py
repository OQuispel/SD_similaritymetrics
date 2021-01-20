import numpy as np
import tempfile
from shutil import copy
import pandas as pd
import subprocess
import shutil
import multiprocessing.pool as mpp
import scipy.spatial.distance as ssd


def run_comparisons(pair_id, metric_id):
    """This function runs through all required steps to compare a pair of maps (see Approach step 2-9)"""
    global map1_id, map2_id
    #map selection is different based on metric type
    if metric_id in single_comparison:
        map1 = map_list[pair_id][0]
        map2 = map_list[pair_id][1]        
    if metric_id in multi_comparison:
        map1 = map_pairs[pair_id][0]
        map2 = map_pairs[pair_id][1]
    # Stores map number of each mLap for later use in processing
    map1_id, map2_id = map_id(map1, map2)
    name1 = 'map' + str(map1_id)
    name2 = 'map' + str(map2_id)
    if metric_id == 'kappa':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/base_csl.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\kappa.sts'
            kappa = extract_stats(comparison, metric_id)
            return name1, name2, kappa
    elif metric_id == 'kfuzzy':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/kfuzzy.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\kfuzzy.sts'
            kfuzzy = extract_stats(comparison, metric_id)
            return name1, name2, kfuzzy
    elif metric_id == 'simp':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/base_csl.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\\simpson.sts'
            sim1, sim2 = extract_stats(comparison, metric_id)
            return name1, name2, sim1, sim2   
    elif metric_id == 'shan':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/base_csl.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\\shannon.sts'
            shan1, shan2 = extract_stats(comparison, metric_id)
            return name1, name2, shan1, shan2 
    else:
        return 'invalid metric id'
    
def copy_files(path, map_nr1, map_nr2, metric_id):
    """This function takes a filepath and a pair of maps then copies the maps and other required files to the specified filepath"""
    #Convert maps
    map1name = 'map' + str(map_nr1) + '.asc'
    map2name = 'map' + str(map_nr2) + '.asc'
    map1 = ascfolder + map1name
    map2 = ascfolder + map2name
    copy(map1, path)   
    copy(map2, path)
    #Copy mask files
    copy(maskfile, path)   
    copy(masklandfile, path)
    #Generate and copy csl file for comparison
    if metric_id == 'kappa':
        csl_kappa(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    elif metric_id == 'kfuzzy':
        csl_kfuzzy(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    elif metric_id == 'simp':
        csl_simpson(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    elif metric_id == 'shan':
        csl_shannon(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    #Copy log file
    gen_log(path, map_nr1, map_nr2)
    #Copy legends folder
    dst = path + '/Legends'
    shutil.copytree(src,dst)
    
def extract_stats(sts, sts_id):
    """This function reads the output file generated by MCK and extracts the necessary stats based on the metric send"""
    if sts_id == 'kappa':
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    # Extract kappa
                    loc2 = line.find(' kappa="')
                    loc_start2 = loc2 + len(' kappa="')
                    loc_end2 =  line.find('"', loc_start2)
                    kappa = float(line[loc_start2:loc_end2])
                    return kappa
    elif sts_id == 'kfuzzy':
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    # Extract fuzzy kappa
                    loc2 = line.find('fuzzy_kappa="')
                    loc_start2 = loc2 + len('fuzzy_kappa="')
                    loc_end2 =  line.find('"', loc_start2)
                    kfuzzy = float(line[loc_start2:loc_end2])
                    return kfuzzy
    elif sts_id == 'simp':
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    #Find first value
                    loc = line.find('overall_first="')
                    loc_start = loc + len('overall_first="')
                    loc_end =  line.find('"', loc_start)
                    sim1 = float(line[loc_start:loc_end])
                    #Find second value
                    loc2= line.find('overall_second="')
                    loc_start2= loc2 + len('overall_second="')
                    loc_end2= line.find('"', loc_start2)
                    sim2 = float(line[loc_start2:loc_end2])
                    return sim1, sim2
    elif sts_id == 'shan':
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    #Find first value
                    loc = line.find('overall_first="')
                    loc_start = loc + len('overall_first="')
                    loc_end =  line.find('"', loc_start)
                    shan1 = float(line[loc_start:loc_end])
                    #Find second value
                    loc2= line.find('overall_second="')
                    loc_start2= loc2 + len('overall_second="')
                    loc_end2= line.find('"', loc_start2)
                    shan2 = float(line[loc_start2:loc_end2])
                    return shan1, shan2
                        
    else:
        return 'Incorrect stats id'
    
def map_id(map1, map2):
    """This function returns the map number for 2 provided maps based on the path of the maps"""
    map1_start = map1.find('_') + 1
    map1_end = map1.find('.')
    map1_id = int(map1[map1_start:map1_end])
    
    map2_start = map2.find('_') + 1
    map2_end = map2.find('.')
    map2_id = int(map2[map2_start:map2_end])
    return map1_id, map2_id
            
def csl_kappa(path, map_nr1, map_nr2):
    """This function copies a base csl file to the provided directory then rewrites it to run the desired kappa comparison"""
    file_dir = path + '\\base_csl.csl'
    copy(base_csl, path)
    with open(file_dir, 'w+') as file:
        file.truncate(0)
        file.write('<comparisonsets>\n')
        map1 = path + '\map' + str(map_nr1) + '.asc'
        map2 = path + '\map' + str(map_nr2) + '.asc'
        mask = path + '\mask.asc'
        file.writelines(['\n\t<comparisonset displayname="kappa"' + ' map1path=' + '"' \
                        + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Kappa"' \
                        + ' outputstatistics=' + '"'+ 'kappa.sts' + '"'+ ' theme1="maps" theme2="maps" up2date="0">' \
                        ,'\n\t\t<parameterset/>' \
                        , '\n\t\t<mask basemappath=' + '"' +  mask + '"' + ' displayname="NL_mask" mergeregions="0">' \
                        ,'\n\t\t\t<selectedregions>' \
                        ,'\n\t\t\t\t<value value="0"/>' \
                        ,'\n\t\t\t\t<value value="1"/>' \
                        ,'\n\t\t\t</selectedregions>' \
                        ,'\n\t\t</mask>' \
                        ,'\n\t</comparisonset>\n'])
        file.write('</comparisonsets>')
        
def csl_kfuzzy(path, map_nr1, map_nr2):
    """This function copies a base csl file to the provided directory then rewrites it to run the desired fuzzy kappa comparison"""
    map1 = path + '\\map' + str(map_nr1) + '.asc'
    map2 = path + '\\map' + str(map_nr2) + '.asc'
    mask = path + '\mask.asc'      
    copy(base_kfuzzy, path)
    
    file_dir = path + "\kfuzzy.csl"   
    
    #Read data
    with open(file_dir, 'r') as file:
        data = file.readlines()
    #Change Data   
    line3= '\t<comparisonset displayname="kfuzzy"' + ' map1path=' + '"' \
                                + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Fuzzy Kappa (2009 version)"' \
                                + ' outputstatistics="kfuzzy.sts"' +' theme1="maps" theme2="maps" up2date="0">\n'
    line909 = '<mask basemappath=' + '"' +  mask + '"' + '  displayname="NL_mask" mergeregions="0">\n'
    data[3] = line3
    data[909] = line909
    #Write new data in
    with open(file_dir, 'w') as file:
        file.writelines(data)
        
def csl_simpson(path, map_nr1, map_nr2):
    """This function copies a base csl file to the provided directory then rewrites it to run the desired simpson's comparison"""
    file_dir = path + '\\base_csl.csl'
    copy(base_csl, path)
    with open(file_dir, 'w+') as file:
        file.truncate(0)
        file.write('<comparisonsets>\n')
        map1 = path + '\map' + str(map_nr1) + '.asc'
        map2 = path + '\map' + str(map_nr2) + '.asc'
        mask = path + '\mask.asc'
        file.writelines(['\n\t<comparisonset displayname="simpson"' + ' map1path=' + '"' \
                        + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Moving Window based Structure"' \
                        + ' outputstatistics=' + '"'+ 'simpson.sts' + '"'+ ' theme1="maps" theme2="maps" up2date="0">' \
                        ,'\n\t\t<parameterset>'\
                        , '\n\t\t\t<comparison_moving_window_structure aggregation="1" average_per_cell="1" ' \
                        + 'background_category="0" category_of_interest="0" display_map="0" ' \
                        + 'distance_weighed="0" halving="2" include_diagonal="1" metric="10" ' \
                        + 'per_category="0" radius="4" use_background="0"/>' \
                        ,'\n\t\t</parameterset>' \
                        , '\n\t\t<mask basemappath=' + '"' +  mask + '"' + ' displayname="NL_mask" mergeregions="0">' \
                        ,'\n\t\t\t\t<selectedregions>' \
                        ,'\n\t\t\t\t\t<value value="0"/>' \
                        ,'\n\t\t\t\t\t<value value="1"/>' \
                        ,'\n\t\t\t</selectedregions>' \
                        ,'\n\t\t</mask>' \
                        ,'\n\t</comparisonset>\n'])
        file.write('</comparisonsets>')  

def csl_shannon(path, map_nr1, map_nr2):
    """This function copies a base csl file to the provided directory then rewrites it to run the desired simpson's comparison"""
    file_dir = path + '\\base_csl.csl'
    copy(base_csl, path)
    with open(file_dir, 'w+') as file:
        file.truncate(0)
        file.write('<comparisonsets>\n')
        map1 = path + '\map' + str(map_nr1) + '.asc'
        map2 = path + '\map' + str(map_nr2) + '.asc'
        mask = path + '\mask.asc'
        file.writelines(['\n\t<comparisonset displayname="shannon"' + ' map1path=' + '"' \
                        + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Moving Window based Structure"' \
                        + ' outputstatistics=' + '"'+ 'shannon.sts' + '"'+ ' theme1="maps" theme2="maps" up2date="0">' \
                        ,'\n\t\t<parameterset>'\
                        , '\n\t\t\t<comparison_moving_window_structure aggregation="1" average_per_cell="1" ' \
                        + 'background_category="0" category_of_interest="0" display_map="0" ' \
                        + 'distance_weighed="0" halving="2" include_diagonal="1" metric="9" ' \
                        + 'per_category="0" radius="4" use_background="0"/>' \
                        ,'\n\t\t</parameterset>' \
                        , '\n\t\t<mask basemappath=' + '"' +  mask + '"' + ' displayname="NL_mask" mergeregions="0">' \
                        ,'\n\t\t\t\t<selectedregions>' \
                        ,'\n\t\t\t\t\t<value value="0"/>' \
                        ,'\n\t\t\t\t\t<value value="1"/>' \
                        ,'\n\t\t\t</selectedregions>' \
                        ,'\n\t\t</mask>' \
                        ,'\n\t</comparisonset>\n'])
        file.write('</comparisonsets>') 
        
def gen_log(path, map_nr1, map_nr2):
    """This function copies a base log file to the provided directory then rewrites it with desired changes"""
    copy(log_path, path)
    file_dir = path + '/log.log'
    line3 = 'maps=' + path + '\map' + str(map_nr1) + '.asc'
    line4 = 'maps=' + path + '\map' + str(map_nr2) + '.asc'
    #Read data
    with open(file_dir, 'r') as file:
        data = file.readlines()
    #Change Data    
    data[3] = line3 + '\n'
    data[4] = line4
    #Write new data in
    with open(file_dir, 'w') as file:
        file.writelines(data) 
        
def call_cmd(mckdir, csl, log, outputdir):
    """"""
    cmd= [mckdir, 'CMD',  '/RunComparisonSet', csl, log, outputdir]
    subprocess.run(cmd, check=True, shell=True)

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
    df_vals = pd.DataFrame(index=map_set)
    df_vals[metric] = stats
    df_vals.to_csv(csv_val)
    #save df to disk
    csv_name = csv_dir + metric + '_df.csv'
    df_clean.to_csv(csv_name, index=False)

def multi_df(map1, map2, stats, metric):
    """This function takes the output from the run comparisons function for multi map comparison metrics and returns the distance matrix"""
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
mpp.Pool.istarmap = istarmap

#### SETUP VARIABLES ####
## ABSOLUTE PATH ##
# changing this variable to where the main folder with all files is is the only path adjustment that is necessary
abs_dir = 'C:/LUMOS/clusterfiles/'
# Name of the csv files generated by landusescanner that contain map data
lus_outputname = "map500exp_"
ext = '.csv'
#copy files source
src = abs_dir + 'Legends'
# Location of required files for MCK Comparisons
ascfolder = abs_dir + 'ascmaps/'
csvfolder = abs_dir + 'csvmaps/'
maskfile = abs_dir + 'mask.asc'
masklandfile = abs_dir + 'maskland.msk'
base_csl = abs_dir + 'base_csl.csl'
#fuzzy kappa has its own base csl due to its different structure
base_kfuzzy = abs_dir + 'kfuzzy.csl'
log_path = abs_dir + 'log.log'
mckdir =  abs_dir + 'Map_Comparison_Kit/MCK.exe'
# max map comparisons to run
nr_maps = 10
#directory where the dataframes containing output are stored
csv_dir = abs_dir + 'Output_DFs/'

map_set = ['map' + str(i) for i in range(nr_maps)]

# Generate list containing a set of maps based on setup variables
maps = []
maps.extend([csvfolder + lus_outputname + str(x) + ext for x in np.arange(0, nr_maps)])

# Generate list containing all possible map comparisons
map_pairs=[]
for i in range(len(maps)):
    for j in range(len(maps)):
        if i < j:
            map1 = maps[i]
            map2 = maps[j]
            map_pairs.append((map1,map2))

# Generate list of map inputs for metrics that have a single map statistic
map_list= []
for i in np.arange(0, len(maps), 2):
        try:
            map1 = maps[i]
            map2 = maps[i + 1]
            map_list.append((map1,map2))
        except IndexError:
            # If there is only one map left send it in with itself
            map1 = maps[i]
            map2 = maps[i]
            map_list.append((map1,map2))  
            
# Lists used in run_comparison to check comparison type
single_comparison = ['pland', 'tca', 'shan', 'simp']
multi_comparison = ['kappa', 'kfuzzy', 'alloc', 'quant', 'td', 'oa', 'prop']

#Variables that store number of iteration for single and multi map comparisons
single_its = len(map_list)
multi_its = len(map_pairs)