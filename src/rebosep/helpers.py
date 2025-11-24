import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_cluster_names(data,
                      resolution,
                      cluster_dic,
                      verbose = True, 
                      path=None,
                      annotation_key="leiden_annotated",
                      algorithm="leiden"):
    """
    add column with tissue annotations to anndata.obs
    """
    if verbose:
        print("Setting cluster names.")
    
    temp_data = copy.deepcopy(data.tl.result[algorithm])
    temp_data["group"] = temp_data["group"].astype("str")
    temp_data["annotation"] = "other"
    
    for leiden_cluster in cluster_dic:
        temp_data.loc[(temp_data['group'] == str(leiden_cluster)), 'annotation'] = cluster_dic[leiden_cluster]
    
    temp_data["group"] = temp_data["annotation"]
    data.tl.result._set_cluster_res(annotation_key, temp_data) # stereopy has different types of result. "cluster_res" can only save one column called "group". https://github.com/STOmics/Stereopy/blob/main/stereo/core/result.py
    


def set_annotation_by_region(df, 
                             region=None, 
                             annotation=None,
                             annotation_key="filtered_annotated",
                             region_num_key="region_num_key"):

    """
    used to set annotation of individual detected regions
    """
    
    if region == None or annotation == None:
        raise ValueError('ERROR: Region or annotation not set')

    df[annotation_key] = np.where(df[region_num_key] == region,
                                  annotation,
                                  df[annotation_key])

    return df


def set_manually_selected_region(obs, 
                                 selector, 
                                 col_name="manually_selected"):

    """
    creates column in anndata.obs with manually selected bins
    """
    
    obs[col_name] = False
    for elem in selector.xys[selector.ind]:
        obs.loc[(obs['x'] == elem[0]) & (obs['y'] == elem[1]*-1),col_name] = True
    
    return obs