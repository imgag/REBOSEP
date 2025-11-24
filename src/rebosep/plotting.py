import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def filter_regions_plotting(data_df, 
                            verbose=True, 
                            region_num_key="region_num_key",
                            plot_masked=False,
                            x_column="x",
                            y_column="y"
                            ):
    """
    visualize masked regions after filter_regions
    """
    
    df = data_df.copy()
    if plot_masked:
        df[region_num_key] = np.where(df[region_num_key].isna(), 1, np.nan)
    
    table = df.pivot(index=y_column, 
                     columns=x_column, 
                     values=region_num_key).to_numpy()

    im = plt.imshow(table, cmap="tab20", interpolation='none')



def plot_boundary_distances(obs, 
                            boundary_dist_key="boundary_dist", 
                            annotation_key="filtered_annotated",
                            manual_selection_key=None,
                            verbose=True, 
                            x_column="x",
                            y_column="y",
                            highlight_boundary=True):
    
    """
    visualize boundary distances
    """
    
    if boundary_dist_key not in obs.columns:
        raise ValueError(f"{boundary_dist_key} not in object")
        
    df_tmp = obs.copy()
    table_regions = df_tmp.pivot(index=y_column, 
                                 columns=x_column, 
                                 values=annotation_key).values # x and y inverted so that matrix orientation fits image
    region_arr_plotting = np.full((len(table_regions), len(table_regions[0])), np.nan)
    table_regions = table_regions.astype("str")
    unique_annotations = np.unique(table_regions)

    # converting string to integers for plotting with imshow
    for i in range(len(table_regions)):
        for j in range(len(table_regions[i])):
            plot_val = table_regions[i][j]
            if plot_val == "nan":
                continue
            
            region_arr_plotting[i][j] = np.where(unique_annotations==plot_val)[0][0]
    
    table_dist = df_tmp.pivot(index=y_column, 
                              columns=x_column, 
                              values=boundary_dist_key).values # x and y inverted so that matrix orientation fits image

    # plot boundary distances only in manually selected area
    if manual_selection_key != None:
        table_mask = df_tmp.pivot(index="y", 
                                  columns="x", 
                                  values=manual_selection_key).values # x and y inverted so that matrix orientation fits image
    
    distance_arr_plotting = np.full((len(table_dist), len(table_dist[0])), np.nan)

    # create matrix of distances for plotting
    for i in range(len(table_dist)):
        for j in range(len(table_dist[i])):
            if manual_selection_key != None:
                if table_mask[i][j]:
                    distance_arr_plotting[i][j] = table_dist[i][j]
            else:
                distance_arr_plotting[i][j] = table_dist[i][j]
    
    
    # plot the background
    plt.imshow(region_arr_plotting, interpolation="none", cmap="binary", vmin=-1, vmax=np.nanmax(region_arr_plotting)*2)
    plt.axis("off")
    plt.tight_layout()

    # plot the distances
    plt.imshow(distance_arr_plotting, interpolation="none", cmap="PuBu", vmin=0, vmax=np.nanmax(distance_arr_plotting)) # vmin usefull to set middle of color map to 0
    plt.colorbar(label="distance from boundary")
    plt.axis('off')
    plt.tight_layout()

    # plot the boundary
    if highlight_boundary:
        distance_arr_plotting[distance_arr_plotting!=1] = np.nan
        plt.imshow(distance_arr_plotting, interpolation="none", cmap="Wistia_r") 
        plt.tight_layout()
        plt.axis('off')



def plot_obs_column(anndata, 
                    column_key): 
    """
    visualize one column of the anndata.obs dataframe
    """
    
    temp = copy.deepcopy(anndata.obs)  
    temp[column_key].unique()
    
    translation = dict()
    rev_translation = dict()
    
    i = 1
    values = temp[column_key].unique()
    values = values[~pd.isnull(values)]
    for val in values:
        translation[val] = i
        rev_translation[i] = val
        i += 1
    
    temp["_translated"] = temp.replace({column_key: translation})[column_key]
    
    table = temp.pivot(index="y", 
                       columns="x", 
                       values="_translated") # x and y inverted so that matrix orientation fits image
    table = table.astype(np.float32)
    
    #values = np.unique(table.values.ravel())
    im = plt.imshow(table, cmap="Set1", interpolation="none")
    plt.axis('off')
    #https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    colors = [im.cmap(im.norm(value+1)) for value in range(len(values))]
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=rev_translation[i+1]) ) for i in range(len(values)) ] # i+1 because translation dic starts at 1 
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
