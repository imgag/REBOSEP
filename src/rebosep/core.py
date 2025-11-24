import itertools
import copy

import numpy as np
import pandas as pd
import scanpy as sc
from skimage.draw import disk

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.widgets import LassoSelector


#from REBOSEP.helpers import bin_recursion


def bin_recursion(to_check, 
                  checked, 
                  unique_regions, 
                  region_num, 
                  current_tissue, 
                  region_np):
    """
    finds all bins belonging to the same region
    """
    
    for position in to_check:
        i = position[0]
        j = position[1]
        
        if not (0 <= i < region_np.shape[0]):
            continue
        if not (0 <= j < region_np.shape[1]):
            continue
        if f"{i}_{j}" in checked:
            continue
        if current_tissue != region_np[i][j]:
            continue
        
        checked.add(f"{i}_{j}")
        #print("adding")
        unique_regions[region_num].add(f"{i}_{j}")
        to_check.extend([[i, j+1],  
                         [i, j-1],  
                         [i+1, j],  
                         [i-1, j],  
                         [i+1, j+1],
                         [i-1, j+1],
                         [i-1, j-1],
                         [i+1, j-1],
                         ])
    
    return checked, unique_regions


def filter_regions(data_df, 
                   column_to_filter_key="leiden_annotated",
                   filtered_annotation_key="filtered_annotated",
                   too_small_label="too_small",
                   x_column="x",
                   y_column="y",
                   region_num_key="region_num_key",
                   min_region_size=500,
                   verbose=True):
    """
    filters regions by size
    """
    
    if verbose:
        print("filtering regions by size")
    
    # deleting old results    
    try:
        data_df = data_df.drop(columns=[region_num_key], errors="raise")
        if verbose: 
            print("deleted previous region num key result!")
    except KeyError:
        print("no region num key column found. If first pass, all good.")
    except Exception:
        raise ValueError('Removing of region num key failed. ERROR SOURCE UNKNOWN!')
    
    # creating matrix of regions
    region_df = data_df.astype({y_column:int, x_column:int})
    region_np = region_df.pivot(index=y_column, 
                                columns=x_column, 
                                values=column_to_filter_key).to_numpy()
    
    # detecting all regions, regardless of size
    checked = set()              # bins already checked, "{i}_{j}" format
    to_check = []                # detected bins, still left to check
    unique_regions = dict()
    too_small_regions = []
    region_num = 0
    
    for i in range(0, len(region_np)-1):
        for j in range(len(region_np[i])-1):
            if f"{i}_{j}" not in checked:        # every bin that still needs to be checked (doesnt yet belong to a region) starts detection of all bins belong to its region
                to_check = [[i, j]]
                unique_regions[region_num] = set()
                
                checked, unique_regions = bin_recursion(to_check, checked, unique_regions, region_num, region_np[i][j], region_np)
                if len(unique_regions[region_num]) < min_region_size:
                    too_small_regions += unique_regions[region_num]
                    del unique_regions[region_num]
                else:
                    region_num = region_num+1
    
    if region_num == 0:
        raise ValueError(f"No region big enough.")

    x_arr = []
    y_arr = []
    bin_coords_x = np.unique(np.sort(region_df["x"].values))
    bin_coords_y = np.unique(np.sort(region_df["y"].values))                            
    
    region_num_key_arr = []
    for region_id in unique_regions:
        for coordinates in unique_regions[region_id]:
            
            x_arr.append(bin_coords_x[int(coordinates.split("_")[1])])   # i = y coordinate
            y_arr.append(bin_coords_y[int(coordinates.split("_")[0])])   #j = x coordinate
            region_num_key_arr.append(region_id)
    
    unique_regions_df = pd.DataFrame({x_column: x_arr, y_column: y_arr, region_num_key: region_num_key_arr})

    new_obs = region_df.merge(unique_regions_df, on=[x_column, y_column], how='left') # has two x, and two y columns.
    new_obs.index = new_obs.index.astype('str')  # anndata expects index to be str

    new_obs[filtered_annotation_key] = new_obs[column_to_filter_key]
    new_obs.loc[new_obs[region_num_key].isna(), filtered_annotation_key] = np.nan

    
    return new_obs


def find_boundary_distances(obs, 
                            clusters,
                            x_column="x",
                            y_column="y",
                            clusters_key="filtered_annotated",
                            is_boundary_key_prefix="is_boundary_",
                            min_boundary_length = 100,
                            boundary_dist_key_prefix="boundary_dist_",
                            boundary_range = 1,
                            max_dist=20,
                            print_modulo=1,
                            verbose=True):
    
    """
    calculates the distance of bins from the boundary
    """
    
    region_np = obs.pivot(index=y_column, 
                          columns=x_column,
                          values=clusters_key).to_numpy() # x and y inverted so that matrix orientation fits image

    cluster_a = clusters[0]
    cluster_b = clusters[1]
    new_obs = obs.copy()

    new_col_name = boundary_dist_key_prefix + cluster_a + "_" + cluster_b
    is_boundary_col_name = is_boundary_key_prefix + cluster_a + "_" + cluster_b

    # deleting old results
    if verbose:
        print(f"detecting boundaries between {cluster_a} and {cluster_b}")
    try:
        new_obs = new_obs.drop(columns=[new_col_name], errors="raise")
        if verbose: 
            print("deleted previous boundary result!")
    except KeyError:
        if verbose:
            print("no boundary column found. If first pass, all good")
    except Exception:
        raise ValueError('Removing of boundary failed. ERROR SOURCE UNKNOWN!')
    
    # detect all boundary bins between two relevant annotations
    is_boundary_arr = np.full((len(region_np), len(region_np[0])), False)
    for i in range(len(region_np)):
        for j in range(len(region_np[0])):
            if is_boundary_bin(region_np, i, j, [cluster_a, cluster_b]):
                is_boundary_arr[i][j] = True
    
    if verbose:
        print("running boundary recursion")

    checked = set()
    unique_boundaries = {}
    unique_boundary_id = 0
    for i in range(len(is_boundary_arr)):
        for j in range(len(is_boundary_arr[0])):
            if f"{i}_{j}" not in checked:
                boundary_elements, checked = boundary_recursion(is_boundary_arr, [f"{i}_{j}"], checked, boundary_range)
                if len(boundary_elements) > 0:
                    unique_boundaries[unique_boundary_id] = boundary_elements
            unique_boundary_id += 1

    max_length = 0
    for k in list(unique_boundaries.keys()):
        max_length = max(max_length, len(unique_boundaries[k]))
        if len(unique_boundaries[k]) < min_boundary_length:
            del unique_boundaries[k]
    
    
    if len(unique_boundaries) == 0:
        #raise ValueError(f"No boundaries long enough. Max length: {max_length}")
        raise ValueError(f"No boundary long enough found between {cluster_a} and {cluster_b}. Max length found: {max_length}")
    
    boundaries_arr_bool = np.full((len(is_boundary_arr), len(is_boundary_arr[0])), False)
    for k in unique_boundaries.keys():
        for coord in unique_boundaries[k]:
            i = int(coord.split("_")[0])
            j = int(coord.split("_")[1])
            boundaries_arr_bool[i][j] = True

    new_obs = create_new_obs(boundaries_arr_bool,
                             new_obs,
                             is_boundary_col_name,
                             x_column="x",
                             y_column="y")
    
    print("running distance detection")
    #4. pad up to max wanted dist
    boundary_distance_table = calculate_boundary_distance(new_obs,  
                                                          max_dist= max_dist,
                                                          testing_out_dir="", 
                                                          is_boundary_key=is_boundary_col_name, 
                                                          boundary_dist_key=new_col_name, 
                                                          filtered_clusters_key=clusters_key, 
                                                          verbose=True,
                                                          print_modulo=print_modulo)

    
    
    boundary_distance_column = pd.DataFrame(data=boundary_distance_table,    # values
                                            index=np.sort(new_obs[y_column].unique()),    # 1st column as index
                                            columns=np.sort(new_obs[x_column].unique()))  # 1st row as the column names
    boundary_distance_column = boundary_distance_column.stack().reset_index(name=new_col_name)
    boundary_distance_column = boundary_distance_column.rename(columns={"level_0": y_column, "level_1": x_column})
    
    new_obs["_bin_names"] = new_obs.index
    new_obs = pd.merge(new_obs, boundary_distance_column,  how='left', left_on=[x_column, y_column], right_on = [x_column, y_column])
    new_obs.index = new_obs["_bin_names"]
    new_obs = new_obs.drop('_bin_names', axis=1)
    
    new_obs.loc[~new_obs[clusters_key].isin(clusters), new_col_name] = np.nan
    new_obs=new_obs.drop(columns=[is_boundary_col_name], errors="raise")
    
    return new_obs


def calculate_boundary_distance(obs,
                                max_dist = 20,
                                x_column="x",
                                y_column="y",
                                testing_out_dir="", 
                                is_boundary_key="is_boundary", 
                                boundary_dist_key="boundary_dist", 
                                filtered_clusters_key="filtered_annotated", 
                                verbose=True,
                                print_modulo=1):

    """
    performs detection of bin distances
    """
    is_boundary_table = obs.pivot(index=y_column, columns=x_column, values=is_boundary_key).values
    annotation_table = obs.pivot(index=y_column, columns=x_column, values=filtered_clusters_key).values
    boundary_distance_table = np.full((len(is_boundary_table), len(is_boundary_table[0])), np.nan)

    # precompute disks, only keeping outer ring to minimize redundant checks
    disc_dic = {1: [(0), (0)]}
    for n in range(max_dist, 1, -1):
        coords         = disk((0,0), n)
        coords         = list(zip(coords[0], coords[1]))
        coords_smaller = disk((0,0), n-1)
        coords_smaller = list(zip(coords_smaller[0], coords_smaller[1]))
        coords = [coord for coord in coords if coord not in coords_smaller]
        coords = list(zip(*coords))
        disc_dic[n] = coords

    # settings distances
    for n in range(max_dist, 0, -1):
        if verbose and n%print_modulo == 0:
            print(f"calculating bins with boundary distance: {n}")
        for i in range(0, len(is_boundary_table)-1):
            for j in range(len(is_boundary_table[i])-1):
                if is_boundary_table[i][j] and not np.isnan(is_boundary_table[i][j]):

                    current_tissue = annotation_table[i][j]

                    coords = disc_dic[n]
                    coords = np.array(coords)
                    coords[0] += i
                    coords[1] += j

                    mask = (coords[0] >= 0) & (coords[1] >= 0) & (coords[0] < len(is_boundary_table)) & (coords[1] < len(is_boundary_table[0])) # removing coordinates outside of array
                    coords_x = coords[0]
                    coords_y = coords[1]
                    coords_x = coords_x[mask]
                    coords_y = coords_y[mask]

                    coords = [coords_x, coords_y]
                    for k in range(len(coords[0])):
                        if annotation_table[coords[0][k]][coords[1][k]] == current_tissue: # make sure that only boundarie bin of same celltype is used to calculate distances in a tissue, otherwise one side of the dge will get completely overwritten
                            boundary_distance_table[coords[0][k]][coords[1][k]] = n
        
    return boundary_distance_table



def select_region_of_interest(anndata, 
                              set_nan_to_zero=False, 
                              col_name="filtered_annotated", 
                              x_column="x",
                              y_column="y",
                              alpha_background=0.3,
                              lasso_color="red",
                              lasso_linewidth=2,
                              s=0.5):
    """
    manual selection of region of interest
    """
    
    temp = anndata.raw.to_adata()
    test_red = temp.obs[[x_column, y_column, col_name]]
    
    fig, ax = plt.subplots()
    
    levels, categories = pd.factorize(test_red[col_name])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]
    
    im = ax.scatter(test_red[x_column], test_red[y_column]*-1, c=colors, s=s)
    plt.gca().set(xlabel='', ylabel='', title='select boundary region')
    plt.gca().set_aspect("equal")
    plt.tick_params(axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False) 
    plt.tick_params(axis='y',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False) 
    #plt.legend(handles=handles, title='type of tissue')

    selector = SelectFromCollection(ax, 
                                    im,
                                    alpha_background=alpha_background,
                                    lasso_color=lasso_color,
                                    lasso_linewidth=lasso_linewidth)
    plt.show()
    
    return selector


def is_boundary_bin(arr, 
                 i, 
                 j, 
                 relevant_annotations):

    """
    boolean check if a bin is a boundary bin
    """
    if arr[i][j] in relevant_annotations:
        for ii in range(max(0, i-1), min(i+2, len(arr))):
            for jj in range(max(0, j-1), min(j+2, len(arr[0]))):
                if arr[i][j] != arr[ii][jj] and arr[ii][jj] in relevant_annotations and not pd.isna(arr[ii][jj]):
                    return True
    
    return False


def boundary_recursion(is_boundary_arr, 
                       to_check, 
                       checked, 
                       boundary_range):
    """
    detects all bins belonging to the same boundary
    """
    boundary_elements = []
    while len(to_check) != 0:
        checked.add(to_check[0])
        i = int(to_check[0].split("_")[0])
        j = int(to_check[0].split("_")[1])
        
        if is_boundary_arr[i][j]:
            boundary_elements.append(to_check[0])
            ############# circular check for nearest other boundary bin
            
            rr, cc = disk((i, j), boundary_range+2) # boundary_range=1 means it can skip empty field. skipping none: boundary_range=0. if disk((x,y)n) n=1: will only return the coordinates [x,y                for k in range(len(rr)):
            for k in range(len(rr)):
                if rr[k] < len(is_boundary_arr) and cc[k] < len(is_boundary_arr[0]) and rr[k] >= 0 and cc[k] >= 0:  # negative integers makes it wrap around
                    if f"{rr[k]}_{cc[k]}" not in checked and f"{rr[k]}_{cc[k]}" not in to_check:
                        to_check.append(f"{rr[k]}_{cc[k]}")

        del to_check[0]
    
    return boundary_elements, checked


def create_new_obs(boundaries_arr_bool,
                   df,
                   is_boundary_key,
                   x_column="x",
                   y_column="y"):
    
    """
    creates new anndata.obs
    helper function for find_boundary_distances
    """
    
    boundaries_arr_bool = pd.DataFrame(data=boundaries_arr_bool,    # values
                        index=np.sort(df[y_column].unique()),    # 1st column as index
                        columns=np.sort(df[x_column].unique()))  # 1st row as the column names
    boundaries_arr_bool = boundaries_arr_bool.stack().reset_index(name=is_boundary_key)
    boundaries_arr_bool = boundaries_arr_bool.rename(columns={"level_0": y_column, "level_1": x_column})
    
    df["_bin_names"] = df.index
    new_obs = pd.merge(df, boundaries_arr_bool,  how='left', left_on=[x_column,y_column], right_on = [x_column,y_column])
    new_obs.index = new_obs["_bin_names"]
    new_obs = new_obs.drop('_bin_names', axis=1)

    return new_obs


def calculate_dist_gene_df(anndata_raw, 
                           relevant_clusters, 
                           boundary_dist_key="boundary_dist", 
                           filtered_clusters_key="filtered_annotated", 
                           bin_count_key = "bin_count",
                           masking_key=None):

    """
    calculates average expression at the different distances
    helper function for create_dist_gene_table
    """
    
    if len(relevant_clusters) < 2 < len(relevant_clusters):
        raise ValueError("Wrong number of relevant clusters!")
    obs = copy.deepcopy(anndata_raw.obs)
    if masking_key != None:
        obs[boundary_dist_key] = np.where(obs[masking_key] == True,
                                          obs[boundary_dist_key],
                                          np.nan)
        
    obs[boundary_dist_key] = obs[boundary_dist_key].astype("category")
    dist_gene_df = pd.DataFrame(columns=anndata_raw.var_names, index=obs[boundary_dist_key].cat.categories)
    stdev_gene_df = pd.DataFrame(columns=anndata_raw.var_names, index=obs[boundary_dist_key].cat.categories)
    
    obs[boundary_dist_key] = obs[boundary_dist_key].astype("category")
    anndata_raw.obs = obs
    
    for distance in anndata_raw.obs[boundary_dist_key].astype("category").cat.categories:
        dist_gene_df.loc[distance] = anndata_raw[anndata_raw.obs[boundary_dist_key].isin([distance]),:].layers["normalized"].mean(axis=0)
        x = anndata_raw[anndata_raw.obs[boundary_dist_key].isin([distance]),:].layers["normalized"].todense()
        stdev_gene_df.loc[distance] = np.median(np.absolute(x - np.median(x, axis=0)),axis=0)
    dist_gene_df.index = dist_gene_df.index.astype("int")
    stdev_gene_df.index = stdev_gene_df.index.astype("int")
    
    dist_gene_df[bin_count_key] = anndata_raw.obs[boundary_dist_key].value_counts()
    
    column_to_move = dist_gene_df.pop(bin_count_key)
    dist_gene_df.insert(0, bin_count_key, column_to_move)
    
    return dist_gene_df, stdev_gene_df


def create_dist_gene_table(anndata, 
                           relevant_clusters, 
                           raw_counts_layer="raw_counts", 
                           boundary_dist_key="boundary_dist", 
                           normalization_target_sum=1e6, 
                           filepath=None, 
                           masking_key=None,
                           filtered_clusters_key="filtered_annotated"):
    """
    create workflow output
    write output tsv
    """
    
    anndata_raw = anndata.raw.to_adata()
    anndata_raw.layers[raw_counts_layer] = copy.deepcopy(anndata_raw.X)

    ###
    anndata_raw.obs[boundary_dist_key] = np.where(anndata_raw.obs[filtered_clusters_key] == relevant_clusters[0],
                                              anndata_raw.obs[boundary_dist_key],
                                              anndata_raw.obs[boundary_dist_key]*-1)
    ###
    
    sc.pp.normalize_total(anndata_raw, target_sum=normalization_target_sum)
    anndata_raw.layers["normalized"] = copy.deepcopy(anndata_raw.X)
    dist_gene_df, stdev_gene_df = calculate_dist_gene_df(anndata_raw, 
                                                         relevant_clusters, 
                                                         boundary_dist_key=boundary_dist_key,
                                                         masking_key=masking_key)
    
    stdev_gene_df = stdev_gene_df.add_suffix("__stdev") # make columns distinguishable by adding __stdev suffic
    conc = pd.concat([dist_gene_df, stdev_gene_df], axis=1)
    
    new_col_order = [x for x in itertools.chain.from_iterable(itertools.zip_longest(dist_gene_df.columns, stdev_gene_df.columns)) if x] # adapted from # https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-interleave-interlace-intertwine-two-lists-in-an-alte
    conc = conc[new_col_order] # sort columns, so that expression values and stdev values are always neighboring columns
    
    if filepath is not None:
        with open(filepath, 'w') as f:
            f.write(f'# {relevant_clusters[1]}: negative distances, {relevant_clusters[0]}: positive distances\n')
            conc.to_csv(f, 
                                sep="\t",
                                header=True,
                                decimal=".")
        f.close()
    
    return conc



class SelectFromCollection:
    # based on https://matplotlib.org/stable/gallery/widgets/polygon_selector_demo.html
    
    def __init__(self, 
                 ax, 
                 collection, 
                 alpha_background=0.3,
                 lasso_color="red",
                 lasso_linewidth=2
                ):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_background
        
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        
        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        
        self.lasso = LassoSelector(ax, 
                                   onselect=self.onselect,
                                   props={'color': lasso_color, 'linewidth': lasso_linewidth, 'alpha': 1})
        self.ind = []
    
    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
    
    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
