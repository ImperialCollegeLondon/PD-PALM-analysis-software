import os
import re
import numpy as np
import pandas as pd
from glob import glob
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Union
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")


@dataclass
class RoiSquare:
    id: str
    x: int
    y: int
    unit: str


def qucik_check(bool_1, bool_2):
    if bool_1 is False and bool_2 is False:
        print(
            "**Warning**: Files will be saved in the output folder without any folder structure. There will be no indication of experiment origin)"
        )
        print(
            "Consider setting save_files_with_experiment_name to True or save_in_nested_folder to True"
        )


def find_cell_area_files_and_prefix(path: str, file_type: str = "csv") -> List[str]:
    all_files = glob(os.path.join(path, f"*.{file_type}"))
    cell_area_files = [re.findall(r".*C.*A\d+.csv", file) for file in all_files]
    all_prefixes = [
        re.findall(r"^(.*?)C.*A.*\.csv$", file.split("/")[-1])[0] for file in all_files
    ]
    combined = list(zip(cell_area_files, all_prefixes))
    grouped_files = {}

    for file, prefix in combined:
        if prefix == "":
            prefix = "no_prefix"
        if prefix[-1] == "_":
            prefix = prefix[:-1]

        if prefix not in grouped_files:
            grouped_files[prefix] = file
        else:
            grouped_files[prefix].extend(file)

    text = "The following prefixes were found: "
    for key in grouped_files.keys():
        text += f"\n\t{key}\t|\t{len(grouped_files[key])} files"
    print(text)

    return grouped_files


def group_cell_area_files_by_cell(path: str, path_ending: str = ".csv") -> dict:
    files = glob(os.path.join(path, f"*{path_ending}"))

    grouped_strings = defaultdict(list)
    for s in files:
        match = re.search(r"C(\d+)", s)
        if match:
            number_after_C = match.group(1)
            grouped_strings[number_after_C].append(s)

    grouped_strings = dict(grouped_strings)
    return grouped_strings


def prepare_roi_data(
    roi_df: Union[pd.DataFrame, str], unit_to_extract: str = "px"
) -> List[RoiSquare]:
    if isinstance(roi_df, str):
        roi_df = pd.read_csv(roi_df)
    roi_t = roi_df.T
    roi_t.columns = roi_t.iloc[0]
    roi_t.columns.name = None
    roi_t = roi_t[1:]
    roi_t.dropna(axis=1, how="all", inplace=True)
    possible_units = np.unique([re.findall(r"\((.*?)\)", x) for x in roi_t.index])
    if unit_to_extract not in possible_units:
        raise ValueError(
            f"Unit {unit_to_extract} not found in the ROI data. Possible units: {possible_units}"
        )

    roi_squares = []
    for col in roi_t.columns:
        roi_squares.append(
            RoiSquare(
                col,
                roi_t[col][f"X ({unit_to_extract})"],
                roi_t[col][f"Y ({unit_to_extract})"],
                unit_to_extract,
            )
        )

    return roi_squares


def make_square(
    df: pd.DataFrame, roi: RoiSquare, side_length: Union[int, float]
) -> pd.DataFrame:
    x_col, y_col = f"X ({roi.unit})", f"Y ({roi.unit})"

    square_df = df.query(
        f"`{x_col}` >= {roi.x} & `{x_col}` <= {roi.x + side_length} & `{y_col}` >= {roi.y} & `{y_col}` <= {roi.y + side_length}"
    )
    return square_df


def make_squares(
    df: pd.DataFrame, rois: List[RoiSquare], side_length: Union[int, float]
) -> Dict[str, pd.DataFrame]:
    squares = {}
    for roi in rois:
        squares[roi.id] = make_square(df, roi, side_length)
    return squares


def update_output_folder(output_folder: str, suffix: str) -> str:
    return os.path.join(output_folder, suffix)


def save_squares(
    squares: Dict[str, pd.DataFrame],
    output_folder: str,
    prefix: str = "",
    overwrite: bool = False,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for key, value in squares.items():
        save_path = os.path.join(output_folder, f"{prefix}{key}.csv")
        if os.path.exists(save_path) and not overwrite:
            continue

        value.to_csv(os.path.join(output_folder, f"{prefix}{key}.csv"), index=False)


def run_squares_pipeline(
    data_file: str,
    ROI_file: str,
    output_folder: str,
    save_in_nested_folder: bool = False,
    length: Union[int, float] = 10,
    unit_to_extract: str = "nm",
    save_files_with_experiment_name: bool = False,
    overwrite: bool = False,
) -> None:
    experiment_name = os.path.basename(data_file).split(".")[0]

    if save_in_nested_folder:
        output_folder = update_output_folder(output_folder, experiment_name)
    data = pd.read_csv(data_file)
    ROI = pd.read_csv(ROI_file)
    roi = prepare_roi_data(ROI, unit_to_extract=unit_to_extract)
    square = make_squares(data, roi, length)
    file_prefix = experiment_name + "_" if save_files_with_experiment_name else ""
    save_squares(square, output_folder, prefix=file_prefix, overwrite=overwrite)
    print("Saved squares for", experiment_name)


# --------- EXISTING CODE ------------ #
def removeblinks(
    df,
    idcol="receptor_id",
    framecol="Frame",
    channel="Channel",
    blink_threshold=100,
    frame_gap=3,
):
    dframe = df[["Channel", "Frame"]]  # Take Frame and Channel information
    nn_blink = calculate_nearest_neighbours(df, blink_threshold)
    nn = create_all_clusters(nn_blink)

    # Create Dataset to find those that blink close together
    clusters = pd.DataFrame([], columns=["clusterID", idcol, framecol, channel])

    x = 1
    y = 1
    for i in range(len(nn)):
        for j in nn[i]:
            c, f = dframe.loc[j]
            data = np.array([(x, j, f, c)])
            d = pd.DataFrame(data, columns=["clusterID", idcol, framecol, channel])
            objs = [clusters, d]
            clusters = pd.concat(objs, ignore_index=True)

        x += 1  # Increase Cluster Number after Iteration

    # Sort Final Dataset
    clusters = clusters.sort_values(by=["clusterID", channel, framecol], ascending=True)
    # Calculate Difference between Frames
    clusters["diffs"] = clusters.groupby(["clusterID", channel])[framecol].transform(
        lambda x: x.diff()
    )
    toremove = clusters[
        (clusters["diffs"] <= frame_gap) & (clusters["diffs"].isna() == False)
    ]  # Remove if below defined value

    # Merge Back on to Dataset and Delete rows in toremove dataset

    final = df.merge(
        toremove,
        left_on=[channel, idcol, framecol],
        right_on=[channel, idcol, framecol],
        how="left",
    )
    final = final[final["diffs"].isna() == True]

    return final


def calculate_nearest_neighbours(df, gap=50, X_COL="X (nm)", Y_COL="Y (nm)"):
    # Calculate the nearest neighbours for each point in the dataframe.
    neighbours = NearestNeighbors(radius=gap)
    neighbours.fit(df[[X_COL, Y_COL]])
    # radius_neighbours finds all points within the specified radius.
    nearest_neighbours = neighbours.radius_neighbors(
        df[[X_COL, Y_COL]], return_distance=False
    )
    return nearest_neighbours


def create_all_clusters(nearest_neighbours):
    all_clusters = []
    indexes_checked = set()
    # For each point, find all connected points to create cluster and then exclude from future search.
    for i in range(len(nearest_neighbours)):
        if i not in indexes_checked:
            need_to_check = {i}
            new_cluster = set()
            while len(need_to_check) > 0:
                point = need_to_check.pop()
                point_neighbours = set(nearest_neighbours[point].tolist())
                new_cluster.add(point)
                point_neighbours.difference_update(new_cluster)
                need_to_check.update(point_neighbours)

            all_clusters.append(new_cluster)
            indexes_checked.update(new_cluster)

    return [list(c) for c in all_clusters]


def create_clusters(
    df, fname, BLINK_THRESHOLD=100, FRAME_GAP=3, CLUSTER_DISTANCE_THRESHOLD=50
):
    blinks = removeblinks(df, frame_gap=FRAME_GAP, blink_threshold=BLINK_THRESHOLD)
    blinks.to_csv(fname)  ## Save dataset to CSV in current working directorty
    # print(f"Calculated nearest neighbours within {CLUSTER_DISTANCE_THRESHOLD} (nm).")
    neighbours = calculate_nearest_neighbours(blinks, CLUSTER_DISTANCE_THRESHOLD)
    clusters = create_all_clusters(neighbours)
    # print(f"Created {len(clusters)} clusters.")
    df["cluster_id"] = None
    # Update the dataframe with the same cluster id for each point in a cluster.
    for i in range(len(clusters)):
        cluster = clusters[i]
        cluster_id = str(int(i))
        for index in cluster:
            df.at[index, "cluster_id"] = cluster_id
    return df


def group_by_cluster_id(df, column_name="receptors_count"):
    cluster_group = (
        df[["receptor_id", "cluster_id"]]
        .groupby("cluster_id")
        .count()
        .reset_index()
        .rename(columns={"receptor_id": column_name})
    )
    return cluster_group


def get_cluster_frequencies(df, channel_name):

    cluster_frequency = (
        df.groupby("receptors_count")
        .count()
        .reset_index()
        .rename(
            columns={
                "receptors_count": "multimer_size",
                "cluster_id": f"channel_{channel_name}",
            }
        )
    )
    return cluster_frequency


# --------- END OF EXISTING CODE ------------ #


def run_single_file_blink_analysis(
    file_path: str,
    blink_folder: str,
    output_changes: str,
    pivot_output: str,
    CLUSTER_DISTANCE_THRESHOLD: int,
    BLINK_THRESHOLD: int,
    FRAME_GAP: int,
):
    """
    Runs blink analysis on a single file.

    Args:
        file_path (str): The path to the input file.
        blink_folder (str): The folder to save the blink files.
        output_changes (str): The folder to save the output changes.
        pivot_output (str): The folder to save the pivot table.
        CLUSTER_DISTANCE_THRESHOLD (int): The threshold for cluster distance.
        BLINK_THRESHOLD (int): The threshold for blink detection.
        FRAME_GAP (int): The gap between frames.

    Returns:
        tuple: A tuple containing the multimer summary and pivot table.
    """
    df = pd.read_csv(file_path)
    df.index.name = "receptor_id"
    df_grouped = df.groupby("Channel")
    summary = []
    for channel, sub_group in df_grouped:
        sub_group.reset_index(inplace=True)

        df_channel = create_clusters(
            sub_group,
            os.path.join(
                blink_folder,
                f'{file_path.split("/")[-1].split(".")[0]}_blinks_{channel}.csv',
            ),
            CLUSTER_DISTANCE_THRESHOLD=CLUSTER_DISTANCE_THRESHOLD,
            BLINK_THRESHOLD=BLINK_THRESHOLD,
            FRAME_GAP=FRAME_GAP,
        )
        channel_grp = group_by_cluster_id(df_channel)
        channel_freq = get_cluster_frequencies(channel_grp, channel)
        summary.append(channel_freq)

    multimer_summary = summary[0]

    if len(summary) > 1:  # We need merge and pivot table
        multimer_summary = pd.merge(
            *summary, left_on="multimer_size", right_on="multimer_size", how="outer"
        )

        all_channels = create_clusters(
            df,
            os.path.join(
                blink_folder,
                f'{file_path.split("/")[-1].split(".")[0]}_blinks_all_channels.csv',
            ),
            CLUSTER_DISTANCE_THRESHOLD=CLUSTER_DISTANCE_THRESHOLD,
            BLINK_THRESHOLD=BLINK_THRESHOLD,
            FRAME_GAP=FRAME_GAP,
        )
        all_channels.reset_index(inplace=True)
        all_channels_grouped = all_channels.groupby("Channel")
        groups = all_channels_grouped.apply(
            lambda x: group_by_cluster_id(x, "receptor_count_channel_")
        )

        groups = groups.groupby(level=0)
        itteration_list = [[name, grp] for name, grp in groups]
        names = [itt[0] for itt in itteration_list]
        grp_list = [itt[1] for itt in itteration_list]
        if len(grp_list) == 1:
            dummy_grp = grp_list[0].copy().reset_index()
            dummy_grp["Channel"] += 1
            dummy_grp["receptor_count_channel_"] *= 0
            dummy_grp.index = pd.MultiIndex.from_frame(
                dummy_grp[["Channel", "level_1"]]
            )
            dummy_grp.drop(columns=["Channel", "level_1"], inplace=True)
            names.append(names[0] + 1)
            grp_list.append(dummy_grp)

        merged_all_channels = (
            pd.merge(
                *grp_list,
                left_on="cluster_id",
                right_on="cluster_id",
                how="outer",
                suffixes=names,
            )
            .fillna(0)
            .astype(int)
        )
        merged_all_channels["total_receptor_count"] = merged_all_channels.values[
            :, 1:
        ].sum(axis=1)
        both_chanel_frequencies = merged_all_channels[
            (merged_all_channels["receptor_count_channel_0"] > 0)
            & (merged_all_channels["receptor_count_channel_1"] > 0)
        ]
        
        both_chanel_frequencies = (
            both_chanel_frequencies[["cluster_id", "total_receptor_count"]]
            .groupby("total_receptor_count")
            .count()
            .reset_index()
            .rename(
                columns={
                    "total_receptor_count": "multimer_size",
                    "cluster_id": "channel_0_and_1",
                }
            )
        )
        multimer_summary = (
            multimer_summary.merge(
                both_chanel_frequencies,
                left_on="multimer_size",
                right_on="multimer_size",
                how="outer",
            )
            .fillna(0)
            .astype(int)
        )

        pivot_table = (
            pd.pivot_table(
                merged_all_channels.reset_index(),
                values="cluster_id",
                index="receptor_count_channel_1",
                columns="receptor_count_channel_0",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )

        pivot_table.to_csv(
            os.path.join(
                pivot_output,
                f"{file_path.split('/')[-1].split('.')[0]}_pivot_table.csv",
            ),
        )
    multimer_summary.to_csv(
        os.path.join(
            output_changes,
            f"{file_path.split('/')[-1].split('.')[0]}_multimer_summary.csv",
        )
    )
    return None


def collect_multimer_files(files: List[str], max_multimer: int, remove_above_max: bool):
    """
    Collects multimer files into a list, filtering out files with multimer sizes above a specified maximum.

    Args:
        files (List[str]): List of file paths to read.
        max_multimer (int): Maximum multimer size to include in the collected files.
        remove_above_max (bool): Flag indicating whether to remove files with multimer sizes above the maximum.

    Returns:
        List[pd.DataFrame]: List of DataFrames containing the collected multimer files.
    """
    df_list = []

    for f in files:  # Read each file for the cell and collect them into a list
        df = pd.read_csv(f, index_col=0).fillna(0).astype(int)
        df.reset_index(inplace=True, drop=True)
        below_max = df.query("multimer_size <= @max_multimer").copy()

        if df.multimer_size.max() > max_multimer and not remove_above_max:
            above_max = df.query("multimer_size > @max_multimer").copy()
            sum_above_max = above_max.sum(axis=0).drop("multimer_size")
            sum_above_max["multimer_size"] = 0
            below_max.iloc[-1] = below_max.iloc[-1] + sum_above_max[below_max.columns]

        df_list.append(below_max)

    return df_list


def cellect_pivot_files(files: List[str]):
    df = pd.DataFrame()
    for file in files:
        temp_df = pd.read_csv(file, index_col=0)

        if df.empty:
            df = temp_df.fillna(0)
        else:
            df = df.add(temp_df, fill_value=0)
    df.columns.name = "receptor_count_channel_0"
    df.fillna(0, inplace=True)
    return df


def find_percentage_from_pivot(df: pd.DataFrame):
    multimer = (
        df.index.astype(int).values[:, None] + df.columns.values.astype(int)[None, :]
    )
    multimer_df = (
        pd.DataFrame(df.values.flatten(), index=multimer.flatten())
        .groupby(level=0)
        .sum()
    )
    multimer_mapping = pd.DataFrame(
        multimer, index=df.index, columns=df.columns
    ).applymap(lambda x: multimer_df.to_dict()[0][x])

    percentage = (df / multimer_mapping).fillna(0)
    return percentage


def create_ratio_from_channel_0(percentage: pd.DataFrame):
    max_multimer = (
        percentage.index.astype(int).max() + percentage.columns.astype(int).max()
    )
    empty_array = np.empty((max_multimer, percentage.columns.size))
    empty_array[:] = np.nan
    empty_df = pd.DataFrame(
        empty_array, index=np.arange(1, max_multimer + 1), columns=percentage.columns
    )
    empty_df.index.name = "multimer_complex"

    for row, col in empty_df.iterrows():
        for column in col.index:
            if row in percentage.index and column in percentage.columns:
                empty_df.at[int(row) + int(column), column] = percentage.at[row, column]
    empty_df.dropna(how="all", inplace=True)
    empty_df.fillna(0, inplace=True)
    empty_df = empty_df.mul(100)
    return empty_df


def analyise_complete_cell_df(
    df: pd.DataFrame, output_path: str, cell: str, percentage_decimals: int = 2
):
    """
    Analyzes a complete cell DataFrame and generates various calculations and ratios.

    Args:
        df (pd.DataFrame): The input DataFrame containing the cell data.
        output_path (str): The path to save the output CSV file.
        cell (str): The name of the cell.
        percentage_decimals (int, optional): The number of decimal places to round the percentage ratios to. Defaults to 2.

    Returns:
        pd.DataFrame: The transposed DataFrame with calculated ratios and totals.
    """
    df_grouped = df.groupby("multimer_size").sum()
    df_collected = df_grouped.reset_index()
    df_collected.sort_values(by="multimer_size", inplace=True)

    scaled = df_collected.mul(df_collected["multimer_size"], axis=0)
    scaled["multimer_size"] = df_collected["multimer_size"]
    total_locs = scaled.sum(axis=0)
    if len(names := total_locs.keys()) > 2:
        total_locs[names[-1]] = total_locs[names[1]] + total_locs[names[2]]
    ratio = scaled.div(total_locs, axis=1)
    # Change ratio to percentage
    ratio = ratio.mul(100)
    if percentage_decimals:
        ratio = ratio.round(percentage_decimals)

    ratio["multimer_size"] = df_collected["multimer_size"]
    ratio.columns = ["multimer_size"] + [
        f"{c}_percentage_ratio" for c in ratio.columns if c != "multimer_size"
    ]
    scaled.columns = ["multimer_size"] + [
        f"{c}_locs" for c in scaled.columns if c != "multimer_size"
    ]
    df_collected = df_collected.merge(scaled, on="multimer_size")
    df_collected = df_collected.merge(ratio, on="multimer_size")
    # Flip df so that multimer_size is columns and columns are rows
    df_collected.set_index("multimer_size", inplace=True)
    df_transposed = df_collected.T
    df_transposed["Total"] = df_transposed.sum(axis=1)
    df_transposed.index.name = "Multimer_complex"
    df_transposed.to_csv(os.path.join(output_path, f"Cell_{cell}.csv"))
    return df_transposed


def get_receptor_ratio_for_cells(X: pd.DataFrame, cell: str):
    X = X.reset_index(level=1, drop=True)
    X.index.name = "Cell"
    X.columns.name = "receptor_count_channel_0"
    X.sort_index(inplace=True)
    X.fillna(0, inplace=True)
    return X
