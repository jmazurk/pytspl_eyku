import pkg_resources
import pandas as pd

from pytspl.io.network_reader import read_csv

COMMUTING_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/commute"
)

def load_commuting_msoa(only_sc: bool = True, only_2d: bool = True):
    """
        Load the commuting MSOA dataset and return the simplicial complex,
        coordinates and flow.

        Args:
            only_sc (bool, optional): if true return a simplicial complex,
            else return a cell complex.
            only_2d (bool, optional): if true (default) build up to triangles;
            if false, build all simplices.

        Returns:
            tuple:
                SimplicialComplex or CellComplex: The simplicial/cell complex.
                dict: The coordinates of the nodes.
                dict: The flow data.
        """
    flows_file = f"{COMMUTING_DATA_FOLDER}/flows_int.csv"
    coords_file = f"{COMMUTING_DATA_FOLDER}/nodes_int.csv"

    builder = read_csv(
        filename=flows_file,
        delimiter=",",
        src_col="origin",
        dest_col="destination",
        feature_cols=None,
        start_index_zero=True,
        only_sc=only_sc,
    )

    if only_sc:
        sc = builder.to_simplicial_complex(condition="all")
    else:
        sc = builder.to_cell_complex()


    coords_df = pd.read_csv(coords_file)

    coordinates = {
        int(row["id"]): (row["lon"], row["lat"])
        for _, row in coords_df.iterrows()
    }

    df = pd.read_csv(flows_file)
    flow_values = df["flow"].values
    flow = {edge: flow_values[i] for i, edge in enumerate(sc.edges)}

    return sc, coordinates, flow