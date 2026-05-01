from itertools import combinations

import pkg_resources
import pandas as pd

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.io.network_reader import read_csv

COMMUTING_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/commute"
)

def load_commuting_msoa(only_sc: bool = True, only_2d: bool = True):
    flows_file = f"{COMMUTING_DATA_FOLDER}/flows.csv"
    coords_file = f"{COMMUTING_DATA_FOLDER}/nodes.csv"

    df = pd.read_csv(flows_file)
    coords_df = pd.read_csv(coords_file)

    vertices = sorted(set(df["origin"]).union(df["destination"]))

    edges = list(set(tuple(sorted((row.origin, row.destination))) for _, row in df.iterrows()))

    edge_set = set(edges)
    triangles = []

    for u, v, w in combinations(vertices, 3):
        if (
                (u, v) in edge_set and
                (u, w) in edge_set and
                (v, w) in edge_set
        ):
            triangles.append(tuple(sorted((u, v, w))))

    simplices = {
        0: vertices,
        1: edges,
    }

    if only_sc:
        if only_2d:
            simplices[2] = triangles
        sc = SimplicialComplex(simplices=simplices)
    else:
        sc = SimplicialComplex(simplices=simplices)

    # vertices = sorted(set(df["origin"]).union(df["destination"]))
    # edges = list(set(tuple(sorted((row.origin, row.destination))) for _, row in df.iterrows()))

    # simplices = {
    #     0: vertices,
    #     1: edges,
    # }

    # builder = read_csv(
    #     filename=flows_file,
    #     delimiter=",",
    #     src_col="origin",
    #     dest_col="destination",
    #     feature_cols=None,
    # )
    # builder = read_csv(
    #     filename=flows_file,
    #     delimiter=",",
    #     src_col="origin",
    #     dest_col="destination",
    #     feature_cols=None,
    #     only_sc=only_sc,
    #     only_2d=only_2d,
    # )
    #
    # if only_sc:
    #     sc = builder.to_simplicial_complex(condition="all")
    # else:
    #     sc = builder.to_cell_complex()

    #sc = builder.to_simplicial_complex(condition="all")

    #sc = SimplicialComplex(simplices=simplices)

    sc.print_summary()

    coordinates = {
        row["id"]: (row["lon"], row["lat"])
        for _, row in coords_df.iterrows()
    }

    flow = {
        tuple(sorted((row.origin, row.destination))): row.flow
        for _, row in df.iterrows()
    }

    return sc, coordinates, flow