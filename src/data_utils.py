import math
import os
import sys
import pickle
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MATERIALS_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def _elem_features(element) -> List[float]:
    try:
        ie1 = float(element.ionization_energies[0])
    except (IndexError, TypeError, AttributeError):
        ie1 = 7.0 #rough avg

    try:
        ea = float(element.electron_affinity)
    except (TypeError, AttributeError):
        ea = 0.0

    return [
        float(element.atomic_mass),
        float(element.X) if element.X else 2.0,
        float(element.atomic_radius) if element.atomic_radius else 1.5,
        float(element.mendeleev_no),
        float(element.row),
        float(element.group),
        float(element.Z),
        ie1,
        ea,
        float(element.max_oxidation_state),
        float(element.min_oxidation_state),
    ]

NODE_FEATURE_DIM = 11


def structure_to_graph(structure, cutoff = 5.0):
    try:
        node_features = [_elem_features(site.specie) for site in structure]
        x = torch.tensor(node_features, dtype=torch.float)

        src, dst, dist = [], [], []
        for i, site in enumerate(structure):
            for nbr in structure.get_neighbors(site, cutoff):
                src.append(i)
                dst.append(nbr.index)
                dist.append(nbr.nn_distance)

        if not src:
            return None

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor([[d] for d in dist], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as exc:
        logger.warning("structure_to_graph failed: %s", exc)
        return None


_SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "nsites",
    "volume",
    "density",
    "nelements",
]


def fetch_materials(subset_mode = False, max_samples = None, band_gap_range = (0.5, 3.5)):
    from mp_api.client import MPRester

    effective_max = max_samples if subset_mode else None

    logger.info(
        "Fetching materials (subset_mode=%s, max_samples=%s, band_gap=%s)",
        subset_mode, effective_max if effective_max is not None else "ALL", band_gap_range,
    )

    kwargs: dict = dict(
        band_gap=band_gap_range,
        fields=_SUMMARY_FIELDS,
        all_fields=False,
    )
    if effective_max is not None:
        chunk_size = min(effective_max, 1000)  # server caps _limit at 1000
        kwargs["chunk_size"] = chunk_size
        kwargs["num_chunks"] = math.ceil(effective_max / chunk_size)

    with MPRester(MATERIALS_API_KEY) as mpr:
        docs = mpr.materials.summary.search(**kwargs)

    if effective_max is not None:
        docs = docs[:effective_max]

    logger.info("Retrieved %d documents", len(docs))

    records = []
    for doc in docs:
        records.append(
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "nsites": doc.nsites,
                "volume": doc.volume,
                "density": doc.density,
                "nelements": doc.nelements,
                "structure": doc.structure,
            }
        )

    df = pd.DataFrame(records)
    logger.info("DataFrame shape: %s", df.shape)
    return df



def cache_data(df, path = "data/materials_cache.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached %d rows → %s", len(df), path)


def load_cached_data(path = "data/materials_cache.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    logger.info("Loaded %d rows from cache (%s)", len(df), path)
    return df

FEATURE_NAMES = [
    "nsites", "volume_per_atom", "density", "nelements",
    "mean_atomic_mass",   "std_atomic_mass",   "range_atomic_mass",
    "mean_electronegativity", "std_electronegativity", "range_electronegativity",
    "mean_atomic_radius", "std_atomic_radius", "range_atomic_radius",
    "mean_atomic_number", "std_atomic_number", "range_atomic_number",
    "mean_row",           "std_row",           "range_row",
    "mean_group",         "std_group",         "range_group",
]


def extract_baseline_features(df):
    rows = []
    for _, row in df.iterrows():
        struct = row["structure"]
        elems = [site.specie for site in struct]

        masses  = [float(e.atomic_mass) for e in elems]
        enegats = [float(e.X) if e.X else 2.0 for e in elems]
        radii   = [float(e.atomic_radius) if e.atomic_radius else 1.5 for e in elems]
        znums   = [float(e.Z) for e in elems]
        erows   = [float(e.row) for e in elems]
        groups  = [float(e.group) for e in elems]

        feat = [
            row["nsites"],
            row["volume"] / max(row["nsites"], 1),
            row["density"],
            row["nelements"],
        ]
        for prop in [masses, enegats, radii, znums, erows, groups]:
            arr = np.array(prop)
            feat.extend([arr.mean(), arr.std() if len(arr) > 1 else 0.0, arr.max() - arr.min()])

        rows.append(feat)

    X = np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)
    y_bg = df["band_gap"].values.astype(np.float32)
    y_fe = df["formation_energy_per_atom"].values.astype(np.float32)
    return X, y_bg, y_fe


def build_graph_dataset(df, cutoff = 5.0):
    graphs, skipped = [], 0
    for _, row in df.iterrows():
        g = structure_to_graph(row["structure"], cutoff=cutoff)
        if g is None:
            skipped += 1
            continue
        g.y = torch.tensor(
            [[row["band_gap"], row["formation_energy_per_atom"]]],
            dtype=torch.float,
        )
        g.material_id = row["material_id"]
        g.formula = row["formula"]
        graphs.append(g)

    logger.info("Built %d graphs (%d skipped)", len(graphs), skipped)
    return graphs


def split_dataset(graphs, train_ratio = 0.70, val_ratio = 0.15, seed = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    n_train = int(len(graphs) * train_ratio)
    n_val = int(len(graphs) * val_ratio)
    return (
        [graphs[i] for i in idx[:n_train]],
        [graphs[i] for i in idx[n_train : n_train + n_val]],
        [graphs[i] for i in idx[n_train + n_val :]],
    )
