# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from epynet import Network


# ============================================================
# Command line arguments
# ============================================================
parser = argparse.ArgumentParser(description="Plot WDS topology and observed signal heatmap.")

parser.add_argument(
    '--wds',
    default='anytown',
    type=str,
    help='Water distribution system name (without .inp)'
)

parser.add_argument(
    '--obsrat',
    default=0.0,
    type=float,
    help='Observation ratio for randomly selecting observed junctions.'
)

parser.add_argument(
    '--seed',
    default=None,
    type=int,
    help='Random seed for observed-node sampling.'
)

parser.add_argument(
    '--savepdf',
    action='store_true',
    help='If set, save figures as PDF instead of showing them.'
)

parser.add_argument(
    '--paint_nodes',
    nargs='+',
    default=None,
    help='List of junction UIDs to color explicitly. Example: --paint_nodes 12 25 40'
)

parser.add_argument(
    '--legend',
    action='store_true',
    help='Show topology legend (pump, reservoir, master nodes).'
)

parser.add_argument(
    '--label_junctions',
    action='store_true',
    help='Show junction IDs next to nodes (only applied to Anytown).'
)

# -----------------------------
# New styling arguments
# -----------------------------
parser.add_argument(
    '--highlight_color',
    default='dodgerblue',
    type=str,
    help='Color used for explicitly highlighted nodes.'
)

parser.add_argument(
    '--highlight_scale',
    default=1.6,
    type=float,
    help='Multiplier for the marker size of highlighted nodes.'
)

parser.add_argument(
    '--pipe_lw',
    default=2.0,
    type=float,
    help='Line width for pipes.'
)

parser.add_argument(
    '--pump_lw',
    default=2.4,
    type=float,
    help='Line width for pump links.'
)

parser.add_argument(
    '--valve_lw',
    default=2.4,
    type=float,
    help='Line width for valve links.'
)

parser.add_argument(
    '--node_edge_lw',
    default=0.8,
    type=float,
    help='Marker edge width for nodes.'
)

parser.add_argument(
    '--junction_ms',
    default=5.0,
    type=float,
    help='Marker size for junctions.'
)

parser.add_argument(
    '--tank_ms',
    default=10.0,
    type=float,
    help='Marker size for tanks.'
)

parser.add_argument(
    '--reservoir_ms',
    default=6.0,
    type=float,
    help='Marker size for reservoirs.'
)

parser.add_argument(
    '--pump_circle_ms',
    default=10.0,
    type=float,
    help='Marker size for pump center circle.'
)

parser.add_argument(
    '--pump_arrow_ms',
    default=6.0,
    type=float,
    help='Marker size for pump arrow symbol.'
)

parser.add_argument(
    '--fig_w',
    default=9.0,
    type=float,
    help='Figure width for topology plot.'
)

parser.add_argument(
    '--fig_h',
    default=7.0,
    type=float,
    help='Figure height for topology plot.'
)

parser.add_argument(
    '--label_fs',
    default=16,
    type=float,
    help='Font size for junction labels.'
)

parser.add_argument(
    '--label_dx',
    default=5.0,
    type=float,
    help='Horizontal offset for junction labels.'
)

parser.add_argument(
    '--label_dy',
    default=8.0,
    type=float,
    help='Vertical offset for junction labels.'
)

args = parser.parse_args()

base_dir = os.path.dirname(os.path.realpath(__file__))
inp_path = os.path.abspath(os.path.join(base_dir, '..', 'water_networks', f'{args.wds}.inp'))

if not os.path.exists(inp_path):
    raise FileNotFoundError(
        f"Could not find WDS input file:\n{inp_path}\n\n"
        f"Check that '{args.wds}.inp' exists in the water_networks folder."
    )

# Load network
wds = Network(inp_path)
wds.solve()

# Fix random seed ONCE
if args.seed is not None:
    np.random.seed(args.seed)

# Auxiliar functions
def get_node_df(elements, get_head=False):
    """
    Build a dataframe of nodes (junctions/tanks/reservoirs)
    with coordinates and optionally normalized head.
    """
    data = []
    for elem in elements:
        row = {
            'uid': elem.uid,
            'x': elem.coordinates[0],
            'y': elem.coordinates[1],
        }
        if get_head:
            row['head'] = elem.head
        data.append(row)

    df = pd.DataFrame(data)

    if get_head and not df.empty:
        hmin = df['head'].min()
        hmax = df['head'].max()
        if hmax > hmin:
            df['head'] = (df['head'] - hmin) / (hmax - hmin)
        else:
            df['head'] = 0.0

    return df


def get_elem_df(elements, nodes):
    """
    Build a dataframe of links (pipes/pumps/valves)
    with start/end coordinates, centers and orientation.
    """
    data = []

    for elem in elements:
        from_row = nodes.loc[nodes['uid'] == elem.from_node.uid]
        to_row = nodes.loc[nodes['uid'] == elem.to_node.uid]

        if from_row.empty or to_row.empty:
            continue

        x1 = from_row['x'].values[0]
        y1 = from_row['y'].values[0]
        x2 = to_row['x'].values[0]
        y2 = to_row['y'].values[0]

        data.append({
            'uid': elem.uid,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        })

    df = pd.DataFrame(data)

    if not df.empty:
        df['center_x'] = (df['x1'] + df['x2']) / 2
        df['center_y'] = (df['y1'] + df['y2']) / 2
        # arctan2 avoids division-by-zero issues
        df['orient'] = np.degrees(np.arctan2(df['y2'] - df['y1'], df['x2'] - df['x1'])) + 90

    return df


def build_lc_from(df):
    """
    Build a LineCollection-compatible list of segments.
    """
    if df.empty:
        return []

    line_collection = []
    for _, row in df.iterrows():
        line_collection.append([
            (row['x1'], row['y1']),
            (row['x2'], row['y2'])
        ])
    return line_collection

# Build topology data

nodes = get_node_df(wds.nodes, get_head=True)
juncs = get_node_df(wds.junctions, get_head=True)
tanks = get_node_df(wds.tanks)
reservoirs = get_node_df(wds.reservoirs)

pipes = get_elem_df(wds.pipes, nodes)
pumps = get_elem_df(wds.pumps, nodes)
valves = get_elem_df(wds.valves, nodes)

pipe_collection = build_lc_from(pipes)
pump_collection = build_lc_from(pumps)
valve_collection = build_lc_from(valves) if not valves.empty else []

# Plot topology

fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))

# Pipes
if pipe_collection:
    lc = mc.LineCollection(
        pipe_collection,
        linewidths=args.pipe_lw,
        color='#6a6e73', #k
        zorder=1
    )
    ax.add_collection(lc)

# Pumps
if pump_collection:
    lc = mc.LineCollection(
        pump_collection,
        linewidths=args.pump_lw,
        color='k',
        zorder=2
    )
    ax.add_collection(lc)

# Valves
if valve_collection:
    lc = mc.LineCollection(
        valve_collection,
        linewidths=args.valve_lw,
        color='k',
        zorder=2
    )
    ax.add_collection(lc)

cmap = plt.get_cmap('plasma')

# Slightly emphasize head differences visually
if not juncs.empty and 'head' in juncs.columns:
    juncs['head'] = np.clip(juncs['head'] * 1.5, 0, 1)

colors = []
signal = []

# ------------------------------------------------------------
# Explicit junction highlighting (if provided)
# If --paint_nodes is used:
#   - those nodes are painted in blue (or chosen highlight color)
#   - those nodes are slightly larger
# Otherwise:
#   - observed nodes are chosen randomly with obsrat
#   - observed nodes are colored with the plasma colormap
# ------------------------------------------------------------
paint_nodes_set = None
if args.paint_nodes is not None:
    paint_nodes_set = {str(uid) for uid in args.paint_nodes}
    print(f"Highlighting {len(paint_nodes_set)} user-specified junction(s): {sorted(paint_nodes_set)}")

    # Optional validation
    available_junctions = {str(uid) for uid in juncs['uid'].tolist()}
    missing = sorted(paint_nodes_set - available_junctions)
    if missing:
        print("Warning: these requested junction IDs were not found in the network:")
        print(missing)

for _, junc in juncs.iterrows():
    junc_uid = str(junc['uid'])

    # --------------------------------------------------------
    # Case A: explicit nodes to highlight
    # --------------------------------------------------------
    if paint_nodes_set is not None:
        is_highlighted = junc_uid in paint_nodes_set

        if is_highlighted:
            color = args.highlight_color
            marker_size = args.junction_ms * args.highlight_scale
            signal.append(junc['head'])   # keep head value for heatmap if desired
        else:
            color = (1.0, 1.0, 1.0, 1.0)
            marker_size = args.junction_ms
            signal.append(np.nan)

    # --------------------------------------------------------
    # Case B: random observed nodes using obsrat
    # --------------------------------------------------------
    else:
        is_observed = np.random.rand() < args.obsrat

        if is_observed:
            color = cmap(junc['head'])
            marker_size = args.junction_ms
            signal.append(junc['head'])
        else:
            color = (1.0, 1.0, 1.0, 1.0)
            marker_size = args.junction_ms
            signal.append(np.nan)

    colors.append(color)

    ax.plot(
        junc['x'], junc['y'],
        'o',
        color='k',
        markerfacecolor=color,
        markeredgecolor='k',
        ms=marker_size,
        mew=args.node_edge_lw,
        zorder=3
    )

# ============================================================
# Junction labels (ONLY for Anytown)
# ============================================================
show_anytown_labels = (args.wds.lower() == 'anytown') and args.label_junctions

if show_anytown_labels:
    paint_nodes_set = {str(uid) for uid in args.paint_nodes} if args.paint_nodes is not None else set()

    for _, junc in juncs.iterrows():
        junc_uid = str(junc['uid'])

        # Opcional: resaltar en azul las etiquetas de los master nodes
        if junc_uid in paint_nodes_set:
            txt_color = args.highlight_color
            txt_weight = 'bold'
        else:
            txt_color = 'black'
            txt_weight = 'normal'

        ax.text(
            junc['x'] + args.label_dx,
            junc['y'] + args.label_dy,
            junc_uid,
            fontsize=args.label_fs,
            color=txt_color,
            fontweight=txt_weight,
            ha='left',
            va='bottom',
            zorder=10
        )

# Tanks
for _, tank in tanks.iterrows():
    ax.plot(
        tank['x'], tank['y'],
        marker=7,
        mfc='k',
        mec='k',
        ms=args.tank_ms,
        mew=args.node_edge_lw,
        zorder=4
    )

# Reservoirs
for _, reservoir in reservoirs.iterrows():
    ax.plot(
        reservoir['x'], reservoir['y'],
        marker='o',
        mfc='k',
        mec='k',
        ms=args.reservoir_ms,
        mew=args.node_edge_lw,
        zorder=4
    )

# Pumps as symbols
if not pumps.empty:
    ax.plot(
        pumps['center_x'],
        pumps['center_y'],
        'o',
        color='k',
        ms=args.pump_circle_ms,
        mfc='white',
        mew=args.node_edge_lw,
        zorder=5
    )

    for _, pump in pumps.iterrows():
        ax.plot(
            pump['center_x'],
            pump['center_y'],
            marker=(3, 0, pump['orient']),
            color='k',
            ms=args.pump_arrow_ms,
            zorder=6
        )

# ============================================================
# Legend
# ============================================================
if args.legend:
    legend_handles = []

    # Pump symbol (triangle-like marker)
    pump_handle = Line2D(
        [0], [0],
        marker='v',
        color='k',
        linestyle='None',
        markersize=7,
        markerfacecolor='k',
        label='Pump'
    )
    legend_handles.append(pump_handle)

    # Reservoir symbol (black filled circle)
    reservoir_handle = Line2D(
        [0], [0],
        marker='o',
        color='k',
        linestyle='None',
        markersize=6,
        markerfacecolor='k',
        markeredgecolor='k',
        label='Reservoir'
    )
    legend_handles.append(reservoir_handle)

    # Master nodes (highlighted nodes)
    if args.paint_nodes is not None:
        master_handle = Line2D(
            [0], [0],
            marker='o',
            color='k',
            linestyle='None',
            markersize=args.junction_ms * args.highlight_scale,
            markerfacecolor=args.highlight_color,
            markeredgecolor='k',
            label='Master nodes'
        )
        legend_handles.append(master_handle)

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='black',
        fontsize=10,
        title=None
    )



ax.autoscale()
ax.axis('off')
plt.tight_layout()

# ============================================================
# Save/show topology
# ============================================================
if args.savepdf:
    topo_name = f'topo-{args.wds}.pdf'
    fig.savefig(topo_name, format='pdf', bbox_inches='tight')
    print(f"Saved: {topo_name}")
else:
    plt.show()
# ============================================================
# Save/show signal heatmap
# ============================================================
if args.savepdf:
    signal_name = f'signal-{args.wds}.pdf'
    fig2.savefig(signal_name, format='pdf', bbox_inches='tight')
    print(f"Saved: {signal_name}")
else:
    plt.show()