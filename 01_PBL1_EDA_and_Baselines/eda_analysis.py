import os
import glob
import pickle
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import Counter
from pathlib import Path

def parse_graph_fast(raw_graph):
    G = nx.Graph()
    pos = {}
    if isinstance(raw_graph, dict):
        if not raw_graph:
            return G, pos
        first_key = next(iter(raw_graph.keys()))
        if isinstance(first_key, (tuple, list)) and len(first_key) == 2:
            edges = []
            for u, nbrs in raw_graph.items():
                u_t = tuple(u)
                for v in nbrs:
                    edges.append((u_t, tuple(v)))
            G.add_edges_from(edges)
            pos = {n: n for n in G.nodes()}
        elif any(k in raw_graph for k in ["edges", "edge_list"]):
            G.add_edges_from(raw_graph.get("edges", raw_graph.get("edge_list", [])))
            pos = raw_graph.get("vertices", {n: n for n in G.nodes()})
    elif isinstance(raw_graph, list):
        G.add_edges_from(raw_graph)
        pos = {n: n for n in G.nodes()}
    elif isinstance(raw_graph, nx.Graph):
        G = raw_graph
        pos = {n: (n[0], n[1]) if isinstance(n, tuple) else G.nodes[n].get('pos', (0,0)) for n in G.nodes()}
    return G, pos

def run_fast_eda(dataset_path: str, output_plot_path: str = "20cities_eda_report.png"):
    data_dir = Path(dataset_path)
    sat_files = sorted([f for f in data_dir.glob("*.png") if not any(x in f.name for x in ["_gt", "_mask", "_rgb"])])
    
    total_files = len(sat_files)
    print(f"[*] Found {total_files} tiles. Starting fast analysis...")

    img_shapes = []
    channel_means, channel_stds = [], []
    road_ratios = []
    node_degree_counter = Counter()
    edge_lengths = []
    total_nodes_per_tile, total_edges_per_tile = [], []
    
    first_valid_graph = None
    first_valid_sat = None

    for idx, sat_path in enumerate(sat_files):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[*] Processing tile {idx + 1}/{total_files}...")

        prefix = sat_path.stem.replace("_sat", "")
        mask_path = data_dir / f"{prefix}_gt.png"
        graph_path = data_dir / f"{prefix}_refine_gt_graph.p"

        # 1. Image
        img = cv2.imread(str(sat_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_shapes.append(img.shape)
            channel_means.append(np.mean(img, axis=(0, 1)))
            channel_stds.append(np.std(img, axis=(0, 1)))

        # 2. Mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                road_px = np.count_nonzero(mask > 127)
                road_ratios.append(road_px / (mask.shape[0] * mask.shape[1]))

        # 3. Graph
        if graph_path.exists():
            try:
                with open(graph_path, "rb") as f:
                    raw_graph = pickle.load(f)
                G, pos = parse_graph_fast(raw_graph)
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                
                if n_nodes > 0:
                    total_nodes_per_tile.append(n_nodes)
                    total_edges_per_tile.append(n_edges)
                    node_degree_counter.update([d for _, d in G.degree()])
                    
                    # Sample up to 300 edges per tile for lightning-fast length metrics
                    edges_to_check = list(G.edges())
                    if len(edges_to_check) > 300:
                        edges_to_check = [edges_to_check[i] for i in np.random.choice(len(edges_to_check), 300, replace=False)]
                        
                    for u, v in edges_to_check:
                        if u in pos and v in pos:
                            p1, p2 = np.array(pos[u]), np.array(pos[v])
                            d = np.linalg.norm(p1 - p2)
                            if d > 0:
                                edge_lengths.append(d)

                    if first_valid_graph is None:
                        first_valid_graph = (G, pos)
                        first_valid_sat = sat_path
            except Exception:
                pass

    # Print Report
    print("\n" + "=" * 60)
    print("                 DATASET EDA SUMMARY                 ")
    print("=" * 60)
    print(f"Total Tiles Processed:        {len(sat_files)}")
    if img_shapes:
        print(f"Tile Shape:                   {img_shapes[0]}")
        print(f"RGB Normalization Means:      {np.round(np.mean(channel_means, axis=0), 2).tolist()}")
        print(f"RGB Normalization Stds:       {np.round(np.mean(channel_stds, axis=0), 2).tolist()}")
    if road_ratios:
        print(f"Mean Road Coverage Ratio:     {np.mean(road_ratios)*100:.2f}%")
        print(f"Background-to-Road Ratio:     ~{int((1.0 - np.mean(road_ratios)) / np.mean(road_ratios))}:1")
    if total_nodes_per_tile:
        print(f"Average Nodes per Tile:       {np.mean(total_nodes_per_tile):.1f}")
        print(f"Average Edges per Tile:       {np.mean(total_edges_per_tile):.1f}")
        print(f"Average Edge Length:          {np.mean(edge_lengths):.2f} px")
        print("\nDegree Distribution:")
        tot = sum(node_degree_counter.values())
        for deg in sorted(node_degree_counter.keys()):
            cnt = node_degree_counter[deg]
            print(f"  Degree {deg}: {cnt} ({cnt/tot*100:.1f}%)")
    print("=" * 60)

    # Fast Plotting
    print("\n[*] Rendering dashboard...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"EDA Summary: {data_dir.name}", fontsize=14, fontweight="bold")

    if channel_means:
        axes[0, 0].boxplot([[m[0] for m in channel_means], [m[1] for m in channel_means], [m[2] for m in channel_means]], 
                           tick_labels=['Red', 'Green', 'Blue'], patch_artist=True)
        axes[0, 0].set_title("RGB Channel Intensities")

    if road_ratios:
        axes[0, 1].hist(np.array(road_ratios) * 100, bins=25, color="steelblue", edgecolor="black")
        axes[0, 1].set_title("Road Coverage (%) Distribution")

    if node_degree_counter:
        degs = sorted(node_degree_counter.keys())
        axes[0, 2].bar([str(d) for d in degs], [node_degree_counter[d] for d in degs], color="teal", edgecolor="black")
        axes[0, 2].set_title("Intersection Degree Distribution")

    if edge_lengths:
        filtered = [l for l in edge_lengths if l <= np.percentile(edge_lengths, 98)]
        axes[1, 0].hist(filtered, bins=30, color="salmon", edgecolor="black")
        axes[1, 0].set_title("Segment Length Distribution (px)")

    if total_nodes_per_tile:
        axes[1, 1].scatter(total_nodes_per_tile, total_edges_per_tile, color="darkslateblue", alpha=0.6)
        axes[1, 1].set_title("Nodes vs Edges per Tile")

    # Fast Vector Overlay using LineCollection
    if first_valid_sat and first_valid_graph:
        s_img = cv2.cvtColor(cv2.imread(str(first_valid_sat)), cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(s_img)
        G_v, p_v = first_valid_graph
        segs = [[p_v[u], p_v[v]] for u, v in G_v.edges() if u in p_v and v in p_v]
        lc = LineCollection(segs, colors="lime", linewidths=1.0)
        axes[1, 2].add_collection(lc)
        axes[1, 2].set_title(f"Overlay: {first_valid_sat.name}")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=100)
    print(f"[+] Saved dashboard in <2 seconds: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\raopo\Downloads\data\data\20cities"
    run_fast_eda(DATASET_DIR)