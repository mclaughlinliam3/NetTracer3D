from PyQt6.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QLabel, 
    QPushButton,
    QMessageBox,
    QFileDialog
)
from PyQt6.QtCore import Qt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os


def convert_to_multigraph(G, weight_attr='weight'):
    """
    Convert weighted graph to MultiGraph by creating parallel edges.
    
    Args:
        G: NetworkX Graph with edge weights representing multiplicity
        weight_attr: Name of the weight attribute (default: 'weight')
    
    Returns:
        MultiGraph with parallel edges instead of weights
    
    Note:
        - Weights are rounded to integers
        - Original node/edge attributes are preserved on first edge
        - Directed graphs become MultiDiGraphs
    """

    MG = nx.MultiGraph()
    
    # Copy nodes with all their attributes
    MG.add_nodes_from(G.nodes(data=True))
    
    # Convert weighted edges to multiple parallel edges
    for u, v, data in G.edges(data=True):
        # Get weight (default to 1 if missing)
        weight = data.get(weight_attr, 1)
        
        # Round to integer for number of parallel edges
        num_edges = int(round(weight))
        
        if num_edges < 1:
            num_edges = 1  # At least one edge
        
        # Create parallel edges
        for i in range(num_edges):
            # First edge gets all the original attributes (except weight)
            if i == 0:
                edge_data = {k: v for k, v in data.items() if k != weight_attr}
                MG.add_edge(u, v, **edge_data)
            else:
                # Subsequent parallel edges are simple
                MG.add_edge(u, v)
    
    return MG

class HistogramSelector(QWidget):
    def __init__(self, network_analysis_instance, stats_dict, G):
        super().__init__()
        self.network_analysis = network_analysis_instance
        self.stats_dict = stats_dict
        self.G_unweighted = G
        self.G = convert_to_multigraph(G)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Network Analysis - Histogram Selector')
        self.setGeometry(300, 300, 400, 700)  # Increased height for more buttons
        
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel('Select Histogram to Generate:')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Create buttons for each histogram type
        self.create_button(layout, "Shortest Path Length Distribution", self.shortest_path_histogram)
        self.create_button(layout, "Degree Centrality", self.degree_centrality_histogram)
        self.create_button(layout, "Betweenness Centrality", self.betweenness_centrality_histogram)
        self.create_button(layout, "Closeness Centrality", self.closeness_centrality_histogram)
        self.create_button(layout, "Eigenvector Centrality", self.eigenvector_centrality_histogram)
        self.create_button(layout, "Clustering Coefficient", self.clustering_coefficient_histogram)
        self.create_button(layout, "Degree Distribution", self.degree_distribution_histogram)
        self.create_button(layout, "Node Connectivity", self.node_connectivity_histogram)
        self.create_button(layout, "Eccentricity", self.eccentricity_histogram)
        self.create_button(layout, "K-Core Decomposition", self.kcore_histogram)
        self.create_button(layout, "Triangle Count", self.triangle_count_histogram)
        self.create_button(layout, "Load Centrality", self.load_centrality_histogram)
        self.create_button(layout, "Communicability Betweenness Centrality", self.communicability_centrality_histogram)
        self.create_button(layout, "Harmonic Centrality", self.harmonic_centrality_histogram)
        self.create_button(layout, "Current Flow Betweenness", self.current_flow_betweenness_histogram)
        self.create_button(layout, "Dispersion", self.dispersion_histogram)
        self.create_button(layout, "Network Bridges", self.bridges_analysis)
        
        # Compute All button - visually distinct
        compute_all_button = QPushButton('Compute All Analyses and Export to CSV')
        compute_all_button.clicked.connect(self.compute_all)
        compute_all_button.setMinimumHeight(50)
        compute_all_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: 3px solid #F57C00;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FB8C00;
                border-color: #E65100;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        layout.addWidget(compute_all_button)
        
        # Close button
        close_button = QPushButton('Close')
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        #layout.addWidget(close_button)
        
        self.setLayout(layout)

    def create_button(self, layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setMinimumHeight(40)
        button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        layout.addWidget(button)

    def compute_all(self):
        """Compute all available analyses and export to CSV files and histogram images"""
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        import os
        import pandas as pd
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self, 
            'Compute All Analyses',
            'This will compute all available analyses and may take a while for large networks.\n\n'
            'Do you want to continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Get save location and folder name
        folder_path, _ = QFileDialog.getSaveFileName(
            self,
            'Select Location and Name for Output Folder',
            'network_analysis_results',
            'Folder (*)'
        )
        
        if not folder_path:
            return
        
        # Create main directory and subdirectories
        try:
            os.makedirs(folder_path, exist_ok=True)
            csvs_path = os.path.join(folder_path, 'csvs')
            graphs_path = os.path.join(folder_path, 'graph_images')
            os.makedirs(csvs_path, exist_ok=True)
            os.makedirs(graphs_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not create directory: {str(e)}')
            return
        
        print(f"Computing all analyses and saving to: {folder_path}")
        
        try:
            # 1. Shortest Path Length Distribution
            print("Computing shortest path distribution...")
            components = list(nx.connected_components(self.G))
            if len(components) > 1:
                all_path_lengths = []
                max_diameter = 0
                for component in components:
                    subgraph = self.G.subgraph(component)
                    if len(component) < 2:
                        continue
                    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                    component_diameter = max(nx.eccentricity(subgraph, sp=shortest_path_lengths).values())
                    max_diameter = max(max_diameter, component_diameter)
                    for pls in shortest_path_lengths.values():
                        all_path_lengths.extend(list(pls.values()))
                all_path_lengths = [pl for pl in all_path_lengths if pl > 0]
                if all_path_lengths:
                    path_lengths = np.zeros(max_diameter + 1, dtype=int)
                    pl, cnts = np.unique(all_path_lengths, return_counts=True)
                    path_lengths[pl] += cnts
                    freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
                    df = pd.DataFrame({
                        'Path_Length': np.arange(1, max_diameter + 1),
                        'Frequency_Percent': freq_percent
                    })
                    df.to_csv(os.path.join(csvs_path, 'shortest_path_distribution.csv'), index=False)
                    
                    # Generate and save plot
                    fig, ax = plt.subplots(figsize=(15, 8))
                    ax.bar(np.arange(1, max_diameter + 1), height=freq_percent)
                    ax.set_title(f"Distribution of shortest path length in G (across {len(components)} components)", 
                                fontdict={"size": 35}, loc="center")
                    ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
                    ax.set_ylabel("Frequency (%)", fontdict={"size": 22})
                    plt.tight_layout()
                    plt.savefig(os.path.join(graphs_path, 'shortest_path_distribution.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
            else:
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(self.G))
                diameter = max(nx.eccentricity(self.G, sp=shortest_path_lengths).values())
                path_lengths = np.zeros(diameter + 1, dtype=int)
                for pls in shortest_path_lengths.values():
                    pl, cnts = np.unique(list(pls.values()), return_counts=True)
                    path_lengths[pl] += cnts
                freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
                df = pd.DataFrame({
                    'Path_Length': np.arange(1, diameter + 1),
                    'Frequency_Percent': freq_percent
                })
                df.to_csv(os.path.join(csvs_path, 'shortest_path_distribution.csv'), index=False)
                
                # Generate and save plot
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.bar(np.arange(1, diameter + 1), height=freq_percent)
                ax.set_title("Distribution of shortest path length in G", fontdict={"size": 35}, loc="center")
                ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
                ax.set_ylabel("Frequency (%)", fontdict={"size": 22})
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_path, 'shortest_path_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # 2. Degree Centrality
            print("Computing degree centrality...")
            degree_centrality = nx.centrality.degree_centrality(self.G)
            df = pd.DataFrame(list(degree_centrality.items()), columns=['Node', 'Degree_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'degree_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(degree_centrality.values(), bins=25)
            plt.xticks(ticks=[0, 0.025, 0.05, 0.1, 0.15, 0.2])
            plt.title("Degree Centrality Histogram", fontdict={"size": 35}, loc="center")
            plt.xlabel("Degree Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'degree_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 3. Betweenness Centrality
            print("Computing betweenness centrality...")
            components = list(nx.connected_components(self.G))
            if len(components) > 1:
                combined_betweenness_centrality = {}
                for component in components:
                    if len(component) < 2:
                        for node in component:
                            combined_betweenness_centrality[node] = 0.0
                        continue
                    subgraph = self.G.subgraph(component)
                    component_betweenness = nx.centrality.betweenness_centrality(subgraph)
                    combined_betweenness_centrality.update(component_betweenness)
                betweenness_centrality = combined_betweenness_centrality
                title_suffix = f" (across {len(components)} components)"
            else:
                betweenness_centrality = nx.centrality.betweenness_centrality(self.G)
                title_suffix = ""
            df = pd.DataFrame(list(betweenness_centrality.items()), columns=['Node', 'Betweenness_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'betweenness_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(betweenness_centrality.values(), bins=100)
            plt.xticks(ticks=[0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5])
            plt.title(f"Betweenness Centrality Histogram{title_suffix}", fontdict={"size": 35}, loc="center")
            plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'betweenness_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 4. Closeness Centrality
            print("Computing closeness centrality...")
            closeness_centrality = nx.centrality.closeness_centrality(self.G)
            df = pd.DataFrame(list(closeness_centrality.items()), columns=['Node', 'Closeness_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'closeness_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(closeness_centrality.values(), bins=60)
            plt.title("Closeness Centrality Histogram", fontdict={"size": 35}, loc="center")
            plt.xlabel("Closeness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'closeness_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 5. Eigenvector Centrality
            print("Computing eigenvector centrality...")
            try:
                eigenvector_centrality = nx.centrality.eigenvector_centrality(self.G_unweighted)
                df = pd.DataFrame(list(eigenvector_centrality.items()), columns=['Node', 'Eigenvector_Centrality'])
                df.to_csv(os.path.join(csvs_path, 'eigenvector_centrality.csv'), index=False)
                
                # Generate and save plot
                fig = plt.figure(figsize=(15, 8))
                plt.hist(eigenvector_centrality.values(), bins=60)
                plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])
                plt.title("Eigenvector Centrality Histogram", fontdict={"size": 35}, loc="center")
                plt.xlabel("Eigenvector Centrality", fontdict={"size": 20})
                plt.ylabel("Counts", fontdict={"size": 20})
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_path, 'eigenvector_centrality.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Could not compute eigenvector centrality: {e}")
            
            # 6. Clustering Coefficient
            print("Computing clustering coefficient...")
            clusters = nx.clustering(self.G_unweighted)
            df = pd.DataFrame(list(clusters.items()), columns=['Node', 'Clustering_Coefficient'])
            df.to_csv(os.path.join(csvs_path, 'clustering_coefficient.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(clusters.values(), bins=50)
            plt.title("Clustering Coefficient Histogram", fontdict={"size": 35}, loc="center")
            plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'clustering_coefficient.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 7. Degree Distribution
            print("Computing degree distribution...")
            degrees = {node: deg for node, deg in self.G.degree()}
            df = pd.DataFrame(list(degrees.items()), columns=['Node', 'Degree'])
            df.to_csv(os.path.join(csvs_path, 'degree_distribution.csv'), index=False)
            
            # Generate and save plot
            degree_values = [deg for node, deg in self.G.degree()]
            fig = plt.figure(figsize=(15, 8))
            plt.hist(degree_values, bins=max(30, int(np.sqrt(len(degree_values)))), alpha=0.7)
            plt.title("Degree Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Degree", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'degree_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 8. Node Connectivity
            print("Computing node connectivity...")
            connectivity = {}
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if len(neighbors) > 1:
                    connectivity[node] = nx.node_connectivity(self.G, neighbors[0], neighbors[1])
                else:
                    connectivity[node] = 0
            df = pd.DataFrame(list(connectivity.items()), columns=['Node', 'Node_Connectivity'])
            df.to_csv(os.path.join(csvs_path, 'node_connectivity.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(connectivity.values(), bins=20, alpha=0.7)
            plt.title("Node Connectivity Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Node Connectivity", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'node_connectivity.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 9. Eccentricity
            print("Computing eccentricity...")
            if not nx.is_connected(self.G):
                largest_cc = max(nx.connected_components(self.G), key=len)
                G_cc = self.G.subgraph(largest_cc)
                eccentricity = nx.eccentricity(G_cc)
            else:
                eccentricity = nx.eccentricity(self.G)
            df = pd.DataFrame(list(eccentricity.items()), columns=['Node', 'Eccentricity'])
            df.to_csv(os.path.join(csvs_path, 'eccentricity.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(eccentricity.values(), bins=20, alpha=0.7)
            plt.title("Eccentricity Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Eccentricity", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'eccentricity.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 10. K-Core
            print("Computing k-core decomposition...")
            kcore = nx.core_number(self.G_unweighted)
            df = pd.DataFrame(list(kcore.items()), columns=['Node', 'K_Core'])
            df.to_csv(os.path.join(csvs_path, 'kcore.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(kcore.values(), bins=max(5, max(kcore.values())), alpha=0.7)
            plt.title("K-Core Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("K-Core Number", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'kcore.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 11. Triangle Count
            print("Computing triangle count...")
            triangles = nx.triangles(self.G)
            df = pd.DataFrame(list(triangles.items()), columns=['Node', 'Triangle_Count'])
            df.to_csv(os.path.join(csvs_path, 'triangle_count.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(triangles.values(), bins=30, alpha=0.7)
            plt.title("Triangle Count Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Number of Triangles", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'triangle_count.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 12. Load Centrality
            print("Computing load centrality...")
            load_centrality = nx.load_centrality(self.G)
            df = pd.DataFrame(list(load_centrality.items()), columns=['Node', 'Load_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'load_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(load_centrality.values(), bins=50, alpha=0.7)
            plt.title("Load Centrality Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Load Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'load_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 13. Communicability Betweenness Centrality
            print("Computing communicability betweenness centrality...")
            components = list(nx.connected_components(self.G_unweighted))
            if len(components) > 1:
                combined_comm_centrality = {}
                for component in components:
                    if len(component) < 2:
                        for node in component:
                            combined_comm_centrality[node] = 0.0
                        continue
                    subgraph = self.G_unweighted.subgraph(component)
                    try:
                        component_comm_centrality = nx.communicability_betweenness_centrality(subgraph)
                        combined_comm_centrality.update(component_comm_centrality)
                    except Exception as comp_e:
                        print(f"Error computing communicability centrality for component: {comp_e}")
                        for node in component:
                            combined_comm_centrality[node] = 0.0
                comm_centrality = combined_comm_centrality
                title_suffix = f" (across {len(components)} components)"
            else:
                comm_centrality = nx.communicability_betweenness_centrality(self.G_unweighted)
                title_suffix = ""
            df = pd.DataFrame(list(comm_centrality.items()), columns=['Node', 'Communicability_Betweenness_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'communicability_betweenness_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(comm_centrality.values(), bins=50, alpha=0.7)
            plt.title(f"Communicability Betweenness Centrality Distribution{title_suffix}", 
                     fontdict={"size": 35}, loc="center")
            plt.xlabel("Communicability Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'communicability_betweenness_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 14. Harmonic Centrality
            print("Computing harmonic centrality...")
            harmonic_centrality = nx.harmonic_centrality(self.G)
            df = pd.DataFrame(list(harmonic_centrality.items()), columns=['Node', 'Harmonic_Centrality'])
            df.to_csv(os.path.join(csvs_path, 'harmonic_centrality.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(harmonic_centrality.values(), bins=50, alpha=0.7)
            plt.title("Harmonic Centrality Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Harmonic Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'harmonic_centrality.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 15. Current Flow Betweenness
            print("Computing current flow betweenness...")
            components = list(nx.connected_components(self.G))
            if len(components) > 1:
                combined_current_flow = {}
                for component in components:
                    if len(component) < 2:
                        for node in component:
                            combined_current_flow[node] = 0.0
                        continue
                    subgraph = self.G.subgraph(component)
                    try:
                        component_current_flow = nx.current_flow_betweenness_centrality(subgraph)
                        combined_current_flow.update(component_current_flow)
                    except Exception as comp_e:
                        print(f"Error computing current flow betweenness for component: {comp_e}")
                        for node in component:
                            combined_current_flow[node] = 0.0
                current_flow = combined_current_flow
                title_suffix = f" (across {len(components)} components)"
            else:
                current_flow = nx.current_flow_betweenness_centrality(self.G)
                title_suffix = ""
            df = pd.DataFrame(list(current_flow.items()), columns=['Node', 'Current_Flow_Betweenness'])
            df.to_csv(os.path.join(csvs_path, 'current_flow_betweenness.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(current_flow.values(), bins=50, alpha=0.7)
            plt.title(f"Current Flow Betweenness Centrality Distribution{title_suffix}", 
                     fontdict={"size": 35}, loc="center")
            plt.xlabel("Current Flow Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'current_flow_betweenness.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 16. Dispersion
            print("Computing dispersion...")
            dispersion_values = {}
            nodes = list(self.G.nodes())
            for u in nodes:
                if self.G.degree(u) < 2:
                    dispersion_values[u] = 0
                    continue
                neighbors = list(self.G.neighbors(u))
                if len(neighbors) < 2:
                    dispersion_values[u] = 0
                    continue
                disp_scores = []
                for v in neighbors:
                    try:
                        disp_score = nx.dispersion(self.G, u, v)
                        disp_scores.append(disp_score)
                    except:
                        continue
                dispersion_values[u] = sum(disp_scores) / len(disp_scores) if disp_scores else 0
            df = pd.DataFrame(list(dispersion_values.items()), columns=['Node', 'Average_Dispersion'])
            df.to_csv(os.path.join(csvs_path, 'dispersion.csv'), index=False)
            
            # Generate and save plot
            fig = plt.figure(figsize=(15, 8))
            plt.hist(dispersion_values.values(), bins=30, alpha=0.7)
            plt.title("Average Dispersion Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Average Dispersion", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'dispersion.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 17. Bridges (CSV only, no plot)
            print("Computing bridges...")
            bridges = list(nx.bridges(self.G))
            try:
                # Get the existing DataFrame from the model
                original_df = self.network_analysis.network_table.model()._data
                
                # Create boolean mask
                mask = pd.Series([False] * len(original_df))
                
                for u, v in bridges:
                    # Check for both (u,v) and (v,u) orientations
                    bridge_mask = (
                        ((original_df.iloc[:, 0] == u) & (original_df.iloc[:, 1] == v)) |
                        ((original_df.iloc[:, 0] == v) & (original_df.iloc[:, 1] == u))
                    )
                    mask |= bridge_mask
                # Filter the DataFrame to only include bridge connections
                df = original_df[mask].copy()
            except:
                df = pd.DataFrame(bridges, columns=['Node_A', 'Node_B'])

            df.to_csv(os.path.join(csvs_path, 'bridges.csv'), index=False)
            
            print(f"\nAll analyses complete! Results saved to: {folder_path}")
            QMessageBox.information(
                self, 
                'Complete', 
                f'All analyses have been computed and saved to:\n\n{folder_path}\n\n'
                f'CSVs: {csvs_path}\nGraphs: {graphs_path}\n\n'
                f'Total files created: 17 CSVs + 16 histogram images'
            )
            
        except Exception as e:
            print(f"Error during compute all: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', f'An error occurred during computation:\n\n{str(e)}')

    def shortest_path_histogram(self):
        try:
            # Check if graph has multiple disconnected components
            components = list(nx.connected_components(self.G))
            
            if len(components) > 1:
                print(f"Warning: Graph has {len(components)} disconnected components. Computing shortest paths within each component separately.")
                
                # Initialize variables to collect data from all components
                all_path_lengths = []
                max_diameter = 0
                
                # Process each component separately
                for i, component in enumerate(components):
                    subgraph = self.G.subgraph(component)
                    
                    if len(component) < 2:
                        # Skip single-node components (no paths to compute)
                        continue
                    
                    # Compute shortest paths for this component
                    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                    component_diameter = max(nx.eccentricity(subgraph, sp=shortest_path_lengths).values())
                    max_diameter = max(max_diameter, component_diameter)
                    
                    # Collect path lengths from this component
                    for pls in shortest_path_lengths.values():
                        all_path_lengths.extend(list(pls.values()))
                
                # Remove self-paths (length 0) and create histogram
                all_path_lengths = [pl for pl in all_path_lengths if pl > 0]
                
                if not all_path_lengths:
                    print("No paths found across components (only single-node components)")
                    return
                    
                # Create combined histogram
                path_lengths = np.zeros(max_diameter + 1, dtype=int)
                pl, cnts = np.unique(all_path_lengths, return_counts=True)
                path_lengths[pl] += cnts
                
                title_suffix = f" (across {len(components)} components)"
                
            else:
                # Single component
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(self.G))
                diameter = max(nx.eccentricity(self.G, sp=shortest_path_lengths).values())
                path_lengths = np.zeros(diameter + 1, dtype=int)
                for pls in shortest_path_lengths.values():
                    pl, cnts = np.unique(list(pls.values()), return_counts=True)
                    path_lengths[pl] += cnts
                max_diameter = diameter
                title_suffix = ""
            
            # Generate visualization and results (same for both cases)
            freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.bar(np.arange(1, max_diameter + 1), height=freq_percent)
            ax.set_title(
                f"Distribution of shortest path length in G{title_suffix}", 
                fontdict={"size": 35}, loc="center"
            )
            ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
            ax.set_ylabel("Frequency (%)", fontdict={"size": 22})
            plt.show()
            
            freq_dict = {freq: length for length, freq in enumerate(freq_percent, start=1)}
            self.network_analysis.format_for_upperright_table(
                freq_dict, 
                metric='Frequency (%)', 
                value='Shortest Path Length', 
                title=f"Distribution of shortest path length in G{title_suffix}"
            )
            
        except Exception as e:
            print(f"Error generating shortest path histogram: {e}")
    
    def degree_centrality_histogram(self):
        try:
            degree_centrality = nx.centrality.degree_centrality(self.G)
            plt.figure(figsize=(15, 8))
            plt.hist(degree_centrality.values(), bins=25)
            plt.xticks(ticks=[0, 0.025, 0.05, 0.1, 0.15, 0.2])
            plt.title("Degree Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Degree Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Degree Centrality'] = degree_centrality
            self.network_analysis.format_for_upperright_table(degree_centrality, metric='Node', 
                                                            value='Degree Centrality', 
                                                            title="Degree Centrality Table")
        except Exception as e:
            print(f"Error generating degree centrality histogram: {e}")

    def betweenness_centrality_histogram(self):
        try:
            # Check if graph has multiple disconnected components
            components = list(nx.connected_components(self.G))
            
            if len(components) > 1:
                print(f"Warning: Graph has {len(components)} disconnected components. Computing betweenness centrality within each component separately.")
                
                # Initialize dictionary to collect betweenness centrality from all components
                combined_betweenness_centrality = {}
                
                # Process each component separately
                for i, component in enumerate(components):
                    if len(component) < 2:
                        # For single-node components, betweenness centrality is 0
                        for node in component:
                            combined_betweenness_centrality[node] = 0.0
                        continue
                    
                    # Create subgraph for this component
                    subgraph = self.G.subgraph(component)
                    
                    # Compute betweenness centrality for this component
                    component_betweenness = nx.centrality.betweenness_centrality(subgraph)
                    
                    # Add to combined results
                    combined_betweenness_centrality.update(component_betweenness)
                
                betweenness_centrality = combined_betweenness_centrality
                title_suffix = f" (across {len(components)} components)"
                
            else:
                # Single component
                betweenness_centrality = nx.centrality.betweenness_centrality(self.G)
                title_suffix = ""
            
            # Generate visualization and results (same for both cases)
            plt.figure(figsize=(15, 8))
            plt.hist(betweenness_centrality.values(), bins=100)
            plt.xticks(ticks=[0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5])
            plt.title(
                f"Betweenness Centrality Histogram{title_suffix}", 
                fontdict={"size": 35}, loc="center"
            )
            plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Betweenness Centrality'] = betweenness_centrality
            
            self.network_analysis.format_for_upperright_table(
                betweenness_centrality, 
                metric='Node', 
                value='Betweenness Centrality', 
                title=f"Betweenness Centrality Table{title_suffix}"
            )
            
        except Exception as e:
            print(f"Error generating betweenness centrality histogram: {e}")
    
    def closeness_centrality_histogram(self):
        try:
            closeness_centrality = nx.centrality.closeness_centrality(self.G)
            plt.figure(figsize=(15, 8))
            plt.hist(closeness_centrality.values(), bins=60)
            plt.title("Closeness Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Closeness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Closeness Centrality'] = closeness_centrality
            self.network_analysis.format_for_upperright_table(closeness_centrality, metric='Node', 
                                                            value='Closeness Centrality', 
                                                            title="Closeness Centrality Table")
        except Exception as e:
            print(f"Error generating closeness centrality histogram: {e}")
    
    def eigenvector_centrality_histogram(self):
        try:
            eigenvector_centrality = nx.centrality.eigenvector_centrality(self.G_unweighted)
            plt.figure(figsize=(15, 8))
            plt.hist(eigenvector_centrality.values(), bins=60)
            plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])
            plt.title("Eigenvector Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Eigenvector Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Eigenvector Centrality'] = eigenvector_centrality
            self.network_analysis.format_for_upperright_table(eigenvector_centrality, metric='Node', 
                                                            value='Eigenvector Centrality', 
                                                            title="Eigenvector Centrality Table")
        except Exception as e:
            print(f"Error generating eigenvector centrality histogram: {e}")
    
    def clustering_coefficient_histogram(self):
        try:
            clusters = nx.clustering(self.G_unweighted)
            plt.figure(figsize=(15, 8))
            plt.hist(clusters.values(), bins=50)
            plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Clustering Coefficient'] = clusters
            self.network_analysis.format_for_upperright_table(clusters, metric='Node', 
                                                            value='Clustering Coefficient', 
                                                            title="Clustering Coefficient Table")
        except Exception as e:
            print(f"Error generating clustering coefficient histogram: {e}")
    
    def bridges_analysis(self):
        try:
            bridges = list(nx.bridges(self.G))
            try:
                # Get the existing DataFrame from the model
                original_df = self.network_analysis.network_table.model()._data
                
                # Create boolean mask
                mask = pd.Series([False] * len(original_df))
                
                for u, v in bridges:
                    # Check for both (u,v) and (v,u) orientations
                    bridge_mask = (
                        ((original_df.iloc[:, 0] == u) & (original_df.iloc[:, 1] == v)) |
                        ((original_df.iloc[:, 0] == v) & (original_df.iloc[:, 1] == u))
                    )
                    mask |= bridge_mask
                # Filter the DataFrame to only include bridge connections
                filtered_df = original_df[mask].copy()
                df_dict = {i: row.tolist() for i, row in enumerate(filtered_df.values)}
                self.network_analysis.format_for_upperright_table(df_dict, metric='Bridge ID', value = ['NodeA', 'NodeB', 'EdgeC'],
                                                            title="Bridges")
            except:
                self.network_analysis.format_for_upperright_table(bridges, metric='Node Pair', 
                                                            title="Bridges")
        except Exception as e:
            print(f"Error generating bridges analysis: {e}")
    
    def degree_distribution_histogram(self):
        """Raw degree distribution - very useful for understanding network topology"""
        try:
            degrees = [self.G.degree(n) for n in self.G.nodes()]
            plt.figure(figsize=(15, 8))
            plt.hist(degrees, bins=max(30, int(np.sqrt(len(degrees)))), alpha=0.7)
            plt.title("Degree Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Degree", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.yscale('log')  # Often useful for degree distributions
            plt.show()
            
            degree_dict = {node: deg for node, deg in self.G.degree()}
            self.network_analysis.format_for_upperright_table(degree_dict, metric='Node', 
                                                            value='Degree', title="Degree Distribution Table")
        except Exception as e:
            print(f"Error generating degree distribution histogram: {e}")
    

    def node_connectivity_histogram(self):
        """Local node connectivity - minimum number of nodes that must be removed to disconnect neighbors"""
        try:
            if self.G.number_of_nodes() > 500:
                print("Note this analysis may be slow for large network (>500 nodes)")
                #return
                
            connectivity = {}
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                if len(neighbors) > 1:
                    connectivity[node] = nx.node_connectivity(self.G, neighbors[0], neighbors[1])
                else:
                    connectivity[node] = 0
            
            plt.figure(figsize=(15, 8))
            plt.hist(connectivity.values(), bins=20, alpha=0.7)
            plt.title("Node Connectivity Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Node Connectivity", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Node Connectivity'] = connectivity
            self.network_analysis.format_for_upperright_table(connectivity, metric='Node', 
                                                            value='Connectivity', title="Node Connectivity Table")
        except Exception as e:
            print(f"Error generating node connectivity histogram: {e}")
    
    def eccentricity_histogram(self):
        """Eccentricity - maximum distance from a node to any other node"""
        try:
            if not nx.is_connected(self.G):
                print("Graph is not connected. Using largest connected component.")
                largest_cc = max(nx.connected_components(self.G), key=len)
                G_cc = self.G.subgraph(largest_cc)
                eccentricity = nx.eccentricity(G_cc)
            else:
                eccentricity = nx.eccentricity(self.G)
            
            plt.figure(figsize=(15, 8))
            plt.hist(eccentricity.values(), bins=20, alpha=0.7)
            plt.title("Eccentricity Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Eccentricity", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Eccentricity'] = eccentricity
            self.network_analysis.format_for_upperright_table(eccentricity, metric='Node', 
                                                            value='Eccentricity', title="Eccentricity Table")
        except Exception as e:
            print(f"Error generating eccentricity histogram: {e}")
    
    def kcore_histogram(self):
        """K-core decomposition - identifies cohesive subgroups"""
        try:
            kcore = nx.core_number(self.G_unweighted)
            plt.figure(figsize=(15, 8))
            plt.hist(kcore.values(), bins=max(5, max(kcore.values())), alpha=0.7)
            plt.title("K-Core Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("K-Core Number", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['K-Core'] = kcore
            self.network_analysis.format_for_upperright_table(kcore, metric='Node', 
                                                            value='K-Core', title="K-Core Table")
        except Exception as e:
            print(f"Error generating k-core histogram: {e}")
    
    def triangle_count_histogram(self):
        """Number of triangles each node participates in"""
        try:
            triangles = nx.triangles(self.G)
            plt.figure(figsize=(15, 8))
            plt.hist(triangles.values(), bins=30, alpha=0.7)
            plt.title("Triangle Count Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Number of Triangles", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Triangle Count'] = triangles
            self.network_analysis.format_for_upperright_table(triangles, metric='Node', 
                                                            value='Triangle Count', title="Triangle Count Table")
        except Exception as e:
            print(f"Error generating triangle count histogram: {e}")
    
    def load_centrality_histogram(self):
        """Load centrality - fraction of shortest paths passing through each node"""
        try:
            if self.G.number_of_nodes() > 1000:
                print("Note this analysis may be slow for large network (>1000 nodes)")
                #return
                
            load_centrality = nx.load_centrality(self.G)
            plt.figure(figsize=(15, 8))
            plt.hist(load_centrality.values(), bins=50, alpha=0.7)
            plt.title("Load Centrality Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Load Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Load Centrality'] = load_centrality
            self.network_analysis.format_for_upperright_table(load_centrality, metric='Node', 
                                                            value='Load Centrality', title="Load Centrality Table")
        except Exception as e:
            print(f"Error generating load centrality histogram: {e}")
    
    def communicability_centrality_histogram(self):
        """Communicability centrality - based on communicability between nodes"""
        try:
            if self.G.number_of_nodes() > 500:
                print("Note this analysis may be slow for large network (>500 nodes)")
                #return
            
            # Check if graph has multiple disconnected components
            components = list(nx.connected_components(self.G_unweighted))
            
            if len(components) > 1:
                print(f"Warning: Graph has {len(components)} disconnected components. Computing communicability centrality within each component separately.")
                
                # Initialize dictionary to collect communicability centrality from all components
                combined_comm_centrality = {}
                
                # Process each component separately
                for i, component in enumerate(components):
                    if len(component) < 2:
                        # For single-node components, communicability betweenness centrality is 0
                        for node in component:
                            combined_comm_centrality[node] = 0.0
                        continue
                    
                    # Create subgraph for this component
                    subgraph = self.G_unweighted.subgraph(component)
                    
                    # Compute communicability betweenness centrality for this component
                    try:
                        component_comm_centrality = nx.communicability_betweenness_centrality(subgraph)
                        # Add to combined results
                        combined_comm_centrality.update(component_comm_centrality)
                    except Exception as comp_e:
                        print(f"Error computing communicability centrality for component {i+1}: {comp_e}")
                        # Set centrality to 0 for nodes in this component if computation fails
                        for node in component:
                            combined_comm_centrality[node] = 0.0
                
                comm_centrality = combined_comm_centrality
                title_suffix = f" (across {len(components)} components)"
                
            else:
                # Single component
                comm_centrality = nx.communicability_betweenness_centrality(self.G_unweighted)
                title_suffix = ""
            
            # Generate visualization and results (same for both cases)
            plt.figure(figsize=(15, 8))
            plt.hist(comm_centrality.values(), bins=50, alpha=0.7)
            plt.title(
                f"Communicability Betweenness Centrality Distribution{title_suffix}", 
                fontdict={"size": 35}, loc="center"
            )
            plt.xlabel("Communicability Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            self.stats_dict['Communicability Betweenness Centrality'] = comm_centrality
            plt.show()
            
            self.network_analysis.format_for_upperright_table(
                comm_centrality, 
                metric='Node', 
                value='Communicability Betweenness Centrality', 
                title=f"Communicability Betweenness Centrality Table{title_suffix}"
            )
            
        except Exception as e:
            print(f"Error generating communicability betweenness centrality histogram: {e}")
    
    def harmonic_centrality_histogram(self):
        """Harmonic centrality - better than closeness for disconnected networks"""
        try:
            harmonic_centrality = nx.harmonic_centrality(self.G)
            plt.figure(figsize=(15, 8))
            plt.hist(harmonic_centrality.values(), bins=50, alpha=0.7)
            plt.title("Harmonic Centrality Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Harmonic Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Harmonic Centrality Distribution'] = harmonic_centrality
            self.network_analysis.format_for_upperright_table(harmonic_centrality, metric='Node', 
                                                            value='Harmonic Centrality', 
                                                            title="Harmonic Centrality Table")
        except Exception as e:
            print(f"Error generating harmonic centrality histogram: {e}")
    
    def current_flow_betweenness_histogram(self):
        """Current flow betweenness - models network as electrical circuit"""
        try:
            if self.G.number_of_nodes() > 500:  
                print("Note this analysis may be slow for large network (>500 nodes)")
                #return
            
            # Check if graph has multiple disconnected components
            components = list(nx.connected_components(self.G))
            
            if len(components) > 1:
                print(f"Warning: Graph has {len(components)} disconnected components. Computing current flow betweenness centrality within each component separately.")
                
                # Initialize dictionary to collect current flow betweenness from all components
                combined_current_flow = {}
                
                # Process each component separately
                for i, component in enumerate(components):
                    if len(component) < 2:
                        # For single-node components, current flow betweenness centrality is 0
                        for node in component:
                            combined_current_flow[node] = 0.0
                        continue
                    
                    # Create subgraph for this component
                    subgraph = self.G.subgraph(component)
                    
                    # Compute current flow betweenness centrality for this component
                    try:
                        component_current_flow = nx.current_flow_betweenness_centrality(subgraph)
                        # Add to combined results
                        combined_current_flow.update(component_current_flow)
                    except Exception as comp_e:
                        print(f"Error computing current flow betweenness for component {i+1}: {comp_e}")
                        # Set centrality to 0 for nodes in this component if computation fails
                        for node in component:
                            combined_current_flow[node] = 0.0
                
                current_flow = combined_current_flow
                title_suffix = f" (across {len(components)} components)"
                
            else:
                # Single component
                current_flow = nx.current_flow_betweenness_centrality(self.G)
                title_suffix = ""
            
            # Generate visualization and results (same for both cases)
            plt.figure(figsize=(15, 8))
            plt.hist(current_flow.values(), bins=50, alpha=0.7)
            plt.title(
                f"Current Flow Betweenness Centrality Distribution{title_suffix}", 
                fontdict={"size": 35}, loc="center"
            )
            plt.xlabel("Current Flow Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Current Flow Betweenness Centrality'] = current_flow            
            self.network_analysis.format_for_upperright_table(
                current_flow, 
                metric='Node', 
                value='Current Flow Betweenness', 
                title=f"Current Flow Betweenness Table{title_suffix}"
            )
            
        except Exception as e:
            print(f"Error generating current flow betweenness histogram: {e}")
    
    def dispersion_histogram(self):
        """Dispersion - measures how scattered a node's neighbors are"""
        try:
            if self.G.number_of_nodes() > 300:  # Skip for large networks (very computationally expensive)
                print("Note this analysis may be slow for large network (>300 nodes)")
                #return
                
            # Calculate average dispersion for each node
            dispersion_values = {}
            nodes = list(self.G.nodes())
            
            for u in nodes:
                if self.G.degree(u) < 2:  # Need at least 2 neighbors for dispersion
                    dispersion_values[u] = 0
                    continue
                    
                # Calculate dispersion for node u with all its neighbors
                neighbors = list(self.G.neighbors(u))
                if len(neighbors) < 2:
                    dispersion_values[u] = 0
                    continue
                
                # Get dispersion scores for this node with all neighbors
                disp_scores = []
                for v in neighbors:
                    try:
                        disp_score = nx.dispersion(self.G, u, v)
                        disp_scores.append(disp_score)
                    except:
                        continue
                
                # Average dispersion for this node
                dispersion_values[u] = sum(disp_scores) / len(disp_scores) if disp_scores else 0
            
            plt.figure(figsize=(15, 8))
            plt.hist(dispersion_values.values(), bins=30, alpha=0.7)
            plt.title("Average Dispersion Distribution", fontdict={"size": 35}, loc="center")
            plt.xlabel("Average Dispersion", fontdict={"size": 20})
            plt.ylabel("Frequency", fontdict={"size": 20})
            plt.show()
            self.stats_dict['Dispersion'] = dispersion_values            
            self.network_analysis.format_for_upperright_table(dispersion_values, metric='Node', 
                                                            value='Average Dispersion', 
                                                            title="Average Dispersion Table")
        except Exception as e:
            print(f"Error generating dispersion histogram: {e}")