import numpy as np
import networkx as nx
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMenu,
                              QSizePolicy, QApplication, QScrollArea, QLabel, QFrame,
                              QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer, QPointF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem, PlotCurveItem, GraphicsLayoutWidget, ROI
import colorsys
import random
import copy


def remove_dupe_ids(idens):

    try:
        ret_dict = {}

        for node, iden in idens.items():
            try:
                ret_dict[node] = [random.choice(iden)]
            except:
                pass

        return ret_dict
    except:
        return None

class GraphLoadThread(QThread):
    """Thread for loading graph layouts without blocking the UI"""
    finished = pyqtSignal(object)  # Emits the computed layout data
    
    def __init__(self, graph, geometric, component, centroids, communities, 
                 community_dict, identities, identity_dict, weight, z_size,
                 shell, node_size, edge_size):
        super().__init__()
        self.graph = graph
        self.geometric = geometric
        self.component = component
        self.centroids = centroids
        self.communities = communities
        self.community_dict = community_dict
        self.identities = identities
        self.identity_dict = identity_dict
        self.weight = weight
        self.z_size = z_size
        self.shell = shell
        self.node_size = node_size
        self.edge_size = edge_size
    
    def run(self):
        """Compute layout and colors in background thread"""
        result = {}
        
        # Compute node positions
        if not self.geometric and not self.component:
            result['pos'] = self._compute_fast_spring_layout()
        elif self.geometric:
            result['pos'] = self._compute_geometric_layout()
        elif self.component:
            nodes = list(self.graph.nodes())
            n = len(nodes)
            result['pos'] = self._spring_layout_numpy_super(nodes, n)

        # Compute node colors and sizes
        result['colors'], result['sizes'] = self._compute_node_attributes()
        
        # Compute edge data
        result['edges'] = self._compute_edge_data(result['pos'])

        # Prepare node spots for rendering (with pre-computed brushes)
        result['node_spots'], result['brush_cache'] = self._prepare_node_spots(result['pos'], result['colors'], result['sizes'])
        
        # Prepare label data
        result['label_data'] = self._prepare_label_data(result['pos'])
        
        # Prepare edge items
        result['edge_pens'] = self._prepare_edge_pens(result['edges'])
        
        self.finished.emit(result)
    
    def _compute_fast_spring_layout(self):
        """Fast vectorized spring layout using numpy"""
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # For small graphs, use networkx (overhead is negligible)
        if n < 500 and not self.shell:
            return nx.spring_layout(self.graph, seed=42, iterations=50)

        # Use fast vectorized implementation for larger graphs
        try:
            if not self.shell:
                return self._spring_layout_numpy(nodes, n)
            else:
                return self._shell_layout_numpy_super(nodes, n)
        except Exception as e:
            pass

    def _shell_layout_numpy_super(self, nodes, n):
        """
        Shell layout with physically separated connected components
        """        
        np.random.seed(42)
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        if len(components) == 1:
            # Single component - compute shell layout directly with numpy
            comp_nodes = nodes
            
            if n == 1:
                return {nodes[0]: np.array([0.0, 0.0])}
            
            # Create node to index mapping
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Compute degree centrality using numpy
            degrees = np.zeros(n)
            for u, v in self.graph.edges():
                if u in node_to_idx and v in node_to_idx:
                    degrees[node_to_idx[u]] += 1
                    degrees[node_to_idx[v]] += 1
            
            # Find most central node (highest degree)
            central_idx = np.argmax(degrees)
            central_node = nodes[central_idx]
            
            # Build adjacency list for BFS
            adj_list = {node: [] for node in nodes}
            for u, v in self.graph.edges():
                if u in node_to_idx and v in node_to_idx:
                    adj_list[u].append(v)
                    adj_list[v].append(u)
            
            # Compute shells using BFS from central node
            visited = set()
            shells = []
            current_shell = [central_node]
            visited.add(central_node)
            
            while current_shell:
                shells.append(current_shell[:])
                next_shell = []
                for node in current_shell:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_shell.append(neighbor)
                current_shell = next_shell
            
            # Position nodes in concentric circles
            pos = {}
            radius = 1.0
            
            for shell_idx, shell in enumerate(shells):
                if shell_idx == 0:
                    # Center node at origin
                    pos[shell[0]] = np.array([0.0, 0.0])
                else:
                    # Arrange nodes in circle at radius * shell_idx
                    num_nodes = len(shell)
                    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
                    for i, node in enumerate(shell):
                        x = radius * shell_idx * np.cos(angles[i])
                        y = radius * shell_idx * np.sin(angles[i])
                        pos[node] = np.array([x, y])
            
            # Center the layout
            positions = np.array(list(pos.values()))
            positions -= positions.mean(axis=0)
            
            return {node: positions[list(pos.keys()).index(node)] for node in nodes}
        
        # Multiple components - layout each component independently
        component_layouts = []
        component_bounds = []
        
        for component in components:
            comp_nodes = list(component)
            comp_n = len(comp_nodes)
            
            # Layout this component
            comp_pos = self._layout_component_shell(comp_nodes, comp_n)
            
            # Calculate bounding box
            positions = np.array(list(comp_pos.values()))
            min_coords = positions.min(axis=0)
            max_coords = positions.max(axis=0)
            size = max_coords - min_coords
            
            component_layouts.append((comp_nodes, comp_pos))
            component_bounds.append(size)
        
        # Arrange components in a grid with spacing
        num_components = len(components)
        grid_cols = int(np.ceil(np.sqrt(num_components)))
        
        # Calculate spacing based on largest component
        max_width = max(bounds[0] for bounds in component_bounds)
        max_height = max(bounds[1] for bounds in component_bounds)
        spacing_x = max_width * 1.5  # 50% padding between components
        spacing_y = max_height * 1.5
        
        # Place components in grid
        final_positions = {}
        for idx, (comp_nodes, comp_pos) in enumerate(component_layouts):
            grid_x = idx % grid_cols
            grid_y = idx // grid_cols
            
            # Calculate offset for this component
            offset = np.array([grid_x * spacing_x, grid_y * spacing_y])
            
            # Apply offset to all nodes in component
            for node in comp_nodes:
                final_positions[node] = comp_pos[node] + offset
        
        # Center the entire layout
        all_pos = np.array([final_positions[node] for node in nodes])
        all_pos -= all_pos.mean(axis=0)
        
        return {node: all_pos[i] for i, node in enumerate(nodes)}

    def _layout_component_shell(self, nodes, n):
        """
        Shell layout for a single component using numpy for centrality
        """
        if n == 1:
            return {nodes[0]: np.array([0.0, 0.0])}
        
        # Create node to index mapping
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Compute degree centrality using numpy
        degrees = np.zeros(n)
        for u, v in self.graph.edges():
            if u in node_to_idx and v in node_to_idx:
                degrees[node_to_idx[u]] += 1
                degrees[node_to_idx[v]] += 1
        
        # Find most central node (highest degree)
        central_idx = np.argmax(degrees)
        central_node = nodes[central_idx]
        
        # Build adjacency list for BFS
        adj_list = {node: [] for node in nodes}
        for u, v in self.graph.edges():
            if u in node_to_idx and v in node_to_idx:
                adj_list[u].append(v)
                adj_list[v].append(u)
        
        # Compute shells using BFS from central node
        visited = set()
        shells = []
        current_shell = [central_node]
        visited.add(central_node)
        
        while current_shell:
            shells.append(current_shell[:])
            next_shell = []
            for node in current_shell:
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_shell.append(neighbor)
            current_shell = next_shell
        
        # Position nodes in concentric circles
        pos = {}
        radius = 1.0
        
        for shell_idx, shell in enumerate(shells):
            if shell_idx == 0:
                # Center node at origin
                pos[shell[0]] = np.array([0.0, 0.0])
            else:
                # Arrange nodes in circle at radius * shell_idx
                num_nodes = len(shell)
                angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
                for i, node in enumerate(shell):
                    x = radius * shell_idx * np.cos(angles[i])
                    y = radius * shell_idx * np.sin(angles[i])
                    pos[node] = np.array([x, y])
        
        # Center the layout
        positions = np.array(list(pos.values()))
        positions -= positions.mean(axis=0)
        
        return {node: positions[list(pos.keys()).index(node)] for node in nodes}
    
    def _spring_layout_numpy_super(self, nodes, n, iterations=50):
        """
        Spring layout with physically separated connected components
        """
        from scipy.spatial import cKDTree
        import networkx as nx
        
        np.random.seed(42)
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        if len(components) == 1:
            # Single component - use original algorithm
            return self._spring_layout_numpy(nodes, n, iterations)
        
        # Layout each component independently
        component_layouts = []
        component_bounds = []
        
        for component in components:
            comp_nodes = list(component)
            comp_n = len(comp_nodes)
            
            # Create subgraph for this component
            subgraph_edges = [
                (u, v) for u, v in self.graph.edges() 
                if u in component and v in component
            ]
            
            # Run spring layout on this component only
            comp_pos = self._layout_component(comp_nodes, comp_n, subgraph_edges, iterations)
            
            # Calculate bounding box
            positions = np.array(list(comp_pos.values()))
            min_coords = positions.min(axis=0)
            max_coords = positions.max(axis=0)
            size = max_coords - min_coords
            
            component_layouts.append((comp_nodes, comp_pos))
            component_bounds.append(size)
        
        # Arrange components in a grid with spacing
        num_components = len(components)
        grid_cols = int(np.ceil(np.sqrt(num_components)))
        
        # Calculate spacing based on largest component
        max_width = max(bounds[0] for bounds in component_bounds)
        max_height = max(bounds[1] for bounds in component_bounds)
        spacing_x = max_width * 1.5  # 50% padding between components
        spacing_y = max_height * 1.5
        
        # Place components in grid
        final_positions = {}
        for idx, (comp_nodes, comp_pos) in enumerate(component_layouts):
            grid_x = idx % grid_cols
            grid_y = idx // grid_cols
            
            # Calculate offset for this component
            offset = np.array([grid_x * spacing_x, grid_y * spacing_y])
            
            # Apply offset to all nodes in component
            for node in comp_nodes:
                final_positions[node] = comp_pos[node] + offset
        
        # Center the entire layout
        all_pos = np.array([final_positions[node] for node in nodes])
        all_pos -= all_pos.mean(axis=0)
        
        return {node: all_pos[i] for i, node in enumerate(nodes)}

    def _layout_component(self, nodes, n, edges, iterations):
        """
        Spring layout for a single component
        """
        from scipy.spatial import cKDTree
        
        np.random.seed(42 + len(nodes))  # Different seed per component size
        pos = np.random.rand(n, 2)
        
        if len(edges) == 0:
            return {node: pos[i] for i, node in enumerate(nodes)}
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_indices = np.array([[node_to_idx[u], node_to_idx[v]] for u, v in edges])
        
        k = np.sqrt(1.0 / n)
        t = 0.1
        dt = t / (iterations + 1)
        cutoff_distance = 4 * k
        
        for iteration in range(iterations):
            displacement = np.zeros_like(pos)
            
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=cutoff_distance, output_type='ndarray')
            
            if len(pairs) > 0:
                i_indices = pairs[:, 0]
                j_indices = pairs[:, 1]
                
                delta = pos[i_indices] - pos[j_indices]
                distance = np.linalg.norm(delta, axis=1, keepdims=True)
                distance = np.maximum(distance, 0.01)
                
                force_magnitude = (k * k) / distance
                force = delta * (force_magnitude / distance)
                
                np.add.at(displacement, i_indices, force)
                np.add.at(displacement, j_indices, -force)
            
            if len(edge_indices) > 0:
                edge_delta = pos[edge_indices[:, 0]] - pos[edge_indices[:, 1]]
                edge_distance = np.sqrt((edge_delta ** 2).sum(axis=1, keepdims=True))
                edge_distance = np.maximum(edge_distance, 0.01)
                
                edge_force_magnitude = (edge_distance * edge_distance) / k
                edge_force = edge_delta * (edge_force_magnitude / edge_distance)
                
                np.add.at(displacement, edge_indices[:, 0], -edge_force)
                np.add.at(displacement, edge_indices[:, 1], edge_force)
            
            disp_magnitude = np.sqrt((displacement ** 2).sum(axis=1, keepdims=True))
            disp_magnitude = np.maximum(disp_magnitude, 0.01)
            displacement = displacement * np.minimum(t / disp_magnitude, 1.0)
            
            pos += displacement
            t -= dt
        
        pos -= pos.mean(axis=0)
        return {node: pos[i] for i, node in enumerate(nodes)}

    def _spring_layout_numpy(self, nodes, n, iterations = 50):
        """
        Original algorithm for single component case
        """
        from scipy.spatial import cKDTree
        np.random.seed(42)
        pos = np.random.rand(n, 2)
        
        edges = list(self.graph.edges())
        if len(edges) == 0:
            return {node: pos[i] for i, node in enumerate(nodes)}
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_indices = np.array([[node_to_idx[u], node_to_idx[v]] for u, v in edges])
        
        k = np.sqrt(1.0 / n)
        t = 0.1
        dt = t / (iterations + 1)
        cutoff_distance = 4 * k
        
        for iteration in range(iterations):
            displacement = np.zeros_like(pos)
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=cutoff_distance, output_type='ndarray')
            
            if len(pairs) > 0:
                i_indices = pairs[:, 0]
                j_indices = pairs[:, 1]
                delta = pos[i_indices] - pos[j_indices]
                distance = np.linalg.norm(delta, axis=1, keepdims=True)
                distance = np.maximum(distance, 0.01)
                force_magnitude = (k * k) / distance
                force = delta * (force_magnitude / distance)
                np.add.at(displacement, i_indices, force)
                np.add.at(displacement, j_indices, -force)
            
            if len(edge_indices) > 0:
                edge_delta = pos[edge_indices[:, 0]] - pos[edge_indices[:, 1]]
                edge_distance = np.sqrt((edge_delta ** 2).sum(axis=1, keepdims=True))
                edge_distance = np.maximum(edge_distance, 0.01)
                edge_force_magnitude = (edge_distance * edge_distance) / k
                edge_force = edge_delta * (edge_force_magnitude / edge_distance)
                np.add.at(displacement, edge_indices[:, 0], -edge_force)
                np.add.at(displacement, edge_indices[:, 1], edge_force)
            
            disp_magnitude = np.sqrt((displacement ** 2).sum(axis=1, keepdims=True))
            disp_magnitude = np.maximum(disp_magnitude, 0.01)
            displacement = displacement * np.minimum(t / disp_magnitude, 1.0)
            pos += displacement
            t -= dt
        
        pos -= pos.mean(axis=0)
        return {node: pos[i] for i, node in enumerate(nodes)}

    def _prepare_node_spots(self, pos, colors, sizes):
        """Prepare spots array and brush caches for ScatterPlotItem"""
        nodes = list(self.graph.nodes())
        pos_array = np.array([pos[n] for n in nodes])
        
        # Pre-compute both normal and selected brushes in separate caches
        spots = []
        brush_cache = {}  # {node: {'normal': brush, 'selected': brush}}
        
        for i, node in enumerate(nodes):
            hex_color = colors[i]
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Create brushes (do this in thread to save time later)
            normal_brush = pg.mkBrush(*rgb, 200)
            selected_brush = pg.mkBrush(255, 255, 0, 255)  # Yellow for selected
            
            # Store brushes separately
            brush_cache[node] = {
                'normal': normal_brush,
                'selected': selected_brush
            }
            
            # Only include pyqtgraph-valid parameters in spot
            spot = {
                'pos': pos_array[i],
                'size': sizes[i],
                'brush': normal_brush,  # Start with normal brush
                'data': node
            }
            spots.append(spot)
        
        return spots, brush_cache

    def _prepare_label_data(self, pos):
        """Prepare label positions and text"""
        label_data = []
        for node in self.graph.nodes():
            if node in pos:
                label_data.append({
                    'node': node,
                    'text': str(node),
                    'pos': pos[node]
                })
        return label_data

    def _prepare_edge_pens(self, edges):
        """Prepare edge drawing data - batch edges by weight bins for efficient rendering"""
        if not edges:
            return []
        
        # If weights are disabled, use uniform thickness
        if not self.weight:
            # All edges get same thickness - combine into single batch
            x_coords = []
            y_coords = []
            for x, y, weight in edges:
                x_coords.extend([x[0], x[1], np.nan])
                y_coords.extend([y[0], y[1], np.nan])
            
            return [{
                'x': np.array(x_coords),
                'y': np.array(y_coords),
                'thickness': self.edge_size
            }]
        
        # Weight-based rendering - batch by thickness
        weights = [w for _, _, w in edges]
        if not weights:
            return []
        
        min_weight = min(weights)
        max_weight = max(weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1
        
        # Define thickness bins (e.g., 10 discrete thickness levels)
        num_bins = 10
        thickness_min = self.edge_size/2
        thickness_max = 3.0 * self.edge_size  # Maximum thickness cap
        
        # Batch edges by thickness bin
        edge_batches = {}  # {thickness: [(x_coords, y_coords), ...]}
        
        for x, y, weight in edges:
            # Normalize weight to thickness
            if weight_range > 0:
                normalized = (weight - min_weight) / weight_range
            else:
                normalized = self.edge_size/2
            
            # Calculate thickness with cap
            thickness = thickness_min + normalized * (thickness_max - thickness_min)
            
            # Bin the thickness to reduce number of batches
            thickness_bin = round(thickness * num_bins) / num_bins
            thickness_bin = min(thickness_bin, thickness_max)  # Apply cap
            
            # Add to batch
            if thickness_bin not in edge_batches:
                edge_batches[thickness_bin] = {'x': [], 'y': []}
            
            # Add edge coordinates with NaN separator
            edge_batches[thickness_bin]['x'].extend([x[0], x[1], np.nan])
            edge_batches[thickness_bin]['y'].extend([y[0], y[1], np.nan])
        
        # Convert to list format for rendering
        batch_data = []
        for thickness, coords in edge_batches.items():
            batch_data.append({
                'x': np.array(coords['x']),
                'y': np.array(coords['y']),
                'thickness': thickness
            })
        
        return batch_data
    
    def _compute_geometric_layout(self):
        """Compute positions from centroids"""
        pos = {}
        for node in self.graph.nodes():
            if node in self.centroids:
                z, y, x = self.centroids[node]
                pos[node] = np.array([x, -y])
            else:
                pos[node] = np.array([0, 0])
        return pos
    
    def _compute_node_attributes(self):
        """Compute node colors and sizes"""
        nodes = list(self.graph.nodes())
        colors = []
        sizes = self._compute_all_node_sizes_vectorized(nodes)

        # Determine coloring mode
        if self.identities and self.identity_dict:
            color_map = self._generate_community_colors(self.identity_dict)
            for node in self.graph.nodes():
                identity = self.identity_dict.get(node, 'Unknown')
                identity = str(identity)
                colors.append(color_map.get(identity, '#808080'))
        elif self.communities and self.community_dict:
            color_map = self._generate_community_colors(self.community_dict)
            for node in self.graph.nodes():
                community = self.community_dict.get(node, -1)
                colors.append(color_map.get(community, '#808080'))
        else:
            # Default coloring
            for node in self.graph.nodes():
                colors.append('#4A90E2')
        
        return colors, sizes
    
    def _compute_all_node_sizes_vectorized(self, nodes):
        if not self.geometric or not self.centroids or not self.z_size:
            return [self.node_size] * len(nodes)

        # GLOBAL z range (matches original behavior)
        all_z = np.array([
            self.centroids[n][0]
            for n in self.graph.nodes()
            if n in self.centroids
        ])

        if len(all_z) == 0:
            return [10] * len(nodes)

        z_min, z_max = all_z.min(), all_z.max()

        sizes = [10] * len(nodes)

        if z_max <= z_min:
            return sizes

        # Collect z-values ONLY for requested nodes
        z_values = []
        node_indices = []

        for i, node in enumerate(nodes):
            if node in self.centroids:
                z_values.append(self.centroids[node][0])
                node_indices.append(i)

        if not z_values:
            return sizes

        z_array = np.array(z_values)

        normalized = 1 - (z_array - z_min) / (z_max - z_min)
        computed_sizes = 5 + normalized * 20

        for idx, node_idx in enumerate(node_indices):
            sizes[node_idx] = float(computed_sizes[idx])

        return sizes
    
    def _generate_identity_colors(self):
        """Generate colors for identities using the specified strategy"""
        unique_categories = list(set(self.identity_dict.values()))
        num_categories = len(unique_categories)
        
        if num_categories <= 12:
            base_colors = [
                '#FF0000', '#0066FF', '#00CC00', '#FF8800',
                '#8800FF', '#FFFF00', '#FF0088', '#00FFFF',
                '#88FF00', '#FF4400', '#0088FF', '#CC00FF'
            ]
            colors = base_colors[:num_categories]
        else:
            colors = []
            for i in range(num_categories):
                hue = (i * 360 / num_categories) % 360
                sat = 0.85 if i % 2 == 0 else 0.95
                val = 0.95 if i % 3 != 0 else 0.85
                
                rgb = colorsys.hsv_to_rgb(hue/360, sat, val)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                )
                colors.append(hex_color)
        
        return dict(zip(unique_categories, colors))
    
    def _generate_community_colors(self, my_dict):
        """Generate colors for communities using the specified strategy"""
        from collections import Counter
        
        try:
            unique_communities = sorted(set(my_dict.values()))
            some_dict = my_dict
        except:
            some_dict = {node: str(comm) for node, comm in my_dict.items()}
            unique_communities = sorted(set(some_dict.values()))

        community_sizes = Counter(some_dict.values())
        sorted_communities = random.Random(42).sample(unique_communities, len(unique_communities))
        colors_rgb = self._generate_distinct_colors_rgb(len(unique_communities))
        color_map = {comm: colors_rgb[i] for i, comm in enumerate(sorted_communities)}
        if 0 in unique_communities:
            color_map[0] = "#8B4513"

        return color_map
    
    def _generate_distinct_colors_rgb(self, n_colors):
        """
        Generate visually distinct RGB colors using HSV color space.
        Colors are generated with maximum saturation and value, varying only in hue.
        """
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # S=1, V=1 for max saturation/brightness
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors
    
    def _compute_edge_data(self, pos):
        """Compute edge coordinates and weights"""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in pos and v in pos:
                weight = data.get('weight', 1.0)
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                edges.append((x, y, weight))
        return edges


class NetworkGraphWidget(QWidget):
    """Interactive NetworkX graph visualization widget"""
    
    node_selected = pyqtSignal(object)  # Emits list of selected nodes
    
    def __init__(self, parent=None, weight=False, geometric=False, component = False, 
                 centroids=None, communities=False, community_dict=None,
                 identities=False, identity_dict=None, labels=False, z_size = False, 
                 shell = False, node_size = 10, black_edges = False, edge_size = 1, popout = False):
        super().__init__(parent)
        
        self.parent_window = parent
        self.weight = weight
        self.geometric = geometric
        self.component = component
        self.centroids = centroids or {}
        self.communities = communities
        self.community_dict = community_dict or {}
        self.identities = identities
        self.identity_dict = remove_dupe_ids(identity_dict) or {}
        self.labels = labels
        self.z_size = z_size
        self.shell = shell
        self.node_size = node_size
        self.black_edges = black_edges
        self.edge_size = edge_size
        self.popout = popout
        
        # Graph data
        self.graph = None
        self.node_positions = {}
        self.node_colors = []
        self.node_sizes = []
        self.node_items = {}
        self.edge_items = []
        self.label_items = {}
        self.label_data = []  # Store label data for on-demand rendering
        self.selected_nodes = set()
        self.node_click = False
        self.rendered = False
        
        # CACHING for fast updates
        self.cached_spots = []  # Full spot data with brushes
        self.cached_node_to_index = {}  # Node -> spot index mapping
        self.cached_brushes = {}  # Node -> {'normal': brush, 'selected': brush}
        self.last_selected_set = set()  # Track last selection state
        self.cached_sizes_for_lod = []  # Base sizes for LOD scaling
        
        # Interaction mode
        self.selection_mode = True
        self.zoom_mode = False
        
        # Area selection
        self.selection_rect = None
        self.selection_start_pos = None
        self.is_area_selecting = False
        self.click_timer = None
        
        # Middle mouse panning in selection mode
        self.temp_pan_active = False
        self.last_mouse_pos = None
        
        # Wheel zoom timer for selection mode
        self.wheel_timer = None
        self.was_in_selection_before_wheel = False
        
        # Thread for loading
        self.load_thread = None
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QHBoxLayout()  # Changed from QVBoxLayout to accommodate legend
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Left side: graph container
        graph_container = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(2)
        
        # Create graphics layout widget
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')
        
        # Create plot
        self.plot = self.graphics_widget.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.showGrid(x=False, y=False)
        # Show loading indicator
        self.loading_text = pg.TextItem(
            text="No network detected",
            color=(100, 100, 100),
            anchor=(0.5, 0.5)
        )
        self.loading_text.setPos(0, 0)  # Center of view
        self.plot.addItem(self.loading_text)
        
        # Enable mouse tracking for area selection
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # Disable default mouse interaction - will enable only in pan mode
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.vb.setMenuEnabled(False)
        self.plot.vb.setMouseMode(pg.ViewBox.PanMode)
        
        # Create scatter plot for nodes
        self.scatter = ScatterPlotItem(size=10, pen=pg.mkPen(None), 
                                      brush=pg.mkBrush(74, 144, 226, 200))
        self.plot.addItem(self.scatter)
        
        # Connect click events
        self.scatter.sigClicked.connect(self._on_node_clicked)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        
        # Connect view change for level-of-detail updates
        self.plot.sigRangeChanged.connect(self._on_view_changed)
        
        # Level of detail parameters
        self.base_node_sizes = []
        self.current_zoom_factor = 1.0
        
        graph_layout.addWidget(self.graphics_widget, stretch=1)  # Keep this one
        
        # Create control panel
        control_panel = self._create_control_panel()
        graph_layout.addWidget(control_panel)
        
        graph_container.setLayout(graph_layout)
        layout.addWidget(graph_container, stretch=1)
        
        # Right side: legend (placeholder, will be populated when graph loads)
        self.legend_container = QWidget()
        self.legend_layout = QVBoxLayout()
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_container.setLayout(self.legend_layout)
        self.legend_container.setMaximumWidth(0)  # Hidden initially
        layout.addWidget(self.legend_container)
        
        self.setLayout(layout)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, 
                          QSizePolicy.Policy.Expanding)
        
        # Install event filter for custom mouse handling
        self.graphics_widget.viewport().installEventFilter(self)
        self.plot.scene().installEventFilter(self)
    
    def _create_identity_legend(self):
        """Create a legend panel for node identities"""

        def _generate_distinct_colors_rgb(n_colors: int):
            """
            Generate visually distinct RGB colors using HSV color space.
            Colors are generated with maximum saturation and value, varying only in hue.
            """
            colors = []
            for i in range(n_colors):
                hue = i / n_colors
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # S=1, V=1 for max saturation/brightness
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                colors.append(hex_color)
            return colors

        if self.identities:
            from collections import Counter
            try:
                unique_identities = sorted(set(self.identity_dict.values()))
                community_sizes = Counter(self.identity_dict.values())
            except:
                some_dict = {node: str(comm) for node, comm in self.identity_dict.items()}
                unique_identities = sorted(set(some_dict.values()))
                community_sizes = Counter(some_dict.values())

            sorted_communities = random.Random(42).sample(unique_identities, len(unique_identities))
            colors_rgb = _generate_distinct_colors_rgb(len(unique_identities))
            color_map = {comm: colors_rgb[i] for i, comm in enumerate(sorted_communities)}
            if 0 in unique_identities:
                color_map[0] = "#8B4513"
        elif self.communities:
            from collections import Counter
            
            unique_identities = sorted(set(self.community_dict.values()))
            community_sizes = Counter(self.community_dict.values())
            sorted_communities = random.Random(42).sample(unique_identities, len(unique_identities))
            colors_rgb = _generate_distinct_colors_rgb(len(unique_identities))
            color_map = {comm: colors_rgb[i] for i, comm in enumerate(sorted_communities)}
            if 0 in unique_identities:
                color_map[0] = "#8B4513"

        # Create legend widget
        legend_widget = QWidget()
        legend_layout = QVBoxLayout()
        legend_layout.setContentsMargins(5, 5, 5, 5)
        legend_layout.setSpacing(2)
        
        # Add title
        if self.identities:
            title = QLabel("Node Identities")
        elif self.communities:
            title = QLabel("Node Community")
        title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 3px;")
        legend_layout.addWidget(title)
        
        # Create scrollable area for legend items
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(200)
        scroll.setMinimumWidth(150)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Container for legend items
        items_widget = QWidget()
        items_layout = QVBoxLayout()
        items_layout.setContentsMargins(2, 2, 2, 2)
        items_layout.setSpacing(3)
        
        # Add each identity with colored box
        for identity in unique_identities:
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)
            
            # Color box
            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"background-color: {color_map[identity]}; border: 1px solid #888;")
            
            # Label
            label = QLabel(str(identity))
            label.setStyleSheet("font-size: 9pt;")
            
            item_layout.addWidget(color_box)
            item_layout.addWidget(label)
            item_layout.addStretch()
            
            item_widget.setLayout(item_layout)
            items_layout.addWidget(item_widget)

        if '#808080' in self.node_colors:
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)
            
            # Color box
            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"background-color: #808080; border: 1px solid #888;")
            
            # Label
            label = QLabel('Unassigned')
            label.setStyleSheet("font-size: 9pt;")
            
            item_layout.addWidget(color_box)
            item_layout.addWidget(label)
            item_layout.addStretch()
            
            item_widget.setLayout(item_layout)
            items_layout.addWidget(item_widget)
        
        items_layout.addStretch()
        items_widget.setLayout(items_layout)
        scroll.setWidget(items_widget)
        
        legend_layout.addWidget(scroll)
        legend_widget.setLayout(legend_layout)
        
        return legend_widget

    def _create_control_panel(self):
        """Create the control panel with emoji buttons"""
        panel = QWidget()
        panel_layout = QHBoxLayout()
        panel_layout.setContentsMargins(2, 2, 2, 2)
        panel_layout.setSpacing(2)
        
        # Create buttons with emojis
        self.select_btn = QPushButton("🖱️")
        self.select_btn.setToolTip("Selection Tool")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.setMaximumSize(32, 32)
        self.select_btn.clicked.connect(self._toggle_selection_mode)
        
        self.pan_btn = QPushButton("✋")
        self.pan_btn.setToolTip("Pan Tool")
        self.pan_btn.setCheckable(True)
        self.pan_btn.setChecked(False)
        self.pan_btn.setMaximumSize(32, 32)
        self.pan_btn.clicked.connect(self._toggle_pan_mode)
        
        self.zoom_btn = QPushButton("🔍")
        self.zoom_btn.setToolTip("Zoom Tool (Left Click: Zoom Out, Right Click: Zoom In)")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setMaximumSize(32, 32)
        self.zoom_btn.clicked.connect(self._toggle_zoom_mode)
        
        self.home_btn = QPushButton("🏠")
        self.home_btn.setToolTip("Reset View")
        self.home_btn.setMaximumSize(32, 32)
        self.home_btn.clicked.connect(self._reset_view)
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setToolTip("Refresh Graph")
        self.refresh_btn.setMaximumSize(32, 32)
        self.refresh_btn.clicked.connect(self.load_graph)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setToolTip("Render Settings")
        self.settings_btn.setMaximumSize(32, 32)
        self.settings_btn.clicked.connect(self.settings)
        
        self.clear_btn = QPushButton("🗑️")
        self.clear_btn.setToolTip("Clear Graph")
        self.clear_btn.setMaximumSize(32, 32)
        self.clear_btn.clicked.connect(self._clear_graph)

        
        # Add buttons to layout
        panel_layout.addWidget(self.select_btn)
        panel_layout.addWidget(self.pan_btn)
        panel_layout.addWidget(self.zoom_btn)
        panel_layout.addWidget(self.home_btn)
        panel_layout.addWidget(self.refresh_btn)
        panel_layout.addWidget(self.settings_btn)
        panel_layout.addWidget(self.clear_btn)

        if self.popout:
            self.popout_btn = QPushButton("⤴")
            self.popout_btn.setToolTip("Full Screen")
            self.popout_btn.setMaximumSize(32, 32)
            self.popout_btn.clicked.connect(self._popout_graph)
            panel_layout.addWidget(self.popout_btn)

        panel_layout.addStretch()
        
        panel.setLayout(panel_layout)
        panel.setMaximumHeight(40)
        
        return panel

    def settings(self):

        self.parent_window.show_netshow_dialog(called = self)
    
    def set_graph(self, graph):
        """Set the NetworkX graph to visualize"""
        self.graph = graph

        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None

        if self.graph is not None and not self.rendered:
            if hasattr(self, 'loading_text') and self.loading_text is not None:
                self.plot.removeItem(self.loading_text)
                self.loading_text = None
            # Show loading indicator
            self.loading_text = pg.TextItem(
                text="Press 🔄 to load your graph",
                color=(100, 100, 100),
                anchor=(0.5, 0.5)
            )
            self.loading_text.setPos(0, 0)  # Center of view
            self.plot.addItem(self.loading_text)
        elif not self.rendered:
            self.loading_text = pg.TextItem(
                text="No network detected",
                color=(100, 100, 100),
                anchor=(0.5, 0.5)
            )
            self.loading_text.setPos(0, 0)  # Center of view
            self.plot.addItem(self.loading_text)
    
    def load_graph(self):
        """Load and render the graph (in separate thread)"""

        # Clear existing visualization
        self._clear_graph()
        self.get_properties()

        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None

        if self.graph is None or len(self.graph.nodes()) == 0:
            # Show loading indicator
            self.loading_text = pg.TextItem(
                text="No network detected",
                color=(100, 100, 100),
                anchor=(0.5, 0.5)
            )
            self.loading_text.setPos(0, 0)  # Center of view
            self.plot.addItem(self.loading_text)
            return

        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None

        # Show loading indicator
        self.loading_text = pg.TextItem(
            text="Loading graph...",
            color=(100, 100, 100),
            anchor=(0.5, 0.5)
        )
        self.loading_text.setPos(0, 0)  # Center of view
        self.plot.addItem(self.loading_text)
        
        # Start loading in thread
        self.load_thread = GraphLoadThread(
            self.graph, self.geometric, self.component, self.centroids,
            self.communities, self.community_dict,
            self.identities, self.identity_dict, self.weight, self.z_size,
            self.shell, self.node_size, self.edge_size
        )
        self.load_thread.finished.connect(self._on_graph_loaded)
        self.load_thread.start()

    @pyqtSlot(object)
    def _on_graph_loaded(self, result):
        """Handle loaded graph data from thread"""
        # Remove loading indicator
        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None
        
        self.node_positions = result['pos']
        self.node_colors = result['colors']
        self.node_sizes = result['sizes']
        self.base_node_sizes = result['sizes'].copy()
        
        # Cache the prepared data for fast updates
        self.cached_spots = result['node_spots']
        self.cached_brushes = result['brush_cache']
        self.cached_sizes_for_lod = result['sizes'].copy()
        
        # Build node-to-index mapping
        self.cached_node_to_index = {spot['data']: i 
                                      for i, spot in enumerate(self.cached_spots)}
        
        # Fast render - data is already prepared
        self._render_prepared_data(result)
        
        # Reset view to show entire graph
        self.rendered = True
        # Block signals during reset to avoid triggering _on_view_changed
        self.plot.blockSignals(True)
        self._reset_view()
        self.plot.blockSignals(False)
        # Add legend if identities are enabled
        if (self.identities and self.identity_dict) or (self.communities and self.community_dict):
            # Clear old legend
            for i in reversed(range(self.legend_layout.count())): 
                self.legend_layout.itemAt(i).widget().setParent(None)
            
            # Create and add new legend
            legend = self._create_identity_legend()
            if legend:
                self.legend_layout.addWidget(legend)
                self.legend_container.setMaximumWidth(200)
        else:
            self.legend_container.setMaximumWidth(0)  # Hide if no identities
        if len(self.parent_window.clicked_values['nodes']) > 0:
            self.select_nodes(self.parent_window.clicked_values['nodes'])



    def _render_prepared_data(self, result):
        """Render pre-computed data (minimal main thread work)"""
        # Clear old items
        self.scatter.clear()
        for item in self.edge_items:
            self.plot.removeItem(item)
        self.edge_items.clear()
        for label_item in self.label_items.values():
            self.plot.removeItem(label_item)
        self.label_items.clear()

        if self.black_edges:
            edge_color = (0, 0, 0)
        else:
            edge_color = (150, 150, 150, 100)
        
        # Render edges - batched by weight for efficiency
        edge_batches = result['edge_pens']
        if edge_batches:
            for batch in edge_batches:
                edge_line = PlotCurveItem(
                    x=batch['x'],
                    y=batch['y'],
                    pen=pg.mkPen(color=edge_color, width=batch['thickness']),
                    connect='finite'  # Break lines at NaN
                )
                self.plot.addItem(edge_line)
                self.edge_items.append(edge_line)
        
        # Render nodes - use cached spots directly
        self.scatter.setData(spots=self.cached_spots)
        self.scatter.setZValue(10)
        
        # Build node items mapping
        nodes = list(self.graph.nodes())
        self.node_items = {node: i for i, node in enumerate(nodes)}
        
        # Store label data for later rendering
        self.label_data = result['label_data']
        
        # Only render labels immediately if graph is small (< 100 nodes)
        if self.labels and len(self.label_data) < 100:
            self._update_labels_in_viewport(len(self.label_data))

    def _render_nodes(self):
        """OPTIMIZED: Only update brushes for nodes that changed selection state"""
        
        if not self.cached_spots or not self.cached_brushes:
            return
        
        # Find nodes whose selection state changed
        newly_selected = self.selected_nodes - self.last_selected_set
        newly_deselected = self.last_selected_set - self.selected_nodes
        
        # If nothing changed, skip update
        if not newly_selected and not newly_deselected:
            return
        
        # Update only changed nodes using cached brushes
        for node in newly_selected:
            if node in self.cached_node_to_index:
                idx = self.cached_node_to_index[node]
                self.cached_spots[idx]['brush'] = self.cached_brushes[node]['selected']
        
        for node in newly_deselected:
            if node in self.cached_node_to_index:
                idx = self.cached_node_to_index[node]
                self.cached_spots[idx]['brush'] = self.cached_brushes[node]['normal']
        
        # Update the scatter plot with modified spots
        self.scatter.setData(spots=self.cached_spots)
        
        # Update last selection state
        self.last_selected_set = self.selected_nodes.copy()
    
    def _render_labels_batch(self, label_data_subset):
        """Render labels in batches (but still slow for large graphs)"""
        # Clear existing labels first
        for label_item in self.label_items.values():
            self.plot.removeItem(label_item)
        self.label_items.clear()
        
        if not label_data_subset:
            return
        
        # Batch size for yielding control back to event loop
        batch_size = 50
        
        for i, label_info in enumerate(label_data_subset):
            text_item = pg.TextItem(
                text=label_info['text'],
                color=(0, 0, 0),
                anchor=(0.5, 0.5)
            )
            text_item.setPos(label_info['pos'][0], label_info['pos'][1])
            text_item.setZValue(20)
            self.plot.addItem(text_item)
            self.label_items[label_info['node']] = text_item
            
            # Yield control periodically to keep UI responsive
            if (i + 1) % batch_size == 0:
                QApplication.processEvents()
    
    def _update_labels_for_zoom(self):
        """Show/hide labels based on zoom level and graph size with viewport awareness"""
        if not self.labels or not self.label_data:
            return
        
        num_nodes = len(self.label_data)
        
        # Determine zoom threshold based on graph size
        if num_nodes < 100:
            zoom_threshold = 0  # Always show labels
        elif num_nodes < 500:
            zoom_threshold = 1.5
        else:
            zoom_threshold = 3.0
        
        # Check if we're above the zoom threshold
        should_show_labels = self.current_zoom_factor > zoom_threshold
        
        if should_show_labels:
            # Update labels based on current viewport
            self._update_labels_in_viewport(num_nodes)
        else:
            # Clear all labels when zoomed out below threshold
            if self.label_items:
                for label_item in self.label_items.values():
                    self.plot.removeItem(label_item)
                self.label_items.clear()
    
    def _update_labels_in_viewport(self, num_nodes):
        """Update labels to show only those in current viewport"""
        # Get current view range
        view_range = self.plot.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]
        
        # Find which labels should be visible
        visible_node_set = set()
        labels_to_render = []
        
        for label_info in self.label_data:
            if x_min <= label_info['pos'][0] <= x_max and y_min <= label_info['pos'][1] <= y_max:
                visible_node_set.add(label_info['node'])
                labels_to_render.append(label_info)
        
        # For small graphs or when not many visible labels, render all in viewport
        max_visible_labels = 200 if num_nodes >= 500 else 1000
        
        if len(labels_to_render) > max_visible_labels:
            # Too many labels to render - skip
            if self.label_items:
                for label_item in self.label_items.values():
                    self.plot.removeItem(label_item)
                self.label_items.clear()
            return
        
        # Get currently rendered nodes
        current_node_set = set(self.label_items.keys())
        
        # Remove labels for nodes no longer in viewport
        nodes_to_remove = current_node_set - visible_node_set
        for node in nodes_to_remove:
            if node in self.label_items:
                self.plot.removeItem(self.label_items[node])
                del self.label_items[node]
        
        # Add labels for new nodes in viewport
        nodes_to_add = visible_node_set - current_node_set
        for label_info in labels_to_render:
            node = label_info['node']
            if node in nodes_to_add:
                text_item = pg.TextItem(
                    text=label_info['text'],
                    color=(0, 0, 0),
                    anchor=(0.5, 0.5)
                )
                text_item.setPos(label_info['pos'][0], label_info['pos'][1])
                text_item.setZValue(20)
                self.plot.addItem(text_item)
                self.label_items[node] = text_item
    
    def _render_labels(self):
        """Render node labels if enabled (legacy method - kept for compatibility)"""
        # Use the smart rendering instead
        if self.label_data:
            self._update_labels_for_zoom()
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _on_plot_clicked(self, ev):
        """Handle clicks on the plot background"""
        if not self.selection_mode or self.node_click or not self.popout:
            self.node_click = False
            return
        
        # Only handle left button clicks
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        
        # Get the position in scene coordinates
        scene_pos = ev.scenePos()
        
        
        # Click was on background
        modifiers = ev.modifiers()
        ctrl_pressed = modifiers & Qt.KeyboardModifier.ControlModifier
        
        if not ctrl_pressed:
            # Deselect all nodes
            self.selected_nodes.clear()
            self._render_nodes()
            self.push_selection()
            self.node_selected.emit([])
        # Ctrl+Click on background does nothing (as requested)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for area selection"""
        if self.is_area_selecting and self.selection_rect:
            # Update selection rectangle
            view_pos = self.plot.vb.mapSceneToView(pos)
            start_pos = self.selection_start_pos
            
            # Update rectangle size
            width = view_pos.x() - start_pos.x()
            height = view_pos.y() - start_pos.y()
            
            self.selection_rect.setSize([width, height])
    
    def _start_area_selection(self, scene_pos):
        """Start area selection with rectangle"""
        self.is_area_selecting = True
        self.node_click = False
        
        # Convert to view coordinates
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        self.selection_start_pos = view_pos
        
        # Create selection rectangle
        if self.selection_rect:
            self.plot.removeItem(self.selection_rect)
        
        # Create ROI for selection area
        self.selection_rect = pg.ROI(
            [view_pos.x(), view_pos.y()],
            [0, 0],
            pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine),
            movable=False,
            resizable=False
        )
        self.selection_rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.plot.addItem(self.selection_rect)
    
    def _finish_area_selection(self, ev):
        """Finish area selection and select nodes in rectangle"""
        if not self.is_area_selecting or not self.selection_rect:
            return
        
        # Get rectangle bounds
        rect_pos = self.selection_rect.pos()
        rect_size = self.selection_rect.size()
        
        x_min = rect_pos[0]
        y_min = rect_pos[1]
        x_max = x_min + rect_size[0]
        y_max = y_min + rect_size[1]
        
        # Normalize bounds (in case user dragged backwards)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Remove selection rectangle
        if self.selection_rect:
            self.plot.removeItem(self.selection_rect)
            self.selection_rect = None
        
        self.is_area_selecting = False
        self.selection_start_pos = None
        
        # ZOOM MODE: Zoom to rectangle
        if self.zoom_mode:
            # Add small padding (5%)
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = 0.05
            
            self.plot.setXRange(x_min - padding * x_range, 
                               x_max + padding * x_range, padding=0)
            self.plot.setYRange(y_min - padding * y_range, 
                               y_max + padding * y_range, padding=0)
            return
        
        # SELECTION MODE: Select nodes in rectangle
        if self.selection_mode:
            # Find nodes in rectangle
            selected_in_rect = []
            for node, pos in self.node_positions.items():
                if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                    selected_in_rect.append(node)
            
            # Add to selection
            modifiers = ev.modifiers()
            ctrl_pressed = modifiers & Qt.KeyboardModifier.ControlModifier
            
            if not ctrl_pressed:
                self.selected_nodes = set()
            self.selected_nodes.update(selected_in_rect)
            self.push_selection()
            
            # Update visual representation
            self._render_nodes()
            
            # Emit signal
            self.node_selected.emit(list(self.selected_nodes))
    
    def _toggle_selection_mode(self):
        """Toggle selection mode"""
        self.selection_mode = self.select_btn.isChecked()
        
        if self.selection_mode:
            self.pan_btn.setChecked(False)
            self.zoom_btn.setChecked(False)
            self.zoom_mode = False
            # Disable panning, but allow wheel events to be handled manually
            self.plot.setCursor(Qt.CursorShape.ArrowCursor)
            self.plot.vb.setMenuEnabled(False)
            self.plot.setMouseEnabled(x=False, y=False)
        else:
            # If nothing else is checked, check pan by default
            if not self.pan_btn.isChecked() and not self.zoom_btn.isChecked():
                self.pan_btn.click()
    
    def _toggle_pan_mode(self):
        """Toggle pan mode"""
        if self.pan_btn.isChecked():
            self.select_btn.setChecked(False)
            self.zoom_btn.setChecked(False)
            self.selection_mode = False
            self.zoom_mode = False
            # Enable panning
            self.plot.vb.setMenuEnabled(True)
            self.plot.setCursor(Qt.CursorShape.OpenHandCursor)
            self.plot.setMouseEnabled(x=True, y=True)
        else:
            # Disable panning
            if not self.select_btn.isChecked() and not self.zoom_btn.isChecked():
                self.select_btn.click()

    def _toggle_zoom_mode(self):
        """Toggle zoom mode"""
        self.zoom_mode = self.zoom_btn.isChecked()
        
        if self.zoom_mode:
            self.select_btn.setChecked(False)
            self.pan_btn.setChecked(False)
            self.selection_mode = False
            # Disable default panning for zoom mode
            self.plot.setCursor(Qt.CursorShape.CrossCursor)
            self.plot.vb.setMenuEnabled(False)
            self.plot.setMouseEnabled(x=False, y=False)
        else:
            # If nothing else is checked, check pan by default
            if not self.pan_btn.isChecked() and not self.select_btn.isChecked():
                self.select_btn.click()
        
    def eventFilter(self, obj, event):
        """Filter events for custom mouse handling"""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        
        # Only handle events for the graphics scene
        if obj != self.plot.scene():
            return super().eventFilter(obj, event)
        
        if event.type() == QEvent.Type.GraphicsSceneMousePress:            
            # Handle middle mouse button for temporary panning in selection mode
            if event.button() == Qt.MouseButton.MiddleButton:
                if self.selection_mode or self.zoom_mode:
                    self._start_temp_pan()
                    return False  # Let the event propagate for panning
            
            # SELECTION MODE: Handle left button for area selection
            elif event.button() == Qt.MouseButton.LeftButton and (self.selection_mode or self.zoom_mode):
                # Store position and start timer for long press detection
                self.last_mouse_pos = event.scenePos()
                if not self.click_timer:
                    self.click_timer = QTimer()
                    self.click_timer.setSingleShot(True)
                    self.click_timer.timeout.connect(self._on_long_press)
                self.click_timer.start(200)  # 200ms threshold for area selection
        
        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
            # Check if we should start area selection
            if (self.selection_mode or self.zoom_mode) and self.click_timer and self.click_timer.isActive():
                if self.last_mouse_pos:
                    # Check if mouse moved significantly
                    current_pos = event.scenePos()
                    delta_x = abs(current_pos.x() - self.last_mouse_pos.x())
                    delta_y = abs(current_pos.y() - self.last_mouse_pos.y())
                    
                    if delta_x > 10 or delta_y > 10:  # Moved significantly
                        self.click_timer.stop()
                        if not self.is_area_selecting:
                            #if not self.was_in_selection_before_wheel or self.zoom_mode:
                            self._start_area_selection(self.last_mouse_pos)

        
        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease:
            # Handle middle mouse release
            if event.button() == Qt.MouseButton.MiddleButton:
                if self.temp_pan_active:
                    self._end_temp_pan()
                    return False  # Let event propagate
            
            # Handle left button release in selection mode
            elif event.button() == Qt.MouseButton.LeftButton and (self.selection_mode or self.zoom_mode):
                if self.click_timer and self.click_timer.isActive():
                    self.click_timer.stop()
                
                if self.is_area_selecting:
                    self._finish_area_selection(event)
                    return True  # Consume event
            elif event.button() == Qt.MouseButton.RightButton and self.selection_mode:
                mouse_point = self.plot.getViewBox().mapSceneToView(event.scenePos())
                x, y = mouse_point.x(), mouse_point.y()
                self.create_context_menu(event)

            # ZOOM MODE: Handle left/right click for zoom
            if self.zoom_mode:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Zoom in
                    self._zoom_at_point(event.scenePos(), 2)
                    return True
                elif event.button() == Qt.MouseButton.RightButton:
                    # Zoom out
                    self._zoom_at_point(event.scenePos(), 0.5)
                    return True
        
        elif event.type() == QEvent.Type.GraphicsSceneWheel:
            # Handle wheel events in selection mode
            if self.selection_mode or self.zoom_mode:
                self._handle_wheel_in_selection(event)
                return False  # Let event propagate for actual zooming
        
        return super().eventFilter(obj, event)
    
    def _on_long_press(self):
        """Handle long press for area selection"""
        if (self.selection_mode or self.zoom_mode) and self.last_mouse_pos:
            # Start area selection at stored mouse position
            self._start_area_selection(self.last_mouse_pos)
    
    def _start_temp_pan(self):
        """Start temporary panning mode with middle mouse"""
        self.temp_pan_active = True
        # Temporarily enable panning
        self.plot.setMouseEnabled(x=True, y=True)
    
    def _end_temp_pan(self):
        """End temporary panning mode"""
        self.temp_pan_active = False
        # Disable mouse if we're in selection mode
        if self.selection_mode or self.zoom_mode:
            self.plot.setMouseEnabled(x=False, y=False)
    
    def _handle_wheel_in_selection(self, event):
        """Handle wheel events in selection mode - temporarily enable pan for zoom"""
        # Temporarily enable mouse for zooming
        if not self.was_in_selection_before_wheel:
            self.was_in_selection_before_wheel = True
            self.plot.setMouseEnabled(x=True, y=True)
        
        # Reset or create timer
        if not self.wheel_timer:
            self.wheel_timer = QTimer()
            self.wheel_timer.setSingleShot(True)
            self.wheel_timer.timeout.connect(self._end_wheel_zoom)
        
        # Restart timer
        self.wheel_timer.start(1)
    
    def _end_wheel_zoom(self):
        """End wheel zoom and return to selection mode"""
        if self.was_in_selection_before_wheel and (self.selection_mode or self.zoom_mode):
            self.plot.setMouseEnabled(x=False, y=False)
            self.was_in_selection_before_wheel = False
    
    def _zoom_at_point(self, scene_pos, scale_factor):
        """Zoom in or out at a specific point"""
        # Convert scene position to view coordinates
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        
        # Get current view range
        view_range = self.plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        
        # Calculate current center and size
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        
        # Calculate new size
        new_x_size = x_size / scale_factor
        new_y_size = y_size / scale_factor
        
        # Calculate offset to zoom toward the point
        x_offset = (view_pos.x() - x_center) * (1 - 1/scale_factor)
        y_offset = (view_pos.y() - y_center) * (1 - 1/scale_factor)
        
        # Set new range
        new_x_center = x_center + x_offset
        new_y_center = y_center + y_offset
        
        self.plot.setXRange(new_x_center - new_x_size/2, new_x_center + new_x_size/2, padding=0)
        self.plot.setYRange(new_y_center - new_y_size/2, new_y_center + new_y_size/2, padding=0)
    
    def _reset_view(self):
        """Reset view to show entire graph"""
        if not self.node_positions:
            return
        
        nodes = list(self.node_positions.keys())
        if not nodes:
            return
        
        pos_array = np.array([self.node_positions[n] for n in nodes])
        
        # Get bounds
        x_min, y_min = pos_array.min(axis=0)
        x_max, y_max = pos_array.max(axis=0)
        
        # Add padding
        if self.shell or self.component:
            padding = 0.75
        else:
            padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.plot.setXRange(x_min - padding * x_range, 
                           x_max + padding * x_range, padding=0)
        self.plot.setYRange(y_min - padding * y_range, 
                           y_max + padding * y_range, padding=0)
    
    def _clear_graph(self):
        """Clear the graph visualization"""
        if self.load_thread is not None and self.load_thread.isRunning():
            self.load_thread.finished.disconnect()
            self.load_thread.terminate()  # Forcefully kill the thread
            self.load_thread.wait()       # Wait for it to fully terminate
            self.load_thread = None       # Clear the reference
        
        # Remove loading indicator if it exists
        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None
        
        # Clear scatter plot
        self.scatter.clear()
        
        # Clear edges
        for item in self.edge_items:
            self.plot.removeItem(item)
        self.edge_items.clear()
        
        # Force clear all labels - be aggressive
        # First clear from our tracking dict
        if hasattr(self, 'label_items') and self.label_items:
            for label_item in list(self.label_items.values()):
                try:
                    self.plot.removeItem(label_item)
                except:
                    pass
            self.label_items.clear()

        #Remove legend
        try:
            for i in reversed(range(self.legend_layout.count())): 
                self.legend_layout.itemAt(i).widget().setParent(None)
        except:
            pass

        # remove ALL TextItems from the plot
        # This catches any labels that might not be tracked properly
        items_to_remove = []
        for item in self.plot.items:
            if isinstance(item, pg.TextItem):
                items_to_remove.append(item)
        for item in items_to_remove:
            self.plot.removeItem(item)
        
        # Clear selection rectangle if exists
        if self.selection_rect:
            self.plot.removeItem(self.selection_rect)
            self.selection_rect = None
        
        # Clear data
        self.node_positions.clear()
        self.node_items.clear()
        self.selected_nodes.clear()
        self.rendered = False
        if hasattr(self, 'label_data'):
            self.label_data.clear()
        
        # Clear cache
        self.cached_spots.clear()
        self.cached_node_to_index.clear()
        self.cached_brushes.clear()
        self.last_selected_set.clear()
        self.cached_sizes_for_lod.clear()

        if self.graph is None or len(self.graph.nodes()) == 0:
            # Show loading indicator
            self.loading_text = pg.TextItem(
                text="No network detected",
                color=(100, 100, 100),
                anchor=(0.5, 0.5)
            )
        else:
            # Show loading indicator
            self.loading_text = pg.TextItem(
                text="Press 🔄 to load your graph",
                color=(100, 100, 100),
                anchor=(0.5, 0.5)
            )

        self.loading_text.setPos(0, 0)  # Center of view
        self.plot.addItem(self.loading_text)

    def _popout_graph(self):

        temp_graph_widget = NetworkGraphWidget(
            parent=self.parent_window,
            weight=self.weight,
            geometric=self.geometric,
            component = self.component,
            centroids=self.centroids,
            communities=self.communities,
            community_dict=self.community_dict,
            labels=self.labels,
            identities = self.identities,
            identity_dict = self.identity_dict,
            z_size = self.z_size,
            shell = self.shell,
            node_size = self.node_size,
            black_edges = self.black_edges,
            edge_size = self.edge_size
        )

        temp_graph_widget.set_graph(self.graph)
        temp_graph_widget.show_in_window(title="Network Graph", width=1000, height=800)
        temp_graph_widget.load_graph()
        self.parent_window.temp_graph_widgets.append(temp_graph_widget)

    
    def select_nodes(self, nodes, add_to_selection=False):
        """
        Programmatically select nodes.
        
        Parameters:
        -----------
        nodes : list
            List of node IDs to select
        add_to_selection : bool
            If True, add to existing selection. If False, replace selection.
        """
        if not add_to_selection:
            self.selected_nodes.clear()

        if nodes is None:
            self.selected_nodes = set()
        elif nodes == []:
            self.selected_nodes = set()
        else:     
            # Add valid nodes to selection
            for node in nodes:
                if node in self.node_items:
                    self.selected_nodes.add(node)
        
        # Update visual representation
        self._render_nodes()
        
        # Emit signal
        self.node_selected.emit(list(self.selected_nodes))
    
    def clear_selection(self):
        """Clear all selected nodes"""
        self.selected_nodes.clear()
        self._render_nodes()
        self.node_selected.emit([])
    
    def _on_node_clicked(self, scatter, points, ev):
        """Handle node click events"""
        if not self.selection_mode or len(points) == 0:
            return
        
        # Get clicked node
        point = points[0]
        clicked_node = point.data()
        
        # Check if Ctrl is pressed
        modifiers = ev.modifiers()
        ctrl_pressed = modifiers & Qt.KeyboardModifier.ControlModifier
        
        if ctrl_pressed:
            # Toggle selection for this node
            if clicked_node in self.selected_nodes:
                self.selected_nodes.remove(clicked_node)
            else:
                self.selected_nodes.add(clicked_node)
        else:
            # Clear previous selection and select only this node
            self.selected_nodes.clear()
            self.selected_nodes.add(clicked_node)

        self.push_selection()
        
        # Update visual representation
        self._render_nodes()
        self.node_click = True
        
        # Emit signal with all selected nodes
        self.node_selected.emit(list(self.selected_nodes))

    def push_selection(self):
        self.parent_window.clicked_values['nodes'] = list(self.selected_nodes)
        self.parent_window.evaluate_mini(subgraph_push = True)
        self.parent_window.handle_info('node')

    def get_selected_nodes(self):
        """Get the list of currently selected nodes"""
        return list(self.selected_nodes)
    
    def get_selected_node(self):
        """
        Get a single selected node (for backwards compatibility).
        Returns the first selected node or None.
        """
        if self.selected_nodes:
            return next(iter(self.selected_nodes))
        return None


    def handle_find_action(self):
        try:
            val = self.parent_window.clicked_values['nodes'][-1]
            self.parent_window.handle_info(sort = 'node')
            if val in self.centroids:
                centroid = self.centroids[val]
                self.parent_window.set_active_channel(0) 
                # Toggle on the nodes channel if it's not already visible
                if not self.parent_window.channel_visible[0]:
                    self.parent_window.channel_buttons[0].setChecked(True)
                    self.parent_window.toggle_channel(0)
                # Navigate to the Z-slice
                self.parent_window.slice_slider.setValue(int(centroid[0]))
                print(f"Found node {val} at [Z,Y,X] -> {centroid}")
                self.push_selection()
        except:
            import traceback
            traceback.print_exc()
            pass


    def save_table_as(self, file_type):
        """Save the table data as either CSV or Excel file."""
        
        if self != self.parent_window.selection_graph_widget:
            table_name = "Network"
            df = self.parent_window.network_table.model()._data
        else:
            df = self.parent_window.selection_table.model()._data
            table_name = "Selection"
        
        # Get save file name
        file_filter = ("CSV Files (*.csv)" if file_type == 'csv' else 
                      "Excel Files (*.xlsx)" if file_type == 'xlsx' else 
                      "Gephi Graph (*.gexf)" if file_type == 'gexf' else
                      "GraphML (*.graphml)" if file_type == 'graphml' else
                      "Pajek Network (*.net)")

        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {table_name} Table As",
            "",
            file_filter
        )

        if filename:
            try:
                if file_type == 'csv':
                    # If user didn't type extension, add .csv
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    df.to_csv(filename, index=False)
                elif file_type == 'xlsx':
                    # If user didn't type extension, add .xlsx
                    if not filename.endswith('.xlsx'):
                        filename += '.xlsx'
                    df.to_excel(filename, index=False)
                elif file_type == 'gexf':
                    # If user didn't type extension, add .gexf
                    if not filename.endswith('.gexf'):
                        filename += '.gexf'
                    #for node in my_network.network.nodes():
                        #my_network.network.nodes[node]['label'] = str(node)
                    nx.write_gexf(self.graph, filename, encoding='utf-8', prettyprint=True)
                elif file_type == 'graphml':
                    # If user didn't type extension, add .graphml
                    if not filename.endswith('.graphml'):
                        filename += '.graphml'
                    nx.write_graphml(self.graph, filename)
                elif file_type == 'net':
                    # If user didn't type extension, add .net
                    if not filename.endswith('.net'):
                        filename += '.net'
                    nx.write_pajek(self.graph, filename)
                    
                QMessageBox.information(
                    self,
                    "Success",
                    f"{table_name} table successfully saved to {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save file: {str(e)}"
                )


    def get_properties(self):

        self.parent_window.update_graph_fields()


    def create_context_menu(self, event):
        # Get the index at the clicked position
        # Create context menu
        context_menu = QMenu(self)
        
        find_action = context_menu.addAction("Find Node")

        find_action.triggered.connect(self.handle_find_action)
        neigh_action = context_menu.addAction("Show Neighbors")
        neigh_action.triggered.connect(self.parent_window.handle_show_neighbors)
        com_action = context_menu.addAction("Show Community")
        com_action.triggered.connect(self.parent_window.handle_show_communities)
        comp_action = context_menu.addAction("Show Connected Component")
        comp_action.triggered.connect(self.parent_window.handle_show_component)

        comp_action2 = context_menu.addAction("Isolate Connected Component")
        comp_action2.triggered.connect(lambda: self.parent_window.handle_show_component(isolate = True))

        # Add separator
        context_menu.addSeparator()
        
        # Add Save As menu
        save_menu = context_menu.addMenu("Save As")
        save_csv = save_menu.addAction("CSV")
        save_excel = save_menu.addAction("Excel")
        save_gephi = save_menu.addAction("Gephi")
        save_graphml = save_menu.addAction("GraphML")
        save_pajek = save_menu.addAction("Pajek")
        
        # Connect the actions - ensure we're saving the active table
        save_csv.triggered.connect(lambda: self.save_table_as('csv'))
        save_excel.triggered.connect(lambda: self.save_table_as('xlsx'))
        save_gephi.triggered.connect(lambda: self.save_table_as('gexf'))
        save_graphml.triggered.connect(lambda: self.save_table_as('graphml'))
        save_pajek.triggered.connect(lambda: self.save_table_as('net'))


        if self == self.parent_window.selection_graph_widget:
            set_action = context_menu.addAction("Swap with network table (also sets internal network properties - may affect related functions)")
            set_action.triggered.connect(self.parent_window.selection_table.set_selection_to_active)
        
        # Show the menu at cursor position
        view_widget = self.plot.getViewWidget()
        
        # Map scene position to view coordinates
        view_pos = view_widget.mapFromScene(event.scenePos())
        
        # Map to global screen coordinates
        global_pos = view_widget.mapToGlobal(view_pos)
        
        # Show the menu
        context_menu.exec(global_pos)
    
    def _on_view_changed(self):
        """Handle view range changes for level-of-detail adjustments"""
        if not self.node_positions or len(self.node_positions) == 0:
            return
        
        # Calculate current zoom factor based on view range
        view_range = self.plot.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]
        
        # Get initial full graph bounds
        nodes = list(self.node_positions.keys())
        pos_array = np.array([self.node_positions[n] for n in nodes])
        
        if len(pos_array) > 0:
            full_x_range = pos_array[:, 0].max() - pos_array[:, 0].min()
            full_y_range = pos_array[:, 1].max() - pos_array[:, 1].min()
            
            if full_x_range > 0 and full_y_range > 0:
                # Calculate zoom factor (smaller view range = more zoomed in)
                zoom_x = full_x_range / x_range if x_range > 0 else 1
                zoom_y = full_y_range / y_range if y_range > 0 else 1
                zoom_factor = max(zoom_x, zoom_y)
                
                # Update if zoom changed significantly (>10% change)
                zoom_changed = abs(zoom_factor - self.current_zoom_factor) / max(self.current_zoom_factor, 0.01) > 0.1
                if zoom_changed:
                    self.current_zoom_factor = zoom_factor
                    self._update_lod_rendering()
                else:
                    # Even if zoom didn't change, update labels for panning
                    # (viewport changed but zoom level stayed the same)
                    if self.labels:
                        self._update_labels_for_zoom()
    
    def _update_lod_rendering(self):
        """OPTIMIZED: Update rendering based on current zoom level using cached data"""
        if not self.cached_spots or not self.cached_sizes_for_lod:
            return
        
        # Adjust node sizes based on zoom
        if self.current_zoom_factor > 1.5:
            scale_factor = 1.0 + np.log10(self.current_zoom_factor) * 0.3
        else:
            scale_factor = 1.0
        
        # Update node sizes in cached spots
        for i, base_size in enumerate(self.cached_sizes_for_lod):
            self.cached_spots[i]['size'] = base_size * scale_factor
        
        # Update edge visibility based on zoom
        if self.current_zoom_factor < 0.5:
            edge_alpha = int(50 * self.current_zoom_factor)
        elif self.current_zoom_factor > 2:
            edge_alpha = min(150, int(100 + self.current_zoom_factor * 10))
        else:
            edge_alpha = 100

        if self.black_edges:
            edge_color = (0, 0, 0)
        else:
            edge_color = (150, 150, 150, edge_alpha)
        
        # Update edge rendering (batched edge items)
        if self.edge_items:
            for edge_item in self.edge_items:
                current_pen = edge_item.opts['pen']
                if current_pen is not None:
                    width = current_pen.widthF()
                    new_pen = pg.mkPen(color=edge_color, width=width)
                    edge_item.setPen(new_pen)
        
        # Update labels based on zoom level
        if self.labels:
            self._update_labels_for_zoom()
        
        # Re-render nodes with new sizes
        self.scatter.setData(spots=self.cached_spots)

    def show_in_window(self, title="Network Graph", width=1000, height=800):
        """Show the graph widget in a separate non-modal window"""
        from PyQt6.QtWidgets import QMainWindow
        
        # Create new window
        self.popup_window = QMainWindow()
        self.popup_window.setWindowTitle(title)
        self.popup_window.setGeometry(100, 100, width, height)
        self.popup_window.setCentralWidget(self)
        
        # Show non-modal
        self.popup_window.show()
        
        return self.popup_window


# Example usage
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import sys
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Network Graph Viewer")
            self.setGeometry(100, 100, 1000, 800)
            
            # Create a sample graph
            G = nx.karate_club_graph()
            
            # Add some weights
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.uniform(0.5, 5.0)
            
            # Create sample community detection
            communities = nx.community.greedy_modularity_communities(G)
            community_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_dict[node] = i
            
            # Create the widget
            self.graph_widget = NetworkGraphWidget(
                parent=self,
                weight=True,
                communities=True,
                community_dict=community_dict,
                labels=True  # Enable labels for testing
            )
            
            self.setCentralWidget(self.graph_widget)
            
            # Set and load the graph
            self.graph_widget.set_graph(G)
            self.graph_widget.load_graph()
            
            # Connect signal
            self.graph_widget.node_selected.connect(self.on_node_selected)
        
        def on_node_selected(self, nodes):
            if nodes:
                print(f"Selected nodes: {nodes}")
            else:
                print("No nodes selected")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())