import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
except:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

# Import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ThresholdPredictor:
    """
    Machine learning model to predict min/max threshold boundaries from histogram slopes and bounds.
    Supports both PyTorch (with GPU/MPS) and sklearn backends with cross-compatible saving.
    """
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (64, 32, 16), 
                 use_pytorch: Optional[bool] = None):
        """
        Initialize the threshold predictor.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            use_pytorch: Force PyTorch (True) or sklearn (False). If None, auto-detect.
        """
        self.hidden_layers = hidden_layers
        
        # Determine which backend to use
        if use_pytorch is None:
            self.use_pytorch = PYTORCH_AVAILABLE
        else:
            self.use_pytorch = use_pytorch and PYTORCH_AVAILABLE
            
        if self.use_pytorch and not PYTORCH_AVAILABLE:
            warnings.warn("PyTorch requested but GPU/MPS not available. Falling back to sklearn.")
            self.use_pytorch = False
            
        print(f"Using backend: {'PyTorch' if self.use_pytorch else 'sklearn'}")
        
        # Initialize scalers for normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model
        self.model = None
        self._initialize_model()

    def norm_slope(self, histo_dict, training = True):
        """Converts histogram data into distribution of slopes, which should more naturally describe the shape of the histogram's curve"""
        output_dict = {}
        for iden, items in histo_dict.items():
            peaks = items[0]
            buckets = items[1]
            if training:
                bounds = items[2]
                bounds[1] = min(bounds[1], buckets[-1]) #User assigning upper bound beyond the max bucket border is meaningless
            slopes = np.zeros(peaks.shape[0] - 1)
            bucket_bounds = np.zeros(peaks.shape[0] - 1)
            for i in range(peaks.shape[0] - 1):
                slopes[i] = (peaks[i + 1] - peaks[i])/(buckets[i + 1] - buckets[i])
                bucket_bounds[i] = buckets[i + 1]
            if training:
                output_dict[iden] = [slopes, bucket_bounds, bounds]
            else:
                output_dict[iden] = [slopes, bucket_bounds]
        return output_dict
        
    def _initialize_model(self):
        """Initialize the appropriate model backend."""
        if self.use_pytorch:
            self.device = torch.device("cuda" if torch.cuda.is_available() 
                                     else "mps" if torch.backends.mps.is_available() 
                                     else "cpu")
            # Model will be created dynamically once we know input size
            self.model = None
        else:
            # MLPRegressor with partial_fit support
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
                warm_start=True  # Allows incremental training
            )
    
    def _pytorch_to_sklearn_weights(self):
        """Convert PyTorch model weights to sklearn format."""
        if not self.use_pytorch or self.model is None:
            return None
        
        weights = []
        biases = []
        
        # Extract weights from PyTorch model
        state_dict = self.model.state_dict()
        layer_idx = 0
        
        while True:
            weight_key = f'{layer_idx * 3}.weight'  # Every 3rd layer (Linear, ReLU, Dropout)
            bias_key = f'{layer_idx * 3}.bias'
            
            if weight_key not in state_dict:
                break
            
            # PyTorch stores weights as [out_features, in_features], sklearn needs [in_features, out_features]
            weights.append(state_dict[weight_key].cpu().numpy().T)
            biases.append(state_dict[bias_key].cpu().numpy())
            layer_idx += 1
        
        return {'weights': weights, 'biases': biases}

    def _sklearn_to_pytorch_weights(self, sklearn_model):
        """Convert sklearn model weights to PyTorch format."""
        if sklearn_model is None or not hasattr(sklearn_model, 'coefs_'):
            return None
        
        state_dict = {}
        
        # Convert sklearn weights to PyTorch format
        for layer_idx, (weight, bias) in enumerate(zip(sklearn_model.coefs_, sklearn_model.intercepts_)):
            # sklearn stores as [in_features, out_features], PyTorch needs [out_features, in_features]
            state_dict[f'{layer_idx * 3}.weight'] = torch.FloatTensor(weight.T)
            state_dict[f'{layer_idx * 3}.bias'] = torch.FloatTensor(bias)
        
        return state_dict

    def _create_pytorch_model(self, input_size: int) -> Any:
        """Create PyTorch neural network."""
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 2))  # Output: [min, max]
        
        return nn.Sequential(*layers)
    
    def _prepare_data(self, data_dict: Dict[str, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from dictionary format.
        
        Args:
            data_dict: Dictionary with entries like:
                {'channel_name': [slopes_array, bounds_array, [min_threshold, max_threshold]]}
        
        Returns:
            X: Input features (concatenated slopes and bounds)
            y: Output targets (min/max thresholds)
        """


        data_dict = self.norm_slope(data_dict)

        X_list = []
        y_list = []
        
        for channel_name, data in data_dict.items():
            if len(data) != 3:
                raise ValueError(f"Channel {channel_name} must have 3 elements: [slopes, bounds, thresholds]")
                
            slopes = np.array(data[0], dtype=np.float32)
            bounds = np.array(data[1], dtype=np.float32)
            thresholds = np.array(data[2], dtype=np.float32)
            
            if len(thresholds) != 2:
                raise ValueError(f"Thresholds for {channel_name} must be a 2-element list [min, max]")
            
            # Ensure slopes and bounds have same length
            if len(slopes) != len(bounds):
                raise ValueError(f"Slopes and bounds must have same length for {channel_name}")
            
            # Concatenate slopes and bounds as features
            features = np.concatenate([slopes, bounds])
            
            X_list.append(features)
            y_list.append(thresholds)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        return X, y
    
    def train(self, training_data: Dict[str, List], epochs: int = 100, 
              batch_size: int = 32, learning_rate: float = 0.001,
              from_scratch: bool = False, save_path: Optional[str] = None):
        """
        Train the model on provided data.
        
        Args:
            training_data: Dictionary of training samples
            epochs: Number of training epochs (PyTorch only)
            batch_size: Batch size for training (PyTorch only)
            learning_rate: Learning rate
            from_scratch: If True, reinitialize model; if False, continue training
            save_path: Path to save model after training (auto-saves by default)
        """
        print(f"Training {'from scratch' if from_scratch else 'incrementally'}...")
        
        # Prepare data
        X, y = self._prepare_data(training_data)
        print(f"Training on {len(X)} samples with {X.shape[1]} features each")
        
        # Fit or update scalers
        if not self.is_fitted or from_scratch:
            self.input_scaler.fit(X)
            self.output_scaler.fit(y)
            self.is_fitted = True
        
        # Normalize data
        X_scaled = self.input_scaler.transform(X)
        y_scaled = self.output_scaler.transform(y)
        
        if self.use_pytorch:
            self._train_pytorch(X_scaled, y_scaled, epochs, batch_size, 
                              learning_rate, from_scratch)
        else:
            self._train_sklearn(X_scaled, y_scaled, from_scratch)
        
        # Auto-save after training
        if save_path is None:
            save_path = "threshold_model.pkl"
        self.save(save_path)
        print(f"Model saved to {save_path}")
    
    def _train_pytorch(self, X: np.ndarray, y: np.ndarray, epochs: int,
                       batch_size: int, learning_rate: float, from_scratch: bool):
        """Train using PyTorch."""
        # Create model if needed
        if self.model is None or from_scratch:
            input_size = X.shape[1]
            self.model = self._create_pytorch_model(input_size).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Final loss: {best_loss:.6f}")
    
    def _train_sklearn(self, X: np.ndarray, y: np.ndarray, from_scratch: bool):
        """Train using sklearn."""
        # For small datasets, adjust parameters to avoid validation errors
        n_samples = X.shape[0]
        
        if from_scratch or not hasattr(self.model, 'coefs_'):
            # Reinitialize model with appropriate settings for dataset size
            if n_samples < 10:
                # Small dataset: disable early stopping
                self.model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=2000,
                    early_stopping=False,  # Disable for small datasets
                    random_state=42,
                    warm_start=True
                )
            else:
                # Larger dataset: use early stopping
                self.model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                    warm_start=True
                )
            # Initial fit
            self.model.fit(X, y)
        else:
            # Incremental training using warm_start
            self.model.fit(X, y)
        
        score = self.model.score(X, y)
        print(f"Training complete. R² score: {score:.4f}")
    
    def predict(self, input_data: Dict[str, List]) -> Dict[str, List[float]]:
        """
        Predict thresholds for new data.
        
        Args:
            input_data: Dictionary with entries like:
                {'channel_name': [slopes_array, bounds_array]}
        
        Returns:
            Dictionary with predictions:
                {'channel_name': [min_threshold, max_threshold]}
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        input_data = self.norm_slope(input_data, training = False)
        
        predictions = {}
        
        for channel_name, data in input_data.items():
            if len(data) != 2:
                raise ValueError(f"Input for {channel_name} must have 2 elements: [slopes, bounds]")
            
            slopes = np.array(data[0], dtype=np.float32)
            bounds = np.array(data[1], dtype=np.float32)
            
            # Concatenate features
            features = np.concatenate([slopes, bounds]).reshape(1, -1)
            
            # Normalize
            features_scaled = self.input_scaler.transform(features)
            
            # Predict
            if self.use_pytorch:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(features_scaled).to(self.device)
                    y_pred_scaled = self.model(X_tensor).cpu().numpy()
            else:
                y_pred_scaled = self.model.predict(features_scaled)
            
            # Denormalize
            y_pred = self.output_scaler.inverse_transform(y_pred_scaled)
            
            predictions[channel_name] = y_pred[0].tolist()
        
        return predictions
    
    def save(self, filepath: str):
        """
        Save model to file in a format compatible with both backends.
        Stores weights in both formats for cross-compatibility.
        
        Args:
            filepath: Path to save the model
        """
        save_data = {
            'trained_with': 'pytorch' if self.use_pytorch else 'sklearn',
            'hidden_layers': self.hidden_layers,
            'is_fitted': self.is_fitted,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
        }
        
        if self.use_pytorch:
            # Save PyTorch model state
            if self.model is not None:
                save_data['pytorch_state'] = self.model.state_dict()
                save_data['input_size'] = list(self.model.parameters())[0].shape[1]
                # Also save in neutral format for sklearn conversion
                save_data['weights_neutral'] = self._pytorch_to_sklearn_weights()
        else:
            # Save sklearn model
            save_data['sklearn_model'] = self.model
            # Weights are accessible from sklearn_model.coefs_ and .intercepts_
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved successfully to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, force_backend: Optional[str] = None) -> 'ThresholdPredictor':
        """
        Load model from file with automatic weight conversion between backends.
        Can load PyTorch-trained models with sklearn and vice versa.
        
        Args:
            filepath: Path to the saved model
            force_backend: Force 'pytorch' or 'sklearn' backend, or None for auto
        
        Returns:
            Loaded ThresholdPredictor instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Determine which backend to use
        trained_with = save_data.get('trained_with', save_data.get('use_pytorch', False))
        if isinstance(trained_with, bool):
            trained_with = 'pytorch' if trained_with else 'sklearn'
        
        if force_backend == 'pytorch':
            use_pytorch = True
        elif force_backend == 'sklearn':
            use_pytorch = False
        else:
            # Auto-detect: use PyTorch if available, otherwise sklearn
            use_pytorch = PYTORCH_AVAILABLE
        
        print(f"Model was trained with: {trained_with}")
        print(f"Loading with backend: {'PyTorch' if use_pytorch else 'sklearn'}")
        
        # Create instance
        predictor = cls(
            hidden_layers=save_data['hidden_layers'],
            use_pytorch=use_pytorch
        )
        
        # Restore scalers
        predictor.input_scaler = save_data['input_scaler']
        predictor.output_scaler = save_data['output_scaler']
        predictor.is_fitted = save_data['is_fitted']
        
        # Load model with potential weight conversion
        if use_pytorch:
            # Loading with PyTorch
            if 'pytorch_state' in save_data:
                # Direct PyTorch load
                input_size = save_data['input_size']
                predictor.model = predictor._create_pytorch_model(input_size).to(predictor.device)
                predictor.model.load_state_dict(save_data['pytorch_state'])
                print("✓ Loaded PyTorch weights directly")
            elif 'sklearn_model' in save_data:
                # Convert sklearn → PyTorch
                sklearn_model = save_data['sklearn_model']
                if hasattr(sklearn_model, 'coefs_'):
                    input_size = sklearn_model.coefs_[0].shape[0]
                    predictor.model = predictor._create_pytorch_model(input_size).to(predictor.device)
                    state_dict = predictor._sklearn_to_pytorch_weights(sklearn_model)
                    predictor.model.load_state_dict(state_dict)
                    print("✓ Converted sklearn weights to PyTorch")
                else:
                    warnings.warn("sklearn model not trained. Model needs retraining.")
            else:
                warnings.warn("No model weights found. Model needs retraining.")
        else:
            # Loading with sklearn
            if 'sklearn_model' in save_data:
                # Direct sklearn load (backward compatibility)
                predictor.model = save_data['sklearn_model']
                print("✓ Loaded sklearn model directly")
            elif 'model' in save_data:
                # Backward compatibility with old save format
                predictor.model = save_data['model']
                print("✓ Loaded sklearn model directly (legacy format)")
            elif 'weights_neutral' in save_data:
                # Convert PyTorch → sklearn using neutral weights
                weights_data = save_data['weights_neutral']
                input_size = weights_data['weights'][0].shape[0]
                
                # Create sklearn model
                n_samples = 10  # Dummy value for initialization
                if n_samples < 10:
                    predictor.model = MLPRegressor(
                        hidden_layer_sizes=predictor.hidden_layers,
                        activation='relu',
                        solver='adam',
                        max_iter=1,  # Just initialize
                        early_stopping=False,
                        random_state=42,
                        warm_start=True
                    )
                else:
                    predictor.model = MLPRegressor(
                        hidden_layer_sizes=predictor.hidden_layers,
                        activation='relu',
                        solver='adam',
                        max_iter=1,
                        early_stopping=False,
                        random_state=42,
                        warm_start=True
                    )
                
                # Initialize with dummy data to create weight structure
                X_dummy = np.zeros((2, input_size), dtype=np.float32)
                y_dummy = np.zeros((2, 2), dtype=np.float32)
                predictor.model.fit(X_dummy, y_dummy)
                
                # Inject converted weights
                predictor.model.coefs_ = weights_data['weights']
                predictor.model.intercepts_ = weights_data['biases']
                print("✓ Converted PyTorch weights to sklearn")
            elif 'pytorch_state' in save_data:
                # Try direct conversion from PyTorch state_dict
                warnings.warn("Converting PyTorch to sklearn without neutral format. This may fail.")
                input_size = save_data['input_size']
                
                # Create and initialize sklearn model
                predictor.model = MLPRegressor(
                    hidden_layer_sizes=predictor.hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=1,
                    early_stopping=False,
                    random_state=42,
                    warm_start=True
                )
                X_dummy = np.zeros((2, input_size), dtype=np.float32)
                y_dummy = np.zeros((2, 2), dtype=np.float32)
                predictor.model.fit(X_dummy, y_dummy)
                
                # Extract and convert PyTorch weights
                state_dict = save_data['pytorch_state']
                weights = []
                biases = []
                layer_idx = 0
                
                while True:
                    weight_key = f'{layer_idx * 3}.weight'
                    bias_key = f'{layer_idx * 3}.bias'
                    
                    if weight_key not in state_dict:
                        break
                    
                    weights.append(state_dict[weight_key].cpu().numpy().T)
                    biases.append(state_dict[bias_key].cpu().numpy())
                    layer_idx += 1
                
                predictor.model.coefs_ = weights
                predictor.model.intercepts_ = biases
                print("✓ Converted PyTorch weights to sklearn")
            else:
                warnings.warn("No model weights found. Model needs retraining.")
        
        print(f"Model loaded successfully from {filepath}")
        return predictor



# Example usage and testing
if __name__ == "__main__":
    # Example training data (based on provided format)
    example_data = {
        'AQP1.tif': [
            np.array([8.21, 6.21, -9.78, -4.84, -1.25]),  # slopes
            np.array([779.14, 1558.28, 2337.42, 3116.57, 3895.71]),  # bounds
            [3194.48, 38957.07]  # thresholds [min, max]
        ],
        'CD3.tif': [
            np.array([-69.71, -0.27, -0.06, 0.01, -0.005]),
            np.array([404.58, 809.17, 1213.75, 1618.33, 2022.91]),
            [452.87, 20229.14]
        ],
        'CD31.tif': [
            np.array([-13.04, -0.94, -0.26, -0.16, -0.08]),
            np.array([1231.26, 2462.52, 3693.78, 4925.04, 6156.30]),
            [1727.74, 61563.0]
        ]
    }
    
    print("=" * 60)
    print("Threshold Predictor - Example Usage")
    print("=" * 60)
    
    # Initialize and train
    predictor = ThresholdPredictor(hidden_layers=(64, 32, 16))
    predictor.train(example_data, epochs=100, from_scratch=True, save_path="model_example.pkl")
    
    print("\n" + "=" * 60)
    print("Testing Prediction")
    print("=" * 60)
    
    # Test prediction
    test_input = {
        'test_channel': [
            np.array([5.0, 3.0, -8.0, -5.0, -2.0]),
            np.array([800.0, 1600.0, 2400.0, 3200.0, 4000.0])
        ]
    }
    
    predictions = predictor.predict(test_input)
    print(f"\nPredictions: {predictions}")
    
    print("\n" + "=" * 60)
    print("Testing Save/Load")
    print("=" * 60)
    
    # Load model
    loaded_predictor = ThresholdPredictor.load("model_example.pkl")
    
    # Verify prediction consistency
    loaded_predictions = loaded_predictor.predict(test_input)
    print(f"Loaded model predictions: {loaded_predictions}")
    
    print("\n" + "=" * 60)
    print("Testing Incremental Training")
    print("=" * 60)
    
    # Add more training data
    additional_data = {
        'CD68.tif': [
            np.array([-3.90, -3.74, -1.61, -0.69, -0.26]),
            np.array([1268.32, 2536.64, 3804.95, 5073.27, 6341.59]),
            [3399.91, 63415.9]
        ]
    }
    
    loaded_predictor.train(additional_data, epochs=50, from_scratch=False)
    print("\nIncremental training complete!")
