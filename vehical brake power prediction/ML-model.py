import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class MLBrakeController:
    """Machine Learning based brake controller with neural network"""
    
    def __init__(self, use_pretrained=False):
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if use_pretrained and os.path.exists("brake_model.joblib"):
            try:
                # Load pretrained model
                self.model = joblib.load("brake_model.joblib")
                self.is_trained = True
                print("Loaded pretrained ML model")
                
                # We need to load or generate training data for the scaler
                self.generate_training_data()
                # Fit scaler with training data
                self.scaler.fit(self.X_train)
                
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Creating new model...")
                self.create_new_model()
        else:
            self.create_new_model()
    
    def create_new_model(self):
        """Create and train a new model"""
        # Initialize neural network
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            learning_rate_init=0.001
        )
        self.is_trained = False
        self.generate_training_data()
        self.train_model()
    
    def generate_training_data(self):
        """Generate training data based on physics and safety rules"""
        np.random.seed(42)
        
        # Generate varied scenarios
        n_samples = 5000
        
        # Input features: [distance, our_speed, object_speed, object_type_encoded]
        self.X = np.zeros((n_samples, 4))
        self.y = np.zeros(n_samples)  # Brake pressure (0-1)
        
        for i in range(n_samples):
            # Random parameters within realistic ranges
            distance = np.random.uniform(1, 200)
            our_speed = np.random.uniform(0, 40)  # m/s (0-144 km/h)
            object_speed = np.random.uniform(0, 40)
            object_type = np.random.choice([0, 1, 2])  # 0=car, 1=pedestrian, 2=curb
            
            # Calculate physics-based target (similar to original rules)
            closing_speed = our_speed - object_speed
            
            if closing_speed > 0.1:
                ttc = distance / closing_speed
            else:
                ttc = float('inf')
            
            # Base brake from TTC (similar to original rules)
            if ttc < 1.0:
                brake = 0.8
            elif ttc < 2.0:
                brake = 0.5
            elif ttc < 3.0:
                brake = 0.3
            elif ttc < 4.0:
                brake = 0.1
            else:
                brake = 0.0
            
            # Object type adjustments (with some randomness for learning)
            if object_type == 1:  # pedestrian
                brake = min(brake * 1.2 + np.random.uniform(0, 0.1), 0.8)
            elif object_type == 2:  # curb
                brake = brake * 0.7 + np.random.uniform(-0.05, 0.05)
            
            # Add some noise for more realistic training
            brake += np.random.uniform(-0.05, 0.05)
            brake = np.clip(brake, 0.0, 0.8)
            
            # Store training sample
            self.X[i] = [distance, our_speed, object_speed, object_type]
            self.y[i] = brake
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def train_model(self):
        """Train the neural network"""
        print("Training ML model...")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        train_score = self.model.score(self.X_train_scaled, self.y_train)
        test_score = self.model.score(self.X_test_scaled, self.y_test)
        print(f"Training R² score: {train_score:.3f}")
        print(f"Testing R² score: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save model
        joblib.dump(self.model, "brake_model.joblib")
        print("Model saved as brake_model.joblib")
    
    def predict_brake(self, distance, our_speed, object_speed, object_type):
        """Predict brake pressure using ML model"""
        if not self.is_trained:
            return 0.0, "MODEL NOT TRAINED", float('inf'), distance
        
        # Encode object type
        type_mapping = {"car": 0, "pedestrian": 1, "curb": 2}
        obj_encoded = type_mapping.get(object_type.lower(), 0)
        
        # Prepare input
        features = np.array([[distance, our_speed, object_speed, obj_encoded]])
        
        try:
            features_scaled = self.scaler.transform(features)
        except Exception as e:
            print(f"Scaler error: {e}")
            # If scaler not fitted, fit it now
            self.scaler.fit(self.X_train)
            features_scaled = self.scaler.transform(features)
        
        # Predict
        brake = self.model.predict(features_scaled)[0]
        brake = np.clip(brake, 0.0, 0.8)
        
        # Calculate TTC for display (though model doesn't use it directly)
        closing_speed = our_speed - object_speed
        ttc = distance / closing_speed if closing_speed > 0.1 else float('inf')
        
        # Determine action based on predicted brake
        if brake >= 0.6:
            action = "EMERGENCY BRAKE"
        elif brake >= 0.4:
            action = "HARD BRAKE"
        elif brake >= 0.2:
            action = "MODERATE BRAKE"
        elif brake >= 0.05:
            action = "GENTLE BRAKE"
        else:
            action = "SAFE"
        
        # Add object type info
        if object_type.lower() == "pedestrian":
            action += " (PEDESTRIAN)"
        elif object_type.lower() == "curb":
            action += " (CURB)"
        
        return brake, action, ttc, distance