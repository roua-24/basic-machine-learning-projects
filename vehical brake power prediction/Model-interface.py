import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Configure appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

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

class RuleBasedController:
    """Rule-based brake controller for comparison"""
    
    def __init__(self):
        self.MAX_BRAKE_PRESSURE = 0.8
    
    def calculate_brake(self, distance, our_speed, object_speed, object_type):
        """Calculate brake pressure using rule-based logic"""
        closing_speed = our_speed - object_speed
        
        if closing_speed > 0.1:
            ttc = distance / closing_speed
        else:
            ttc = float('inf')
            return 0.0, "SAFE", ttc, distance
        
        if ttc < 1.0:
            brake = 0.8
            action = "EMERGENCY"
        elif ttc < 2.0:
            brake = 0.5
            action = "HARD BRAKE"
        elif ttc < 3.0:
            brake = 0.3
            action = "MODERATE"
        elif ttc < 4.0:
            brake = 0.1
            action = "GENTLE"
        else:
            return 0.0, "SAFE", ttc, distance
        
        if object_type == "pedestrian":
            brake = min(brake * 1.2, self.MAX_BRAKE_PRESSURE)
            action += " (PEDESTRIAN)"
        elif object_type == "curb":
            brake *= 0.7
            action += " (CURB)"
        
        brake = min(brake, self.MAX_BRAKE_PRESSURE)
        return brake, action, ttc, distance

class MLBrakeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Braking System Simulator")
        self.root.geometry("1400x850")
        
        # Variables
        self.distance_var = ctk.DoubleVar(value=50.0)
        self.our_speed_var = ctk.DoubleVar(value=72.0)
        self.object_speed_var = ctk.DoubleVar(value=0.0)
        self.object_type_var = ctk.StringVar(value="car")
        self.use_ml_var = ctk.BooleanVar(value=True)
        
        self.is_animating = False
        self.history = []  # Store history for learning
        
        # Initialize controllers (with error handling)
        try:
            self.ml_controller = MLBrakeController(use_pretrained=True)
            print("ML controller initialized successfully")
        except Exception as e:
            print(f"Error initializing ML controller: {e}")
            # Create a simple fallback
            self.ml_controller = None
        
        self.rule_controller = RuleBasedController()
        
        self.setup_ui()
        self.update_simulation()  # Initial update
    
    def setup_ui(self):
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Left Panel - Controls
        self.create_control_panel()
        
        # Right Panel - Visualizations
        self.create_visualization_panel()
    
    def create_control_panel(self):
        control_panel = ctk.CTkFrame(self.root, width=320)
        control_panel.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        
        # Title
        title_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        title_frame.pack(pady=(20, 30), padx=20)
        
        ctk.CTkLabel(title_frame, text="ML BRAKING SYSTEM", 
                    font=("Arial", 22, "bold")).pack()
        ctk.CTkLabel(title_frame, text="Neural Network Controller", 
                    font=("Arial", 12), text_color="gray").pack(pady=(5, 0))
        
        # ML Toggle
        ml_toggle_frame = ctk.CTkFrame(control_panel, fg_color="gray95", corner_radius=8)
        ml_toggle_frame.pack(padx=20, pady=(0, 20), fill="x")
        
        ctk.CTkSwitch(ml_toggle_frame, text="Use Machine Learning", 
                     variable=self.use_ml_var, command=self.update_simulation,
                     font=("Arial", 12, "bold")).pack(pady=12)
        
        # Control Section
        controls_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        controls_frame.pack(padx=20, fill="x")
        
        # Distance Control
        self.create_ml_slider(controls_frame, "Distance to Object (m)", 
                             self.distance_var, 1, 200)
        
        # Speed Controls
        self.create_ml_slider(controls_frame, "Our Car Speed (km/h)", 
                             self.our_speed_var, 0, 144)
        
        self.create_ml_slider(controls_frame, "Object Speed (km/h)", 
                             self.object_speed_var, 0, 144)
        
        # Object Type
        type_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        type_frame.pack(fill="x", pady=(20, 10))
        
        ctk.CTkLabel(type_frame, text="Object Type:", 
                    font=("Arial", 12)).pack(anchor="w")
        
        self.object_type_combo = ctk.CTkSegmentedButton(type_frame,
                                                       values=["Car", "Pedestrian", "Curb"],
                                                       variable=self.object_type_var,
                                                       command=self.update_simulation)
        self.object_type_combo.pack(fill="x", pady=(8, 0))
        self.object_type_combo.set("Car")
        
        # Buttons
        button_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(20, 0))
        
        self.simulate_btn = ctk.CTkButton(button_frame, text="Simulate Scenario",
                                         command=self.animate_scenario,
                                         height=40,
                                         font=("Arial", 12))
        self.simulate_btn.pack(fill="x", pady=(0, 10))
        
        ctk.CTkButton(button_frame, text="Retrain Model",
                     command=self.retrain_model,
                     height=40,
                     font=("Arial", 12),
                     fg_color="#4CAF50",
                     hover_color="#45a049").pack(fill="x", pady=(0, 10))
        
        ctk.CTkButton(button_frame, text="Reset Values",
                     command=self.reset_values,
                     height=40,
                     font=("Arial", 12),
                     fg_color="gray",
                     hover_color="darkgray").pack(fill="x")
        
        # ML Info
        info_frame = ctk.CTkFrame(control_panel, fg_color="gray95", corner_radius=8)
        info_frame.pack(padx=20, pady=(30, 20), fill="x")
        
        ctk.CTkLabel(info_frame, text="Neural Network Info", 
                    font=("Arial", 12, "bold")).pack(pady=(12, 8))
        
        info_text = "• Layers: 64-32-16 neurons\n• Activation: ReLU\n• Optimizer: Adam\n• Training samples: 5,000\n• Input features: 4\n• Output: Brake pressure"
        ctk.CTkLabel(info_frame, text=info_text, 
                    font=("Arial", 11), justify="left").pack(pady=(0, 12), padx=12)
    
    def create_ml_slider(self, parent, label, variable, from_, to):
        """Create slider with value display"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 20))
        
        # Label
        label_frame = ctk.CTkFrame(frame, fg_color="transparent")
        label_frame.pack(fill="x")
        
        ctk.CTkLabel(label_frame, text=label, 
                    font=("Arial", 11)).pack(side="left")
        
        value_label = ctk.CTkLabel(label_frame, text=f"{variable.get():.1f}", 
                                  font=("Arial", 11, "bold"))
        value_label.pack(side="right")
        
        # Slider
        slider = ctk.CTkSlider(frame, from_=from_, to=to, variable=variable,
                              command=lambda v: self.on_slider_change(v, value_label))
        slider.pack(fill="x", pady=(8, 0))
        
        # Min/Max labels
        minmax_frame = ctk.CTkFrame(frame, fg_color="transparent")
        minmax_frame.pack(fill="x")
        
        ctk.CTkLabel(minmax_frame, text=f"{from_}", 
                    font=("Arial", 9), text_color="gray").pack(side="left")
        ctk.CTkLabel(minmax_frame, text=f"{to}", 
                    font=("Arial", 9), text_color="gray").pack(side="right")
        
        return slider
    
    def create_visualization_panel(self):
        viz_panel = ctk.CTkFrame(self.root)
        viz_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        viz_panel.grid_rowconfigure(2, weight=1)
        viz_panel.grid_columnconfigure(0, weight=1)
        
        # Header with mode
        header = ctk.CTkFrame(viz_panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(header, text="ML Braking Simulation", 
                    font=("Arial", 20, "bold")).grid(row=0, column=0, sticky="w")
        
        self.mode_label = ctk.CTkLabel(header, text="ML Mode: ACTIVE", 
                                      font=("Arial", 12),
                                      text_color="#2196F3")
        self.mode_label.grid(row=0, column=1, sticky="e")
        
        # Comparison Metrics
        metrics_frame = ctk.CTkFrame(viz_panel, fg_color="transparent")
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Create metric boxes
        self.ml_brake_box = self.create_comparison_box(metrics_frame, 0, "ML Brake", "%", "#2196F3")
        self.rule_brake_box = self.create_comparison_box(metrics_frame, 1, "Rule Brake", "%", "#FF9800")
        self.ttc_box = self.create_comparison_box(metrics_frame, 2, "Time to Collision", "s", "#4CAF50")
        self.action_box = self.create_comparison_box(metrics_frame, 3, "Decision", "", "#F44336")
        
        # Graphs
        graph_frame = ctk.CTkFrame(viz_panel)
        graph_frame.grid(row=2, column=0, sticky="nsew")
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure
        plt.style.use('default')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(9, 7))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Preset Scenarios
        preset_frame = ctk.CTkFrame(viz_panel, fg_color="transparent")
        preset_frame.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        
        ctk.CTkLabel(preset_frame, text="Test Scenarios:", 
                    font=("Arial", 12)).pack(anchor="w", pady=(0, 10))
        
        scenarios = [
            ("Emergency Stop", 5.0, 108.0, 0.0, "Car"),
            ("Pedestrian", 15.0, 36.0, 0.0, "Pedestrian"),
            ("Cutting In", 20.0, 90.0, 54.0, "Car"),
            ("Curb Avoid", 8.0, 18.0, 0.0, "Curb"),
            ("Highway", 50.0, 120.0, 118.0, "Car"),
        ]
        
        for i, (name, dist, our, obj, obj_type) in enumerate(scenarios):
            btn = ctk.CTkButton(preset_frame, text=name,
                               command=lambda d=dist, o=our, b=obj, t=obj_type: 
                               self.load_preset(d, o, b, t),
                               height=32,
                               font=("Arial", 11),
                               fg_color="gray",
                               hover_color="darkgray")
            btn.pack(side="left", padx=(0, 10) if i < len(scenarios)-1 else 0)
    
    def create_comparison_box(self, parent, column, title, unit, color):
        """Create comparison metric box"""
        box = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color="lightgray")
        box.grid(row=0, column=column, padx=(0, 10) if column < 3 else 0, sticky="nsew")
        
        # Title
        ctk.CTkLabel(box, text=title, 
                    font=("Arial", 11, "bold")).pack(pady=(12, 8))
        
        # Value
        value_label = ctk.CTkLabel(box, text="0", 
                                  font=("Arial", 22, "bold"),
                                  text_color=color)
        value_label.pack()
        
        # Unit
        if unit:
            ctk.CTkLabel(box, text=unit, 
                        font=("Arial", 11), text_color="gray").pack(pady=(0, 12))
        else:
            ctk.CTkLabel(box, text=" ", 
                        font=("Arial", 11)).pack(pady=(0, 12))
        
        return value_label
    
    def on_slider_change(self, value, label):
        label.configure(text=f"{float(value):.1f}")
        self.update_simulation()
    
    def update_simulation(self):
        """Update all displays with ML predictions"""
        try:
            # Get values
            true_distance = self.distance_var.get()
            our_speed_kmh = self.our_speed_var.get()
            object_speed_kmh = self.object_speed_var.get()
            our_speed_mps = our_speed_kmh / 3.6
            object_speed_mps = object_speed_kmh / 3.6
            object_type = self.object_type_var.get().lower()
            
            # Calculate TTC
            closing_speed = our_speed_mps - object_speed_mps
            ttc = true_distance / closing_speed if closing_speed > 0.1 else float('inf')
            
            # Update mode label
            if self.use_ml_var.get():
                self.mode_label.configure(text="ML Mode: ACTIVE", text_color="#2196F3")
                mode_text = "ML"
            else:
                self.mode_label.configure(text="Rule-based Mode", text_color="#FF9800")
                mode_text = "Rule-based"
            
            # Check if ML controller is available
            if self.ml_controller is None or not self.ml_controller.is_trained:
                # Fallback to rule-based only
                brake, action, ttc, _ = self.rule_controller.calculate_brake(
                    true_distance, our_speed_mps, object_speed_mps, object_type
                )
                ml_brake = brake
                rule_brake = brake
                ml_action = action
                rule_action = action
                self.use_ml_var.set(False)  # Force rule-based mode
            else:
                # Calculate using ML or rules
                if self.use_ml_var.get():
                    brake, action, ttc, _ = self.ml_controller.predict_brake(
                        true_distance, our_speed_mps, object_speed_mps, object_type
                    )
                    ml_brake = brake
                    ml_action = action
                    
                    # Calculate rule-based for comparison
                    rule_brake, rule_action, _, _ = self.rule_controller.calculate_brake(
                        true_distance, our_speed_mps, object_speed_mps, object_type
                    )
                else:
                    brake, action, ttc, _ = self.rule_controller.calculate_brake(
                        true_distance, our_speed_mps, object_speed_mps, object_type
                    )
                    rule_brake = brake
                    rule_action = action
                    
                    # Calculate ML for comparison
                    ml_brake, ml_action, _, _ = self.ml_controller.predict_brake(
                        true_distance, our_speed_mps, object_speed_mps, object_type
                    )
            
            # Update metric boxes
            ttc_text = f"{ttc:.1f}" if ttc != float('inf') else "∞"
            self.ttc_box.configure(text=ttc_text)
            
            self.ml_brake_box.configure(text=f"{ml_brake*100:.0f}")
            self.rule_brake_box.configure(text=f"{rule_brake*100:.0f}")
            
            # Show current system's action
            if self.use_ml_var.get():
                self.action_box.configure(text=ml_action[:20] if len(ml_action) > 20 else ml_action)
            else:
                self.action_box.configure(text=rule_action[:20] if len(rule_action) > 20 else rule_action)
            
            # Store in history for learning
            self.history.append({
                'distance': true_distance,
                'our_speed': our_speed_mps,
                'object_speed': object_speed_mps,
                'object_type': object_type,
                'ml_brake': ml_brake,
                'rule_brake': rule_brake,
                'ttc': ttc
            })
            
            # Keep only recent history
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            # Update graphs
            self.update_graphs(true_distance, true_distance, our_speed_mps,
                             object_speed_mps, ttc, ml_brake, rule_brake,
                             0.0, self.use_ml_var.get())
            
        except Exception as e:
            print(f"Error in update_simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def update_graphs(self, true_distance, measured_distance, our_speed_mps,
                     object_speed_mps, ttc, ml_brake, rule_brake,
                     lidar_error, use_ml):
        """Update all graphs with ML comparisons"""
        try:
            # Clear axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            closing_speed = our_speed_mps - object_speed_mps
            
            # Graph 1: ML vs Rule-based Comparison
            methods = ['ML System', 'Rule-based']
            brakes = [ml_brake * 100, rule_brake * 100]
            colors = ['#2196F3', '#FF9800']
            
            bars = self.ax1.bar(methods, brakes, color=colors, width=0.6)
            self.ax1.set_ylabel('Brake Pressure (%)')
            self.ax1.set_ylim(0, 100)
            
            if use_ml:
                self.ax1.set_title('ML vs Rule-based Comparison', fontsize=12, fontweight='bold')
            else:
                self.ax1.set_title('Rule-based vs ML Comparison', fontsize=12, fontweight='bold')
            
            # Add values on bars
            for bar, brake in zip(bars, brakes):
                self.ax1.text(bar.get_x() + bar.get_width()/2, brake + 2,
                             f'{brake:.0f}%', ha='center', va='bottom', fontsize=11)
            
            # Graph 2: Speed Analysis
            speeds = [our_speed_mps, object_speed_mps]
            speed_labels = ['Our Car', 'Object']
            speed_colors = ['#2196F3', '#FF9800']
            
            bars3 = self.ax2.bar(speed_labels, speeds, color=speed_colors, width=0.6)
            self.ax2.set_ylabel('Speed (m/s)')
            self.ax2.set_title('Speed Profile', fontsize=12, fontweight='bold')
            
            for bar, speed in zip(bars3, speeds):
                self.ax2.text(bar.get_x() + bar.get_width()/2, speed + 0.5,
                             f'{speed:.1f} m/s\n{speed*3.6:.0f} km/h',
                             ha='center', va='bottom', fontsize=9)
            
            # Graph 3: Brake Pressure Components
            components = ['Emergency\nThreshold', 'Current\nBrake', 'Maximum\nBrake']
            values = [60, min(ml_brake * 100, 100), 80]
            colors3 = ['#FF5252', '#4CAF50', '#2196F3']
            
            bars3 = self.ax3.bar(components, values, color=colors3, width=0.6)
            self.ax3.set_ylabel('Brake Pressure (%)')
            self.ax3.set_ylim(0, 100)
            self.ax3.set_title('Brake Pressure Analysis', fontsize=12, fontweight='bold')
            
            for bar, value in zip(bars3, values):
                self.ax3.text(bar.get_x() + bar.get_width()/2, value + 2,
                             f'{value:.0f}%', ha='center', va='bottom', fontsize=10)
            
            # Graph 4: Distance Decay Prediction
            if ttc != float('inf') and closing_speed > 0:
                time_points = np.linspace(0, min(ttc, 5), 50)
                distance_points = true_distance - closing_speed * time_points
                distance_points = np.maximum(distance_points, 0)
                
                # Add prediction zone
                current_brake = ml_brake if use_ml else rule_brake
                adjustment = 0.1 * current_brake * time_points
                upper_bound = distance_points + adjustment
                lower_bound = np.maximum(distance_points - adjustment, 0)
                
                self.ax4.plot(time_points, distance_points, '#4CAF50', linewidth=2, label='Actual')
                self.ax4.fill_between(time_points, lower_bound, upper_bound, alpha=0.2, 
                                      color='#2196F3', label='Uncertainty Zone')
                self.ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Collision')
                self.ax4.set_xlabel('Time (s)')
                self.ax4.set_ylabel('Distance (m)')
                
                if use_ml:
                    self.ax4.set_title('ML Distance Prediction', fontsize=12, fontweight='bold')
                else:
                    self.ax4.set_title('Distance Prediction', fontsize=12, fontweight='bold')
                self.ax4.legend()
                self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.text(0.5, 0.5, 'No Collision Risk\n(TTC is infinite or negative)',
                             ha='center', va='center', transform=self.ax4.transAxes,
                             fontsize=12, color='gray')
                self.ax4.set_title('Distance Prediction', fontsize=12, fontweight='bold')
            
            # Adjust layout
            self.fig.tight_layout(pad=2.0)
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating graphs: {e}")
            import traceback
            traceback.print_exc()
    
    def animate_scenario(self):
        """Animate scenario with ML predictions"""
        if self.is_animating:
            return
        
        self.is_animating = True
        original_text = self.simulate_btn.cget("text")
        self.simulate_btn.configure(text="Simulating...", state="disabled")
        
        original_distance = self.distance_var.get()
        
        def animate_step(step):
            if step >= 20 or not self.is_animating:
                self.is_animating = False
                self.simulate_btn.configure(text=original_text, state="normal")
                self.update_simulation()
                return
            
            # Decrease distance
            current_distance = original_distance * (1 - step/20)
            self.distance_var.set(max(1.0, current_distance))
            self.update_simulation()
            
            self.root.after(200, lambda: animate_step(step + 1))
        
        animate_step(0)
    
    def retrain_model(self):
        """Retrain the ML model with current history"""
        if self.ml_controller is None:
            self.mode_label.configure(text="ML controller not available", text_color="red")
            return
        
        if len(self.history) < 10:
            self.mode_label.configure(text="Need more data to retrain", text_color="orange")
            return
        
        # Prepare new training data from history
        X_new = []
        y_new = []
        
        for record in self.history:
            # Encode object type
            type_mapping = {"car": 0, "pedestrian": 1, "curb": 2}
            obj_encoded = type_mapping.get(record['object_type'], 0)
            
            X_new.append([record['distance'], record['our_speed'], 
                         record['object_speed'], obj_encoded])
            
            # Use ML brake as target
            y_new.append(record['ml_brake'])
        
        # Convert to numpy arrays
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        # Retrain model with new data
        X_combined = np.vstack([self.ml_controller.X, X_new])
        y_combined = np.concatenate([self.ml_controller.y, y_new])
        
        # Scale and retrain
        X_scaled = self.ml_controller.scaler.fit_transform(X_combined)
        self.ml_controller.model.fit(X_scaled, y_combined)
        
        # Update status
        self.mode_label.configure(text="Model Retrained with New Data", text_color="#4CAF50")
        self.update_simulation()
    
    def load_preset(self, distance, our_speed_kmh, object_speed_kmh, object_type):
        """Load preset scenario"""
        self.distance_var.set(distance)
        self.our_speed_var.set(our_speed_kmh)
        self.object_speed_var.set(object_speed_kmh)
        self.object_type_var.set(object_type)
        self.object_type_combo.set(object_type)
        
        self.update_simulation()
        self.mode_label.configure(text=f"Loaded: {object_type}", text_color="blue")
    
    def reset_values(self):
        """Reset to default values"""
        self.distance_var.set(50.0)
        self.our_speed_var.set(72.0)
        self.object_speed_var.set(0.0)
        self.object_type_var.set("car")
        self.object_type_combo.set("Car")
        self.use_ml_var.set(True)
        
        self.update_simulation()
        self.mode_label.configure(text="ML Mode: ACTIVE", text_color="#2196F3")

if __name__ == "__main__":
    root = ctk.CTk()
    app = MLBrakeGUI(root)
    root.mainloop()
