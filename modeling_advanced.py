"""
Advanced Modeling Module with Deep Learning, Ensemble Methods, and Model Interpretability
Includes: TensorFlow/PyTorch models, SHAP, LIME, Ensemble methods, AutoML
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from tpot import TPOTClassifier, TPOTRegressor
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

from pipeline_history import PipelineHistory


class AdvancedModeling:
    def __init__(self):
        self.history = PipelineHistory()

    
    def render_ensemble_methods(self, df, dataset_name):
        """Render ensemble methods interface"""
        st.subheader("🎭 Ensemble Methods")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for modeling.")
            return
        
        # Select target and features
        target_col = st.selectbox("Select Target Column", df.columns.tolist(), key="ens_target")
        
        if target_col:
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            if not feature_cols:
                st.warning("No feature columns available.")
                return
            
            # Task type
            task_type = st.radio("Task Type", ["Classification", "Regression"], key="ens_task")
            
            # Ensemble method
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["Voting", "Stacking", "Bagging", "Boosting"],
                key="ens_method"
            )
            
            # Model selection for ensemble
            if task_type == "Classification":
                available_models = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
                }
            else:
                available_models = {
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression()
                }
            
            selected_models = st.multiselect(
                "Select Base Models",
                list(available_models.keys()),
                default=list(available_models.keys())[:2],
                key="ens_models"
            )
            
            if len(selected_models) < 2:
                st.warning("Please select at least 2 models for ensemble.")
                return
            
            # Additional parameters
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="ens_test_size")
            
            if st.button("🎭 Train Ensemble Model", type="primary", key="train_ensemble"):
                self._train_ensemble_model(
                    df, dataset_name, feature_cols, target_col, task_type,
                    ensemble_method, selected_models, available_models, test_size
                )

    
    def _train_ensemble_model(self, df, dataset_name, feature_cols, target_col, task_type,
                             ensemble_method, selected_model_names, available_models, test_size):
        """Train ensemble model"""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Create base models
            base_models = [(name, available_models[name]) for name in selected_model_names]
            
            # Create ensemble
            if ensemble_method == "Voting":
                if task_type == "Classification":
                    ensemble = VotingClassifier(estimators=base_models, voting='soft')
                else:
                    ensemble = VotingRegressor(estimators=base_models)
            
            elif ensemble_method == "Stacking":
                if task_type == "Classification":
                    ensemble = StackingClassifier(
                        estimators=base_models,
                        final_estimator=LogisticRegression(random_state=42)
                    )
                else:
                    ensemble = StackingRegressor(
                        estimators=base_models,
                        final_estimator=LinearRegression()
                    )
            
            # Train ensemble
            with st.spinner(f"Training {ensemble_method} ensemble..."):
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                
                # Calculate metrics
                if task_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    st.success(f"✅ Ensemble trained successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("Precision", f"{precision:.4f}")
                    col3.metric("Recall", f"{recall:.4f}")
                    col4.metric("F1-Score", f"{f1:.4f}")
                    
                    metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    st.success(f"✅ Ensemble trained successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    col4.metric("MSE", f"{mse:.4f}")
                    
                    metrics = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'mse': mse
                    }
                
                # Compare with individual models
                st.markdown("### 📊 Individual Model Performance")
                self._compare_ensemble_models(base_models, X_train, X_test, y_train, y_test, task_type)
                
                # Store model
                if 'trained_models' not in st.session_state:
                    st.session_state.trained_models = {}
                
                model_key = f"{dataset_name}_{ensemble_method}_ensemble"
                st.session_state.trained_models[model_key] = {
                    'model': ensemble,
                    'type': task_type,
                    'method': ensemble_method,
                    'metrics': metrics,
                    'feature_cols': feature_cols,
                    'target_col': target_col
                }
                
                # Log operation
                self.history.log_step(
                    "Ensemble Training",
                    f"Trained {ensemble_method} ensemble with {len(selected_model_names)} models",
                    {
                        "method": ensemble_method,
                        "task_type": task_type,
                        "base_models": selected_model_names,
                        "metrics": metrics
                    },
                    "success"
                )
        
        except Exception as e:
            st.error(f"Error training ensemble: {str(e)}")
            self.history.log_step(
                "Ensemble Training",
                f"Failed ensemble training",
                {"error": str(e)},
                "error"
            )

    
    def _compare_ensemble_models(self, base_models, X_train, X_test, y_train, y_test, task_type):
        """Compare individual model performance"""
        try:
            results = []
            
            for name, model in base_models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if task_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    metric_name = "Accuracy"
                else:
                    score = r2_score(y_test, y_pred)
                    metric_name = "R² Score"
                
                results.append({
                    'Model': name,
                    metric_name: score
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualize comparison
            fig = go.Figure(data=[
                go.Bar(x=results_df['Model'], y=results_df[metric_name], marker_color='lightblue')
            ])
            fig.update_layout(
                title=f"Individual Model Performance ({metric_name})",
                xaxis_title="Model",
                yaxis_title=metric_name,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="ensemble_comparison")
        
        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")

    
    def render_deep_learning(self, df, dataset_name):
        """Render deep learning interface"""
        st.subheader("🧠 Deep Learning Models")
        
        if not HAS_TENSORFLOW and not HAS_PYTORCH:
            st.error("Deep learning libraries not available. Install TensorFlow or PyTorch.")
            st.code("pip install tensorflow torch")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for modeling.")
            return
        
        # Select target and features
        target_col = st.selectbox("Select Target Column", df.columns.tolist(), key="dl_target")
        
        if target_col:
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            if not feature_cols:
                st.warning("No feature columns available.")
                return
            
            # Task type
            task_type = st.radio("Task Type", ["Classification", "Regression"], key="dl_task")
            
            # Framework selection
            framework = st.selectbox(
                "Deep Learning Framework",
                ["TensorFlow/Keras", "PyTorch"],
                key="dl_framework"
            )
            
            if framework == "TensorFlow/Keras" and not HAS_TENSORFLOW:
                st.error("TensorFlow not installed.")
                return
            if framework == "PyTorch" and not HAS_PYTORCH:
                st.error("PyTorch not installed.")
                return
            
            # Network architecture
            st.markdown("### 🏗️ Network Architecture")
            
            n_layers = st.slider("Number of Hidden Layers", 1, 5, 2, key="dl_layers")
            
            layer_sizes = []
            for i in range(n_layers):
                size = st.slider(f"Layer {i+1} Size", 8, 512, 64, 8, key=f"dl_layer_{i}")
                layer_sizes.append(size)
            
            activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"], key="dl_activation")
            
            # Training parameters
            st.markdown("### ⚙️ Training Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                epochs = st.number_input("Epochs", 10, 1000, 100, 10, key="dl_epochs")
            with col2:
                batch_size = st.number_input("Batch Size", 8, 256, 32, 8, key="dl_batch")
            with col3:
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f", key="dl_lr")
            
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="dl_test_size")
            
            if st.button("🧠 Train Deep Learning Model", type="primary", key="train_dl"):
                if framework == "TensorFlow/Keras":
                    self._train_tensorflow_model(
                        df, dataset_name, feature_cols, target_col, task_type,
                        layer_sizes, activation, epochs, batch_size, learning_rate, test_size
                    )
                else:
                    self._train_pytorch_model(
                        df, dataset_name, feature_cols, target_col, task_type,
                        layer_sizes, activation, epochs, batch_size, learning_rate, test_size
                    )

    
    def _train_tensorflow_model(self, df, dataset_name, feature_cols, target_col, task_type,
                                layer_sizes, activation, epochs, batch_size, learning_rate, test_size):
        """Train TensorFlow/Keras model"""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean()).values
            y = df[target_col].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Build model
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(X_train.shape[1],)))
            
            for size in layer_sizes:
                model.add(keras.layers.Dense(size, activation=activation))
                model.add(keras.layers.Dropout(0.2))
            
            if task_type == "Classification":
                n_classes = len(np.unique(y))
                if n_classes == 2:
                    model.add(keras.layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:
                    model.add(keras.layers.Dense(n_classes, activation='softmax'))
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
            else:
                model.add(keras.layers.Dense(1))
                loss = 'mse'
                metrics = ['mae']
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=metrics
            )
            
            # Train model
            with st.spinner("Training deep learning model..."):
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Evaluate
                y_pred = model.predict(X_test)
                
                if task_type == "Classification":
                    if len(np.unique(y)) == 2:
                        y_pred_class = (y_pred > 0.5).astype(int).flatten()
                    else:
                        y_pred_class = np.argmax(y_pred, axis=1)
                    
                    accuracy = accuracy_score(y_test, y_pred_class)
                    st.success(f"✅ Model trained! Test Accuracy: {accuracy:.4f}")
                    
                    result_metrics = {'accuracy': accuracy}
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.success(f"✅ Model trained! Test R²: {r2:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("MAE", f"{mae:.4f}")
                    col3.metric("MSE", f"{mse:.4f}")
                    
                    result_metrics = {'r2': r2, 'mae': mae, 'mse': mse}
                
                # Plot training history
                self._plot_training_history(history, task_type)
                
                # Store model
                if 'trained_models' not in st.session_state:
                    st.session_state.trained_models = {}
                
                model_key = f"{dataset_name}_tensorflow_dl"
                st.session_state.trained_models[model_key] = {
                    'model': model,
                    'scaler': scaler,
                    'type': task_type,
                    'framework': 'TensorFlow',
                    'metrics': result_metrics,
                    'feature_cols': feature_cols,
                    'target_col': target_col
                }
                
                # Log operation
                self.history.log_step(
                    "Deep Learning Training",
                    f"Trained TensorFlow model with {len(layer_sizes)} layers",
                    {
                        "framework": "TensorFlow",
                        "task_type": task_type,
                        "layers": layer_sizes,
                        "epochs": epochs,
                        "metrics": result_metrics
                    },
                    "success"
                )
        
        except Exception as e:
            st.error(f"Error training TensorFlow model: {str(e)}")
            self.history.log_step(
                "Deep Learning Training",
                f"Failed TensorFlow training",
                {"error": str(e)},
                "error"
            )

    
    def _plot_training_history(self, history, task_type):
        """Plot training history"""
        try:
            st.markdown("### 📈 Training History")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Metrics"))
            
            # Loss
            fig.add_trace(
                go.Scatter(y=history.history['loss'], name='Training Loss', mode='lines'),
                row=1, col=1
            )
            if 'val_loss' in history.history:
                fig.add_trace(
                    go.Scatter(y=history.history['val_loss'], name='Validation Loss', mode='lines'),
                    row=1, col=1
                )
            
            # Metrics
            metric_key = 'accuracy' if task_type == "Classification" else 'mae'
            if metric_key in history.history:
                fig.add_trace(
                    go.Scatter(y=history.history[metric_key], name=f'Training {metric_key}', mode='lines'),
                    row=1, col=2
                )
            if f'val_{metric_key}' in history.history:
                fig.add_trace(
                    go.Scatter(y=history.history[f'val_{metric_key}'], name=f'Validation {metric_key}', mode='lines'),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, key="dl_history")
        
        except Exception as e:
            st.error(f"Error plotting history: {str(e)}")

    
    def render_model_interpretability(self, df, dataset_name):
        """Render model interpretability interface"""
        st.subheader("🔍 Model Interpretability")
        
        if not HAS_SHAP and not HAS_LIME:
            st.error("Interpretability libraries not available. Install SHAP or LIME.")
            st.code("pip install shap lime")
            return
        
        # Check if models are trained
        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.warning("No trained models available. Please train a model first.")
            return
        
        # Select model
        model_names = list(st.session_state.trained_models.keys())
        selected_model_key = st.selectbox("Select Model", model_names, key="interp_model")
        
        if selected_model_key:
            model_info = st.session_state.trained_models[selected_model_key]
            model = model_info['model']
            feature_cols = model_info['feature_cols']
            
            # Interpretability method
            method = st.selectbox(
                "Interpretability Method",
                ["SHAP (SHapley Additive exPlanations)", "LIME (Local Interpretable Model-agnostic Explanations)"],
                key="interp_method"
            )
            
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            if method == "SHAP (SHapley Additive exPlanations)" and HAS_SHAP:
                if st.button("🔍 Generate SHAP Explanations", type="primary", key="gen_shap"):
                    self._generate_shap_explanations(model, X, feature_cols, model_info)
            
            elif method == "LIME (Local Interpretable Model-agnostic Explanations)" and HAS_LIME:
                instance_idx = st.number_input(
                    "Select Instance Index to Explain",
                    0, len(X)-1, 0,
                    key="lime_idx"
                )
                
                if st.button("🔍 Generate LIME Explanation", type="primary", key="gen_lime"):
                    self._generate_lime_explanation(model, X, feature_cols, instance_idx, model_info)

    
    def _generate_shap_explanations(self, model, X, feature_cols, model_info):
        """Generate SHAP explanations"""
        try:
            with st.spinner("Generating SHAP explanations..."):
                # Create explainer
                if hasattr(model, 'predict_proba'):
                    explainer = shap.Explainer(model, X[:100])  # Use sample for speed
                else:
                    explainer = shap.Explainer(model.predict, X[:100])
                
                # Calculate SHAP values
                shap_values = explainer(X[:100])
                
                st.success("✅ SHAP explanations generated!")
                
                # Summary plot
                st.markdown("### 📊 SHAP Summary Plot")
                st.info("Shows feature importance across all samples")
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X[:100], feature_names=feature_cols, show=False)
                st.pyplot(fig)
                plt.close()
                
                # Feature importance
                st.markdown("### 📊 Feature Importance (SHAP)")
                
                shap_importance = np.abs(shap_values.values).mean(axis=0)
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': shap_importance
                }).sort_values('Importance', ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h')
                ])
                fig.update_layout(
                    title="SHAP Feature Importance",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Feature",
                    height=max(400, len(feature_cols) * 25)
                )
                st.plotly_chart(fig, use_container_width=True, key="shap_importance")
                
                # Log operation
                self.history.log_step(
                    "Model Interpretability",
                    "Generated SHAP explanations",
                    {
                        "method": "SHAP",
                        "n_samples": 100,
                        "n_features": len(feature_cols)
                    },
                    "success"
                )
        
        except Exception as e:
            st.error(f"Error generating SHAP explanations: {str(e)}")
            self.history.log_step(
                "Model Interpretability",
                "Failed SHAP generation",
                {"error": str(e)},
                "error"
            )
    
    def _generate_lime_explanation(self, model, X, feature_cols, instance_idx, model_info):
        """Generate LIME explanation for a single instance"""
        try:
            with st.spinner("Generating LIME explanation..."):
                # Create explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=feature_cols,
                    mode='classification' if model_info['type'] == 'Classification' else 'regression'
                )
                
                # Explain instance
                if hasattr(model, 'predict_proba'):
                    exp = explainer.explain_instance(
                        X.iloc[instance_idx].values,
                        model.predict_proba,
                        num_features=len(feature_cols)
                    )
                else:
                    exp = explainer.explain_instance(
                        X.iloc[instance_idx].values,
                        model.predict,
                        num_features=len(feature_cols)
                    )
                
                st.success("✅ LIME explanation generated!")
                
                # Show instance details
                st.markdown(f"### 📊 Explanation for Instance {instance_idx}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Instance Values:**")
                    instance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Value': X.iloc[instance_idx].values
                    })
                    st.dataframe(instance_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Feature Contributions:**")
                    lime_exp = exp.as_list()
                    lime_df = pd.DataFrame(lime_exp, columns=['Feature', 'Contribution'])
                    st.dataframe(lime_df, use_container_width=True)
                
                # Visualize
                fig = go.Figure(data=[
                    go.Bar(
                        x=[item[1] for item in lime_exp],
                        y=[item[0] for item in lime_exp],
                        orientation='h',
                        marker=dict(color=['green' if item[1] > 0 else 'red' for item in lime_exp])
                    )
                ])
                fig.update_layout(
                    title=f"LIME Explanation for Instance {instance_idx}",
                    xaxis_title="Contribution",
                    yaxis_title="Feature",
                    height=max(400, len(lime_exp) * 25)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"lime_exp_{instance_idx}")
                
                # Log operation
                self.history.log_step(
                    "Model Interpretability",
                    f"Generated LIME explanation for instance {instance_idx}",
                    {
                        "method": "LIME",
                        "instance_idx": instance_idx
                    },
                    "success"
                )
        
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")
            self.history.log_step(
                "Model Interpretability",
                "Failed LIME generation",
                {"error": str(e)},
                "error"
            )
