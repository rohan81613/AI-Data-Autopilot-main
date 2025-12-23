import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Try to import TPOT for AutoML
try:
    from tpot import TPOTClassifier, TPOTRegressor
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

# Try to import Auto-sklearn
try:
    import autosklearn.classification
    import autosklearn.regression
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False

from pipeline_history import PipelineHistory


class EnhancedModelTraining:
    def __init__(self):
        self.history = PipelineHistory()
        self.models = {
            'Classification': {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42),
                'SVM': SVC(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB()
            },
            'Regression': {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.models['Classification']['XGBoost'] = xgb.XGBClassifier(random_state=42)
            self.models['Regression']['XGBoost'] = xgb.XGBRegressor(random_state=42)

    def render_modeling_ui(self):
        """Render the enhanced modeling interface"""
        
        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        if not dataset_names:
            st.warning("No datasets available. Please load a dataset first.")
            return
            
        selected_dataset = st.selectbox("Select Dataset for Modeling", dataset_names, key="enhanced_model_dataset_selector")

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Dataset:** {selected_dataset} ({df.shape[0]} rows × {df.shape[1]} columns)")

        # Enhanced tabs for different modeling sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Problem Definition",
            "🤖 No-Code ML Builder",
            "🎛️ AutoML Features",
            "📊 Model Interpretability",
            "🚀 Model Deployment"
        ])

        with tab1:
            self._render_problem_definition(df, selected_dataset)

        with tab2:
            self._render_nocode_ml_builder(df, selected_dataset)

        with tab3:
            self._render_automl_features(df, selected_dataset)

        with tab4:
            self._render_model_interpretability(df, selected_dataset)

        with tab5:
            self._render_model_deployment(df, selected_dataset)

    def _render_problem_definition(self, df, dataset_name):
        """Render problem definition section"""
        st.subheader("🎯 Problem Definition")
        
        st.markdown("### 📋 Define Your Machine Learning Problem")
        
        # Task type selection
        task_type = st.radio("Select Task Type", ["Classification", "Regression"])
        
        # Target variable selection
        target_col = st.selectbox("Select Target Variable", df.columns.tolist())
        
        if target_col:
            # Show target variable statistics
            st.markdown("### 📊 Target Variable Analysis")
            
            if task_type == "Classification":
                # Classification target analysis
                st.markdown("**Classification Target Distribution:**")
                target_counts = df[target_col].value_counts()
                st.dataframe(target_counts.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}))
                
                # Pie chart for class distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
                ax.set_title(f"Class Distribution: {target_col}")
                st.pyplot(fig)
                plt.close()
                
            else:
                # Regression target analysis
                st.markdown("**Regression Target Statistics:**")
                target_stats = df[target_col].describe()
                st.dataframe(target_stats)
                
                # Histogram for target distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(df[target_col], bins=30, edgecolor='black')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {target_col}')
                st.pyplot(fig)
                plt.close()
            
            # Feature selection
            st.markdown("### 🎯 Feature Selection")
            feature_cols = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect("Select Features for Modeling", feature_cols, default=feature_cols)
            
            if selected_features:
                # Data preprocessing
                st.markdown("### 🧹 Data Preprocessing")
                
                # Handle categorical variables
                categorical_cols = df[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    encoding_method = st.selectbox("Categorical Encoding Method", [
                        "One-Hot Encoding",
                        "Label Encoding"
                    ])
                    
                    if encoding_method == "One-Hot Encoding":
                        # One-hot encode categorical features
                        dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
                        df_encoded = pd.concat([
                            df.drop(categorical_cols, axis=1), 
                            dummies
                        ], axis=1)
                    else:
                        # Label encode categorical features
                        df_encoded = df.copy()
                        le = LabelEncoder()
                        for col in categorical_cols:
                            df_encoded[col] = le.fit_transform(df[col].astype(str))
                else:
                    df_encoded = df.copy()
                
                # Handle missing values
                if df_encoded[selected_features].isnull().sum().sum() > 0:
                    imputation_method = st.selectbox("Missing Value Imputation", [
                        "Mean (for numeric)",
                        "Median (for numeric)",
                        "Mode (for categorical)",
                        "Drop rows with missing values"
                    ])
                    
                    if imputation_method == "Drop rows with missing values":
                        df_encoded = df_encoded.dropna(subset=selected_features + [target_col])
                    else:
                        for col in selected_features:
                            if df_encoded[col].isnull().sum() > 0:
                                if df_encoded[col].dtype in ['object', 'category']:
                                    mode_value = df_encoded[col].mode()
                                    if not mode_value.empty:
                                        df_encoded[col].fillna(mode_value.iloc[0], inplace=True)
                                else:
                                    if imputation_method == "Mean (for numeric)":
                                        df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)
                                    else:
                                        df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
                
                # Save processed data to session state
                st.session_state.datasets[f"{dataset_name}_processed"] = df_encoded
                
                # Create modeling configuration
                if 'modeling_config' not in st.session_state:
                    st.session_state.modeling_config = {}
                
                st.session_state.modeling_config[dataset_name] = {
                    'task_type': task_type,
                    'target_col': target_col,
                    'selected_features': selected_features,
                    'dataset_name': f"{dataset_name}_processed"
                }
                
                st.success("✅ Problem definition completed successfully!")
                st.info(f"Ready to build models with {len(selected_features)} features for {task_type} task.")

    def _render_nocode_ml_builder(self, df, dataset_name):
        """Render no-code ML builder section"""
        st.subheader("🤖 No-Code ML Builder")
        
        st.markdown("### 🧱 Drag-and-Drop Model Builder")
        st.info("Create your machine learning pipeline with a visual interface.")
        
        # Check if problem is defined
        if 'modeling_config' not in st.session_state or dataset_name not in st.session_state.modeling_config:
            st.warning("Please define your problem first in the 'Problem Definition' tab.")
            return
        
        config = st.session_state.modeling_config[dataset_name]
        task_type = config['task_type']
        target_col = config['target_col']
        selected_features = config['selected_features']
        processed_dataset_name = config['dataset_name']
        
        # Load processed data
        df_processed = st.session_state.datasets[processed_dataset_name]
        
        # Pipeline builder
        st.markdown("### ⚙️ Build Your ML Pipeline")
        
        # Step 1: Data Splitting
        st.markdown("#### Step 1: Data Splitting")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20)
        random_state = st.number_input("Random State", value=42)
        
        # Step 2: Model Selection
        st.markdown("#### Step 2: Model Selection")
        available_models = list(self.models[task_type].keys())
        selected_models = st.multiselect("Select Models to Compare", available_models, default=available_models[:3])
        
        # Step 3: Hyperparameter Tuning
        st.markdown("#### Step 3: Hyperparameter Tuning")
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        if enable_tuning:
            st.info("Hyperparameter tuning will use Grid Search with cross-validation.")
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        # Step 4: Model Training
        st.markdown("#### Step 4: Model Training")
        
        if st.button("🚀 Train Models", key="train_models_nocode"):
            with st.spinner("Training models..."):
                try:
                    # Prepare data
                    X = df_processed[selected_features]
                    y = df_processed[target_col]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=random_state
                    )
                    
                    # Store split data in session state
                    st.session_state.model_data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'task_type': task_type,
                        'target_col': target_col
                    }
                    
                    # Train selected models
                    results = []
                    trained_models = {}
                    
                    for model_name in selected_models:
                        with st.spinner(f"Training {model_name}..."):
                            try:
                                model = self.models[task_type][model_name]
                                
                                if enable_tuning:
                                    # Define parameter grids for tuning
                                    param_grids = {
                                        'Random Forest': {
                                            'n_estimators': [50, 100, 200],
                                            'max_depth': [None, 10, 20],
                                            'min_samples_split': [2, 5]
                                        },
                                        'Logistic Regression': {
                                            'C': [0.1, 1, 10],
                                            'penalty': ['l1', 'l2']
                                        },
                                        'SVM': {
                                            'C': [0.1, 1, 10],
                                            'kernel': ['rbf', 'linear']
                                        },
                                        'K-Nearest Neighbors': {
                                            'n_neighbors': [3, 5, 7],
                                            'weights': ['uniform', 'distance']
                                        },
                                        'Decision Tree': {
                                            'max_depth': [None, 10, 20],
                                            'min_samples_split': [2, 5]
                                        }
                                    }
                                    
                                    if model_name in param_grids:
                                        # Perform grid search
                                        grid_search = GridSearchCV(
                                            model, 
                                            param_grids[model_name], 
                                            cv=cv_folds, 
                                            scoring='accuracy' if task_type == 'Classification' else 'r2'
                                        )
                                        grid_search.fit(X_train, y_train)
                                        best_model = grid_search.best_estimator_
                                        
                                        # Store best parameters
                                        st.info(f"{model_name} best parameters: {grid_search.best_params_}")
                                    else:
                                        # Train without tuning
                                        best_model = model
                                        best_model.fit(X_train, y_train)
                                else:
                                    # Train without tuning
                                    best_model = model
                                    best_model.fit(X_train, y_train)
                                
                                # Make predictions
                                y_pred = best_model.predict(X_test)
                                
                                # Calculate metrics
                                if task_type == "Classification":
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                    
                                    metrics = {
                                        'Accuracy': accuracy,
                                        'Precision': precision,
                                        'Recall': recall,
                                        'F1-Score': f1
                                    }
                                else:
                                    mse = mean_squared_error(y_test, y_pred)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    metrics = {
                                        'MSE': mse,
                                        'MAE': mae,
                                        'R²': r2
                                    }
                                
                                # Cross-validation score
                                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
                                cv_mean = cv_scores.mean()
                                cv_std = cv_scores.std()
                                
                                # Store results
                                results.append({
                                    'Model': model_name,
                                    'Metrics': metrics,
                                    'CV Mean': cv_mean,
                                    'CV Std': cv_std
                                })
                                
                                # Store trained model
                                trained_models[model_name] = best_model
                                
                                st.success(f"✅ {model_name} trained successfully!")
                                
                            except Exception as e:
                                st.error(f"Error training {model_name}: {str(e)}")
                    
                    # Display results
                    if results:
                        st.markdown("### 📊 Model Comparison Results")
                        
                        # Create results DataFrame
                        results_data = []
                        for result in results:
                            row = {'Model': result['Model']}
                            row.update(result['Metrics'])
                            row['CV Mean'] = result['CV Mean']
                            row['CV Std'] = result['CV Std']
                            results_data.append(row)
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df)
                        
                        # Visualize results
                        if task_type == "Classification":
                            metric_to_plot = "Accuracy"
                        else:
                            metric_to_plot = "R²"
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        models = results_df['Model']
                        scores = results_df[metric_to_plot]
                        cv_means = results_df['CV Mean']
                        
                        x = np.arange(len(models))
                        width = 0.35
                        
                        ax.bar(x - width/2, scores, width, label=f'Test {metric_to_plot}')
                        ax.bar(x + width/2, cv_means, width, label='CV Mean')
                        
                        ax.set_xlabel('Models')
                        ax.set_ylabel(metric_to_plot)
                        ax.set_title('Model Performance Comparison')
                        ax.set_xticks(x)
                        ax.set_xticklabels(models, rotation=45)
                        ax.legend()
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Store results in session state
                        st.session_state.model_results = {
                            'results_df': results_df,
                            'trained_models': trained_models,
                            'best_model': results_df.loc[results_df[metric_to_plot].idxmax(), 'Model']
                        }
                        
                        # Log to history
                        self.history.log_step(
                            "Model Training",
                            f"Trained {len(selected_models)} models for {dataset_name}",
                            {
                                "models_trained": len(selected_models),
                                "best_model": results_df.loc[results_df[metric_to_plot].idxmax(), 'Model'],
                                "best_score": results_df[metric_to_plot].max()
                            },
                            "success"
                        )
                        
                        st.success("✅ All models trained successfully!")
                        
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

    def _render_automl_features(self, df, dataset_name):
        """Render AutoML features section"""
        st.subheader("🎛️ AutoML Features")
        
        st.markdown("### 🤖 Automated Machine Learning")
        st.info("Automatically find the best model and hyperparameters for your data.")
        
        # Check if problem is defined
        if 'modeling_config' not in st.session_state or dataset_name not in st.session_state.modeling_config:
            st.warning("Please define your problem first in the 'Problem Definition' tab.")
            return
        
        config = st.session_state.modeling_config[dataset_name]
        task_type = config['task_type']
        target_col = config['target_col']
        selected_features = config['selected_features']
        processed_dataset_name = config['dataset_name']
        
        # Load processed data
        df_processed = st.session_state.datasets[processed_dataset_name]
        
        # AutoML options
        st.markdown("### ⚙️ AutoML Configuration")
        
        # Time limit for optimization
        time_limit = st.slider("Time Limit (minutes)", 1, 60, 10)
        
        # Model types to include
        st.markdown("#### Model Types to Include")
        include_rf = st.checkbox("Random Forest", value=True)
        include_linear = st.checkbox("Linear Models", value=True)
        include_svm = st.checkbox("Support Vector Machines", value=True)
        include_knn = st.checkbox("K-Nearest Neighbors", value=True)
        include_tree = st.checkbox("Decision Trees", value=True)
        include_xgb = st.checkbox("XGBoost", value=HAS_XGBOOST)
        include_nn = st.checkbox("Neural Networks", value=False)
        
        # TPOT AutoML
        if HAS_TPOT:
            st.markdown("### 🧬 TPOT Genetic Programming")
            st.info("TPOT uses genetic programming to optimize machine learning pipelines.")
            
            generations = st.slider("Generations", 1, 20, 5)
            population_size = st.slider("Population Size", 10, 200, 50)
            
            if st.button("🧬 Run TPOT Optimization", key="run_tpot_automl"):
                with st.spinner("Running TPOT optimization..."):
                    try:
                        # Prepare data
                        X = df_processed[selected_features]
                        y = df_processed[target_col]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        if task_type == "Classification":
                            tpot = TPOTClassifier(
                                generations=generations,
                                population_size=population_size,
                                verbosity=2,
                                random_state=42,
                                max_time_mins=time_limit
                            )
                        else:
                            tpot = TPOTRegressor(
                                generations=generations,
                                population_size=population_size,
                                verbosity=2,
                                random_state=42,
                                max_time_mins=time_limit
                            )
                        
                        # Fit TPOT
                        tpot.fit(X_train, y_train)
                        
                        # Evaluate best pipeline
                        y_pred = tpot.predict(X_test)
                        
                        if task_type == "Classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            st.success(f"✅ TPOT optimization completed! Test Accuracy: {accuracy:.4f}")
                        else:
                            r2 = r2_score(y_test, y_pred)
                            st.success(f"✅ TPOT optimization completed! Test R²: {r2:.4f}")
                        
                        # Show best pipeline
                        st.markdown("#### Best Pipeline Found:")
                        st.code(tpot.fitted_pipeline_, language="python")
                        
                        # Export pipeline
                        pipeline_code = tpot.export()
                        st.download_button(
                            label="💾 Download Best Pipeline",
                            data=pipeline_code,
                            file_name="tpot_pipeline.py",
                            mime="text/plain"
                        )
                        
                        # Log to history
                        self.history.log_step(
                            "TPOT Optimization",
                            f"Completed TPOT optimization for {dataset_name}",
                            {
                                "task_type": task_type,
                                "generations": generations,
                                "population_size": population_size,
                                "time_limit": time_limit
                            },
                            "success"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during TPOT optimization: {str(e)}")
        else:
            st.info("TPOT not installed. Install with: pip install tpot")
        
        # Auto-sklearn (if available)
        if HAS_AUTOSKLEARN:
            st.markdown("### 🧠 Auto-sklearn")
            st.info("Auto-sklearn automatically searches for the best machine learning pipeline.")
            
            time_left_for_this_task = st.slider("Time for Task (seconds)", 30, 3600, 300)
            per_run_time_limit = st.slider("Per Run Time Limit (seconds)", 30, 300, 60)
            
            if st.button("🧠 Run Auto-sklearn", key="run_autosklearn"):
                with st.spinner("Running Auto-sklearn optimization..."):
                    try:
                        # Prepare data
                        X = df_processed[selected_features]
                        y = df_processed[target_col]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        if task_type == "Classification":
                            automl = autosklearn.classification.AutoSklearnClassifier(
                                time_left_for_this_task=time_left_for_this_task,
                                per_run_time_limit=per_run_time_limit,
                                tmp_folder=f'/tmp/autosklearn_{dataset_name}_classification',
                                output_folder=f'/tmp/autosklearn_{dataset_name}_classification_out'
                            )
                        else:
                            automl = autosklearn.regression.AutoSklearnRegressor(
                                time_left_for_this_task=time_left_for_this_task,
                                per_run_time_limit=per_run_time_limit,
                                tmp_folder=f'/tmp/autosklearn_{dataset_name}_regression',
                                output_folder=f'/tmp/autosklearn_{dataset_name}_regression_out'
                            )
                        
                        # Fit Auto-sklearn
                        automl.fit(X_train, y_train)
                        
                        # Evaluate best pipeline
                        y_pred = automl.predict(X_test)
                        
                        if task_type == "Classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            st.success(f"✅ Auto-sklearn optimization completed! Test Accuracy: {accuracy:.4f}")
                        else:
                            r2 = r2_score(y_test, y_pred)
                            st.success(f"✅ Auto-sklearn optimization completed! Test R²: {r2:.4f}")
                        
                        # Show model leaderboard
                        st.markdown("#### Model Leaderboard:")
                        leaderboard = automl.leaderboard()
                        st.dataframe(leaderboard)
                        
                        # Show ensemble details
                        st.markdown("#### Ensemble Details:")
                        ensemble = automl.show_models()
                        st.text(ensemble)
                        
                        # Log to history
                        self.history.log_step(
                            "Auto-sklearn Optimization",
                            f"Completed Auto-sklearn optimization for {dataset_name}",
                            {
                                "task_type": task_type,
                                "time_left_for_this_task": time_left_for_this_task,
                                "per_run_time_limit": per_run_time_limit
                            },
                            "success"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during Auto-sklearn optimization: {str(e)}")
        else:
            st.info("Auto-sklearn not installed. Install with: pip install auto-sklearn")

    def _render_model_interpretability(self, df, dataset_name):
        """Render model interpretability section"""
        st.subheader("📊 Model Interpretability")
        
        st.markdown("### 📈 Feature Importance & Model Insights")
        st.info("Understand which features are most important for your model's predictions.")
        
        # Check if models have been trained
        if 'model_results' not in st.session_state:
            st.warning("Please train models first in the 'No-Code ML Builder' tab.")
            return
        
        model_results = st.session_state.model_results
        trained_models = model_results['trained_models']
        results_df = model_results['results_df']
        
        # Select model for interpretation
        selected_model = st.selectbox("Select Model for Interpretation", list(trained_models.keys()))
        
        if selected_model:
            model = trained_models[selected_model]
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🌟 Feature Importance")
                
                # Get feature names
                if 'model_data' in st.session_state:
                    feature_names = st.session_state.model_data['X_train'].columns.tolist()
                    importances = model.feature_importances_
                    
                    # Create feature importance DataFrame
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Display top features
                    st.dataframe(feature_importance_df.head(15))
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    top_features = feature_importance_df.head(15)
                    ax.barh(top_features['Feature'], top_features['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Top 15 Feature Importances - {selected_model}')
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show cumulative importance
                    cumulative_importance = feature_importance_df['Importance'].cumsum()
                    num_features_95 = (cumulative_importance >= 0.95).idxmax() + 1
                    
                    st.info(f"Top {num_features_95} features account for 95% of the importance.")
            
            # Model predictions analysis
            st.markdown("### 📊 Prediction Analysis")
            
            if 'model_data' in st.session_state:
                X_test = st.session_state.model_data['X_test']
                y_test = st.session_state.model_data['y_test']
                task_type = st.session_state.model_data['task_type']
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                if task_type == "Classification":
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    st.markdown("#### Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification report
                    from sklearn.metrics import classification_report
                    st.markdown("#### Classification Report")
                    st.text(classification_report(y_test, y_pred))
                
                else:
                    # Actual vs Predicted plot
                    st.markdown("#### Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title(f'Actual vs Predicted - {selected_model}')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Residuals plot
                    st.markdown("#### Residuals Plot")
                    residuals = y_test - y_pred
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_pred, residuals, alpha=0.5)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Residuals')
                    ax.set_title(f'Residuals Plot - {selected_model}')
                    st.pyplot(fig)
                    plt.close()
            
            # Partial dependence plots (if scikit-learn version supports it)
            try:
                from sklearn.inspection import PartialDependenceDisplay
                
                st.markdown("### 📉 Partial Dependence Plots")
                st.info("Shows how features affect model predictions.")
                
                if 'model_data' in st.session_state:
                    X_train = st.session_state.model_data['X_train']
                    feature_names = X_train.columns.tolist()
                    
                    # Select features for partial dependence
                    pd_features = st.multiselect(
                        "Select features for partial dependence analysis",
                        feature_names,
                        default=feature_names[:3] if len(feature_names) >= 3 else feature_names
                    )
                    
                    if pd_features:
                        try:
                            fig, ax = plt.subplots(figsize=(12, 4 * len(pd_features)))
                            PartialDependenceDisplay.from_estimator(
                                model, X_train, pd_features, ax=ax
                            )
                            ax.set_title(f'Partial Dependence - {selected_model}')
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Could not generate partial dependence plots: {str(e)}")
            except ImportError:
                st.info("Partial dependence plots require scikit-learn 0.24+")

    def _render_model_deployment(self, df, dataset_name):
        """Render model deployment section"""
        st.subheader("🚀 Model Deployment")
        
        st.markdown("### ☁️ One-Click Model Deployment")
        st.info("Deploy your trained models as APIs for real-time predictions.")
        
        # Check if models have been trained
        if 'model_results' not in st.session_state:
            st.warning("Please train models first in the 'No-Code ML Builder' tab.")
            return
        
        model_results = st.session_state.model_results
        trained_models = model_results['trained_models']
        best_model_name = model_results['best_model']
        
        # Select model for deployment
        selected_model = st.selectbox(
            "Select Model for Deployment", 
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name) if best_model_name in trained_models else 0
        )
        
        if selected_model:
            model = trained_models[selected_model]
            
            # Deployment options
            st.markdown("### 🛠️ Deployment Configuration")
            
            # API framework selection
            framework = st.selectbox("Select API Framework", [
                "Flask",
                "FastAPI",
                "Streamlit"
            ])
            
            # Model serialization
            st.markdown("### 💾 Model Serialization")
            
            if st.button("💾 Save Model", key="save_model_deployment"):
                try:
                    import joblib
                    import pickle
                    
                    # Save with joblib
                    joblib_filename = f"{selected_model.replace(' ', '_')}_{dataset_name}.joblib"
                    joblib.dump(model, joblib_filename)
                    
                    # Save with pickle
                    pickle_filename = f"{selected_model.replace(' ', '_')}_{dataset_name}.pkl"
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(model, f)
                    
                    st.success(f"✅ Model saved successfully!")
                    st.info(f"Files created: {joblib_filename}, {pickle_filename}")
                    
                    # Provide download links
                    with open(joblib_filename, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Joblib Model",
                            data=f,
                            file_name=joblib_filename,
                            mime="application/octet-stream"
                        )
                    
                    with open(pickle_filename, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Pickle Model",
                            data=f,
                            file_name=pickle_filename,
                            mime="application/octet-stream"
                        )
                    
                    # Log to history
                    self.history.log_step(
                        "Model Deployment",
                        f"Saved model {selected_model} for {dataset_name}",
                        {
                            "model_name": selected_model,
                            "framework": framework,
                            "files_created": [joblib_filename, pickle_filename]
                        },
                        "success"
                    )
                    
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
            
            # Generate API code
            st.markdown("### 🧾 API Code Generation")
            
            if st.button("🐍 Generate API Code", key="generate_api_code"):
                try:
                    # Get feature names
                    if 'model_data' in st.session_state:
                        feature_names = st.session_state.model_data['X_train'].columns.tolist()
                    else:
                        feature_names = ["feature_1", "feature_2", "feature_3"]  # Default
                    
                    # Generate Flask API code
                    if framework == "Flask":
                        api_code = f'''from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("{selected_model.replace(' ', '_')}_{dataset_name}.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json(force=True)
        
        # Extract features (modify according to your features)
        features = [data.get(feat, 0) for feat in {feature_names}]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Return result
        return jsonify({{
            'prediction': str(prediction)
        }})
    
    except Exception as e:
        return jsonify({{
            'error': str(e)
        }})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
                    
                    # Generate FastAPI code
                    elif framework == "FastAPI":
                        api_code = f'''from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("{selected_model.replace(' ', '_')}_{dataset_name}.joblib")

# Define input data model
class PredictionInput(BaseModel):
    # Modify according to your features
    pass

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Extract features
        features = []  # Extract from input_data
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Return result
        return {{"prediction": str(prediction)}}
    
    except Exception as e:
        return {{"error": str(e)}}
'''
                    
                    # Generate Streamlit API code
                    else:  # Streamlit
                        api_code = f'''import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("{selected_model.replace(' ', '_')}_{dataset_name}.joblib")

st.title("Model Prediction API")

# Create input fields for features
# Modify according to your features
feature_inputs = []
for i, feature in enumerate({feature_names}):
    value = st.number_input(f"Enter {feature}", value=0.0)
    feature_inputs.append(value)

if st.button("Predict"):
    try:
        # Make prediction
        prediction = model.predict([feature_inputs])[0]
        
        # Display result
        st.success(f"Prediction: {{prediction}}")
    
    except Exception as e:
        st.error(f"Error: {{str(e)}}")
'''
                    
                    # Display code
                    st.markdown("#### Generated API Code:")
                    st.code(api_code, language="python")
                    
                    # Download button
                    st.download_button(
                        label="💾 Download API Code",
                        data=api_code,
                        file_name=f"{selected_model.replace(' ', '_')}_api.py",
                        mime="text/plain"
                    )
                    
                    st.success(f"✅ {framework} API code generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating API code: {str(e)}")
            
            # Docker deployment
            st.markdown("### 🐳 Docker Deployment")
            st.info("Containerize your model for easy deployment.")
            
            if st.button("🐳 Generate Dockerfile", key="generate_dockerfile"):
                try:
                    dockerfile_content = f'''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "{selected_model.replace(' ', '_')}_api.py"]
'''
                    
                    requirements_content = f'''flask==2.0.1
joblib==1.1.0
numpy==1.21.0
scikit-learn==1.0.0
'''
                    
                    # Display Dockerfile
                    st.markdown("#### Dockerfile:")
                    st.code(dockerfile_content, language="dockerfile")
                    
                    # Display requirements.txt
                    st.markdown("#### requirements.txt:")
                    st.code(requirements_content, language="text")
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="💾 Download Dockerfile",
                            data=dockerfile_content,
                            file_name="Dockerfile",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            label="💾 Download requirements.txt",
                            data=requirements_content,
                            file_name="requirements.txt",
                            mime="text/plain"
                        )
                    
                    st.success("✅ Docker deployment files generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating Docker files: {str(e)}")
            
            # Model monitoring
            st.markdown("### 📡 Model Monitoring")
            st.info("Track model performance in production.")
            
            monitoring_options = st.multiselect("Select Monitoring Features", [
                "Performance Tracking",
                "Data Drift Detection",
                "Prediction Logging",
                "Alerts & Notifications"
            ])
            
            if monitoring_options:
                st.info("In a production environment, you would implement:")
                for option in monitoring_options:
                    st.markdown(f"- **{option}**: Set up appropriate monitoring infrastructure")

# For backward compatibility
def render_modeling_ui():
    """Render modeling interface (for backward compatibility)"""
    st.subheader("🤖 Enhanced Feature Selection, Model Training & Validation")
    st.info("This interface has been enhanced. Please use the new enhanced modeling tab for advanced features.")