
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import base64
from pipeline_history import PipelineHistory

# Import models
try:
    from sklearn.linear_model import (
        LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
        BayesianRidge, HuberRegressor
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor,
        BaggingClassifier, BaggingRegressor, IsolationForest
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

    # Try to import additional models
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False

    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
    except ImportError:
        HAS_LIGHTGBM = False

    try:
        import catboost as cb
        HAS_CATBOOST = True
    except ImportError:
        HAS_CATBOOST = False

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    st.error(f"Some ML libraries are missing: {e}")

class ModelTraining:
    def __init__(self):
        self.history = PipelineHistory()

        # Define available models
        self.classification_models = {
            'Logistic Regression': LogisticRegression,
            'Random Forest': RandomForestClassifier,
            'Decision Tree': DecisionTreeClassifier,
            'SVM': SVC,
            'KNN': KNeighborsClassifier,
            'Naive Bayes': GaussianNB,
            'MLP Neural Network': MLPClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'AdaBoost': AdaBoostClassifier,
            'Extra Trees': ExtraTreesClassifier,
            'Bagging': BaggingClassifier,
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis,
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis,
            'Gaussian Process': GaussianProcessClassifier
        }

        self.regression_models = {
            'Linear Regression': LinearRegression,
            'Random Forest': RandomForestRegressor,
            'Decision Tree': DecisionTreeRegressor,
            'SVR': SVR,
            'KNN': KNeighborsRegressor,
            'MLP Neural Network': MLPRegressor,
            'Gradient Boosting': GradientBoostingRegressor,
            'AdaBoost': AdaBoostRegressor,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'Extra Trees': ExtraTreesRegressor,
            'Bagging': BaggingRegressor,
            'Bayesian Ridge': BayesianRidge,
            'Huber Regressor': HuberRegressor,
            'Gaussian Process': GaussianProcessRegressor
        }

        self.clustering_models = {
            'KMeans': KMeans,
            'DBSCAN': DBSCAN
        }

        self.other_models = {
            'PCA': PCA,
            'Isolation Forest': IsolationForest
        }

        # Add external models if available
        if HAS_XGBOOST:
            self.classification_models['XGBoost'] = xgb.XGBClassifier
            self.regression_models['XGBoost'] = xgb.XGBRegressor

        if HAS_LIGHTGBM:
            self.classification_models['LightGBM'] = lgb.LGBMClassifier
            self.regression_models['LightGBM'] = lgb.LGBMRegressor

        if HAS_CATBOOST:
            self.classification_models['CatBoost'] = cb.CatBoostClassifier
            self.regression_models['CatBoost'] = cb.CatBoostRegressor

    def render_modeling_ui(self):
        """Render the main modeling interface"""

        if not MODELS_AVAILABLE:
            st.error("Required ML libraries are not available. Please install scikit-learn and related packages.")
            return

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Modeling", dataset_names)

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Dataset:** {selected_dataset} ({df.shape[0]} rows × {df.shape[1]} columns)")

        # Tabs for different modeling tasks
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Model Setup",
            "🏃 Training",
            "📊 Results & Comparison",
            "💾 Model Management"
        ])

        with tab1:
            self._render_model_setup(df)

        with tab2:
            self._render_training_ui(df)

        with tab3:
            self._render_results_ui()

        with tab4:
            self._render_model_management()

    def _render_model_setup(self, df):
        """Render model setup interface"""
        st.subheader("🎯 Model Configuration")

        # Task type selection
        task_type = st.selectbox(
            "Select Task Type",
            ["Classification", "Regression", "Clustering", "Dimensionality Reduction", "Anomaly Detection"]
        )

        # Feature and target selection
        if task_type in ["Classification", "Regression"]:
            self._render_supervised_setup(df, task_type)
        elif task_type == "Clustering":
            self._render_clustering_setup(df)
        elif task_type == "Dimensionality Reduction":
            self._render_dimensionality_reduction_setup(df)
        elif task_type == "Anomaly Detection":
            self._render_anomaly_detection_setup(df)

    def _render_supervised_setup(self, df, task_type):
        """Render setup for supervised learning"""

        col1, col2 = st.columns(2)

        with col1:
            # Target column selection
            target_column = st.selectbox(
                "Select Target Column",
                df.columns.tolist(),
                help="The column you want to predict"
            )

            if target_column:
                # Show target column info
                st.markdown("**Target Column Info:**")
                if task_type == "Classification":
                    value_counts = df[target_column].value_counts()
                    st.write(f"Classes: {list(value_counts.index)}")
                    st.write(f"Class distribution: {dict(value_counts)}")
                else:
                    st.write(f"Min: {df[target_column].min():.2f}")
                    st.write(f"Max: {df[target_column].max():.2f}")
                    st.write(f"Mean: {df[target_column].mean():.2f}")
                    st.write(f"Std: {df[target_column].std():.2f}")

        with col2:
            # Feature column selection
            available_features = [col for col in df.columns if col != target_column]

            feature_selection_mode = st.radio(
                "Feature Selection Mode",
                ["Auto (all features)", "Manual selection", "Auto-suggest best features"],
                horizontal=False
            )

            if feature_selection_mode == "Manual selection":
                selected_features = st.multiselect(
                    "Select Feature Columns",
                    available_features,
                    default=available_features[:10] if len(available_features) > 10 else available_features
                )
            elif feature_selection_mode == "Auto-suggest best features":
                n_features = st.slider("Number of features to auto-select", 1, len(available_features),
                                     min(10, len(available_features)))

                if st.button("🎯 Auto-select Best Features"):
                    selected_features = self._auto_select_features(df, target_column, available_features, n_features, task_type)
                    st.session_state.auto_selected_features = selected_features
                    st.success(f"Auto-selected {len(selected_features)} features")
                    st.write("Selected features:", selected_features)

                selected_features = st.session_state.get('auto_selected_features', available_features[:n_features])
            else:
                selected_features = available_features

            st.write(f"Using {len(selected_features)} features")

        # Store selection in session state
        st.session_state.modeling_config = {
            'task_type': task_type,
            'target_column': target_column,
            'feature_columns': selected_features,
            'dataset_name': st.session_state.current_dataset
        }

        # Data preprocessing for modeling
        st.markdown("### 🔧 Preprocessing for Modeling")

        # Check for missing values and categorical columns
        X = df[selected_features]
        y = df[target_column]

        missing_features = X.columns[X.isnull().any()].tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            if missing_features:
                st.warning(f"⚠️ Missing values in: {missing_features}")
                handle_missing = st.selectbox(
                    "Handle Missing Values",
                    ["Drop rows with missing", "Impute with mean/mode", "Keep as-is"]
                )
            else:
                st.success("✅ No missing values in features")
                handle_missing = "Keep as-is"

        with col2:
            if categorical_features:
                st.info(f"ℹ️ Categorical features: {categorical_features}")
                handle_categorical = st.selectbox(
                    "Handle Categorical Features",
                    ["One-hot encoding", "Label encoding", "Keep as-is"]
                )
            else:
                st.success("✅ All features are numeric")
                handle_categorical = "Keep as-is"

        # Store preprocessing options
        st.session_state.modeling_config.update({
            'handle_missing': handle_missing,
            'handle_categorical': handle_categorical,
            'missing_features': missing_features,
            'categorical_features': categorical_features
        })

    def _auto_select_features(self, df, target_column, available_features, n_features, task_type):
        """Auto-select best features using correlation/mutual information"""

        X = df[available_features]
        y = df[target_column]

        # Handle missing values for feature selection
        X_clean = X.fillna(X.mean(numeric_only=True)).fillna(X.mode().iloc[0])

        # Encode categorical variables for feature selection
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        for col in X_clean.select_dtypes(include=['object', 'category']).columns:
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))

        feature_scores = {}

        try:
            if task_type == "Classification":
                from sklearn.feature_selection import mutual_info_classif
                scores = mutual_info_classif(X_clean, y)
            else:
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X_clean, y)

            for i, feature in enumerate(available_features):
                feature_scores[feature] = scores[i]

        except Exception:
            # Fallback to correlation
            if task_type == "Regression":
                correlations = X_clean.corrwith(pd.Series(y))
                for feature in available_features:
                    feature_scores[feature] = abs(correlations.get(feature, 0))
            else:
                # For classification, use basic statistics
                for feature in available_features:
                    feature_scores[feature] = np.random.random()  # Random as fallback

        # Select top N features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        return [feature for feature, score in top_features]

    def _render_clustering_setup(self, df):
        """Render setup for clustering"""

        # Feature selection for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        selected_features = st.multiselect(
            "Select Features for Clustering",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for clustering.")
            return

        # Store configuration
        st.session_state.modeling_config = {
            'task_type': 'Clustering',
            'feature_columns': selected_features,
            'dataset_name': st.session_state.current_dataset
        }

    def _render_dimensionality_reduction_setup(self, df):
        """Render setup for dimensionality reduction"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        selected_features = st.multiselect(
            "Select Features for Dimensionality Reduction",
            numeric_cols,
            default=numeric_cols
        )

        n_components = st.slider(
            "Number of Components",
            min_value=1,
            max_value=min(len(selected_features), 50),
            value=min(2, len(selected_features))
        )

        st.session_state.modeling_config = {
            'task_type': 'Dimensionality Reduction',
            'feature_columns': selected_features,
            'n_components': n_components,
            'dataset_name': st.session_state.current_dataset
        }

    def _render_anomaly_detection_setup(self, df):
        """Render setup for anomaly detection"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        selected_features = st.multiselect(
            "Select Features for Anomaly Detection",
            numeric_cols,
            default=numeric_cols
        )

        contamination = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Expected proportion of anomalies in the dataset"
        )

        st.session_state.modeling_config = {
            'task_type': 'Anomaly Detection',
            'feature_columns': selected_features,
            'contamination': contamination,
            'dataset_name': st.session_state.current_dataset
        }

    def _render_training_ui(self, df):
        """Render training interface"""
        st.subheader("🏃 Model Training")

        if 'modeling_config' not in st.session_state:
            st.warning("Please configure your model in the Model Setup tab first.")
            return

        config = st.session_state.modeling_config
        task_type = config['task_type']

        # Model selection based on task type
        if task_type == "Classification":
            available_models = self.classification_models
        elif task_type == "Regression":
            available_models = self.regression_models
        elif task_type == "Clustering":
            available_models = self.clustering_models
        else:
            available_models = self.other_models

        # Model selection
        col1, col2 = st.columns(2)

        with col1:
            selected_model_names = st.multiselect(
                "Select Models to Train",
                list(available_models.keys()),
                default=list(available_models.keys())[:3]
            )

        with col2:
            # Training configuration
            if task_type in ["Classification", "Regression"]:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("Random State", value=42)

                # Cross-validation
                use_cv = st.checkbox("Use Cross-Validation")
                if use_cv:
                    cv_folds = st.slider("CV Folds", 2, 10, 5)

                # Hyperparameter tuning
                use_tuning = st.checkbox("Hyperparameter Tuning")
                if use_tuning:
                    tuning_method = st.selectbox("Tuning Method", ["Grid Search", "Random Search"])
                    n_iter = st.number_input("Number of Iterations (Random Search)", value=10, min_value=5)

        # Preprocessing data
        if st.button("🚀 Start Training", type="primary"):
            self._train_models(df, config, selected_model_names, locals())

    def _train_models(self, df, config, model_names, training_params):
        """Train selected models"""

        try:
            # Prepare data
            X, y = self._prepare_data(df, config)

            if X is None:
                return

            # Initialize results storage
            if 'training_results' not in st.session_state:
                st.session_state.training_results = []

            task_type = config['task_type']

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, model_name in enumerate(model_names):
                status_text.text(f"Training {model_name}...")

                # Get model class
                if task_type == "Classification":
                    model_class = self.classification_models[model_name]
                elif task_type == "Regression":
                    model_class = self.regression_models[model_name]
                elif task_type == "Clustering":
                    model_class = self.clustering_models[model_name]
                else:
                    model_class = self.other_models[model_name]

                # Train model
                result = self._train_single_model(
                    X, y, model_class, model_name, config, training_params
                )

                if result:
                    st.session_state.training_results.append(result)

                    # Log to history
                    self.history.log_step(
                        "Model Training",
                        f"Trained {model_name} for {task_type}",
                        {
                            "model": model_name,
                            "task_type": task_type,
                            "features": len(config['feature_columns']),
                            "data_points": len(X)
                        },
                        "success"
                    )

                progress_bar.progress((i + 1) / len(model_names))

            status_text.text("Training completed!")
            st.success(f"✅ Successfully trained {len(model_names)} models!")

        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            self.history.log_step(
                "Model Training",
                "Failed model training",
                {"error": str(e)},
                "error"
            )

    def _prepare_data(self, df, config):
        """Prepare data for training"""

        try:
            # Extract features and target
            if config['task_type'] in ["Classification", "Regression"]:
                X = df[config['feature_columns']].copy()
                y = df[config['target_column']].copy()
            else:
                X = df[config['feature_columns']].copy()
                y = None

            # Handle missing values
            if config.get('handle_missing') == "Drop rows with missing":
                if y is not None:
                    mask = X.notna().all(axis=1) & y.notna()
                    X = X[mask]
                    y = y[mask]
                else:
                    X = X.dropna()
            elif config.get('handle_missing') == "Impute with mean/mode":
                # Numeric columns: impute with mean
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

                # Categorical columns: impute with mode
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown')

            # Handle categorical features
            if config.get('handle_categorical') == "One-hot encoding":
                X = pd.get_dummies(X, drop_first=True)
            elif config.get('handle_categorical') == "Label encoding":
                le = LabelEncoder()
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    X[col] = le.fit_transform(X[col].astype(str))

            # Encode target for classification
            if config['task_type'] == "Classification" and y.dtype == 'object':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
                config['target_encoder'] = le  # Store for later use

            return X, y

        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None

    def _train_single_model(self, X, y, model_class, model_name, config, training_params):
        """Train a single model"""

        try:
            task_type = config['task_type']

            # Create model instance with default parameters
            model = model_class(random_state=42)

            start_time = datetime.now()

            if task_type in ["Classification", "Regression"]:
                # Supervised learning
                test_size = training_params.get('test_size', 0.2)
                random_state = training_params.get('random_state', 42)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if task_type == "Classification" and len(np.unique(y)) > 1 else None
                )

                # Hyperparameter tuning
                if training_params.get('use_tuning', False):
                    model = self._tune_hyperparameters(
                        model, X_train, y_train, model_name,
                        training_params.get('tuning_method', 'Grid Search'),
                        training_params.get('n_iter', 10)
                    )

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                train_pred = model.predict(X_train)

                # Calculate metrics
                if task_type == "Classification":
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_train, train_pred)

                    # Add probability predictions if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_prob = model.predict_proba(X_test)
                            metrics['y_prob'] = y_prob
                        except:
                            pass

                else:  # Regression
                    metrics = self._calculate_regression_metrics(y_test, y_pred, y_train, train_pred)

                # Cross-validation
                if training_params.get('use_cv', False):
                    cv_folds = training_params.get('cv_folds', 5)
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                    metrics['cv_scores'] = cv_scores
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()

                # Store predictions for analysis
                metrics.update({
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'X_test': X_test,
                    'X_train': X_train
                })

            else:
                # Unsupervised learning
                model.fit(X)
                metrics = {}

                if task_type == "Clustering":
                    labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
                    metrics['labels'] = labels

                    # Clustering metrics
                    try:
                        from sklearn.metrics import silhouette_score, calinski_harabasz_score
                        metrics['silhouette_score'] = silhouette_score(X, labels)
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
                    except:
                        pass

                elif task_type == "Dimensionality Reduction":
                    transformed = model.transform(X)
                    metrics['transformed_data'] = transformed
                    metrics['explained_variance_ratio'] = getattr(model, 'explained_variance_ratio_', None)

                elif task_type == "Anomaly Detection":
                    anomaly_scores = model.decision_function(X)
                    predictions = model.predict(X)
                    metrics['anomaly_scores'] = anomaly_scores
                    metrics['predictions'] = predictions
                    metrics['n_anomalies'] = np.sum(predictions == -1)

            training_time = (datetime.now() - start_time).total_seconds()

            # Create result dictionary
            result = {
                'model_name': model_name,
                'model': model,
                'task_type': task_type,
                'training_time': training_time,
                'feature_columns': config['feature_columns'],
                'target_column': config.get('target_column'),
                'dataset_name': config['dataset_name'],
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None

    def _tune_hyperparameters(self, model, X_train, y_train, model_name, method, n_iter):
        """Tune hyperparameters for the model"""

        # Define hyperparameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }

        if model_name in param_grids:
            param_grid = param_grids[model_name]

            try:
                if method == "Grid Search":
                    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                else:
                    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=3, n_jobs=-1)

                search.fit(X_train, y_train)
                return search.best_estimator_
            except:
                st.warning(f"Hyperparameter tuning failed for {model_name}, using default parameters.")
                return model

        return model

    def _calculate_classification_metrics(self, y_test, y_pred, y_train, train_pred):
        """Calculate classification metrics"""

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')

        # Training metrics
        metrics['train_accuracy'] = accuracy_score(y_train, train_pred)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
            except:
                pass

        return metrics

    def _calculate_regression_metrics(self, y_test, y_pred, y_train, train_pred):
        """Calculate regression metrics"""

        metrics = {}

        # Basic metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)

        # Training metrics
        metrics['train_mse'] = mean_squared_error(y_train, train_pred)
        metrics['train_r2'] = r2_score(y_train, train_pred)

        return metrics

    def _render_results_ui(self):
        """Render results and comparison interface"""
        st.subheader("📊 Model Results & Comparison")

        if 'training_results' not in st.session_state or not st.session_state.training_results:
            st.info("No trained models available. Please train some models first.")
            return

        results = st.session_state.training_results

        # Results overview
        st.markdown("### 🏆 Model Performance Overview")

        # Create comparison table
        comparison_data = []
        for result in results:
            metrics = result['metrics']

            if result['task_type'] == "Classification":
                comparison_data.append({
                    'Model': result['model_name'],
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{metrics.get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{metrics.get('f1', 0):.4f}",
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'CV Score': f"{metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}" if 'cv_mean' in metrics else "N/A"
                })
            elif result['task_type'] == "Regression":
                comparison_data.append({
                    'Model': result['model_name'],
                    'R²': f"{metrics.get('r2', 0):.4f}",
                    'RMSE': f"{metrics.get('rmse', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'CV Score': f"{metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}" if 'cv_mean' in metrics else "N/A"
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

        # Detailed model analysis
        st.markdown("### 🔍 Detailed Model Analysis")

        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            [r['model_name'] for r in results]
        )

        if selected_model:
            model_result = next(r for r in results if r['model_name'] == selected_model)
            self._render_detailed_model_analysis(model_result)

    def _render_detailed_model_analysis(self, result):
        """Render detailed analysis for a specific model"""

        metrics = result['metrics']
        task_type = result['task_type']

        col1, col2 = st.columns(2)

        with col1:
            # Key metrics
            st.markdown("**📊 Key Metrics:**")

            if task_type == "Classification":
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")

                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

            elif task_type == "Regression":
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                st.metric("MSE", f"{metrics.get('mse', 0):.4f}")

            # Cross-validation results
            if 'cv_scores' in metrics:
                st.markdown("**🔄 Cross-Validation:**")
                cv_scores = metrics['cv_scores']
                st.write(f"Mean: {cv_scores.mean():.4f}")
                st.write(f"Std: {cv_scores.std():.4f}")
                st.write(f"Scores: {[f'{score:.4f}' for score in cv_scores]}")

        with col2:
            # Visualizations
            if task_type == "Classification":
                self._render_classification_plots(metrics)
            elif task_type == "Regression":
                self._render_regression_plots(metrics)

        # Feature importance (if available)
        if hasattr(result['model'], 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")

            importances = result['model'].feature_importances_
            feature_names = result['feature_columns']

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            fig = px.bar(
                importance_df.head(20),  # Top 20 features
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 20 Feature Importances"
            )
            fig.update_layout(height=max(400, len(importance_df.head(20)) * 25))
            st.plotly_chart(fig, use_container_width=True)

        # Model download option
        st.markdown("### 💾 Download Model")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Download Model (Joblib)"):
                model_bytes = io.BytesIO()
                joblib.dump(result['model'], model_bytes)
                model_bytes.seek(0)

                st.download_button(
                    label="Download .joblib file",
                    data=model_bytes.getvalue(),
                    file_name=f"{result['model_name'].replace(' ', '_')}_model.joblib",
                    mime="application/octet-stream"
                )

        with col2:
            if st.button("Download Model (Pickle)"):
                model_bytes = io.BytesIO()
                pickle.dump(result['model'], model_bytes)
                model_bytes.seek(0)

                st.download_button(
                    label="Download .pkl file",
                    data=model_bytes.getvalue(),
                    file_name=f"{result['model_name'].replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )

    def _render_classification_plots(self, metrics):
        """Render classification-specific plots"""

        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            st.markdown("**🔤 Confusion Matrix:**")

            cm = metrics['confusion_matrix']

            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                color_continuous_scale="Blues"
            )
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual")
            st.plotly_chart(fig, use_container_width=True)

        # ROC Curve (for binary classification)
        if 'y_test' in metrics and 'y_prob' in metrics and len(np.unique(metrics['y_test'])) == 2:
            st.markdown("**📈 ROC Curve:**")

            try:
                fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob'][:, 1])
                auc_score = roc_auc_score(metrics['y_test'], metrics['y_prob'][:, 1])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {auc_score:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
                fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(f"Could not generate ROC curve: {str(e)}")

    def _render_regression_plots(self, metrics):
        """Render regression-specific plots"""

        # Actual vs Predicted
        if 'y_test' in metrics and 'y_pred' in metrics:
            st.markdown("**📊 Actual vs Predicted:**")

            fig = go.Figure()

            # Scatter plot
            fig.add_trace(go.Scatter(
                x=metrics['y_test'],
                y=metrics['y_pred'],
                mode='markers',
                name='Predictions',
                opacity=0.6
            ))

            # Perfect prediction line
            min_val = min(metrics['y_test'].min(), metrics['y_pred'].min())
            max_val = max(metrics['y_test'].max(), metrics['y_pred'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Residuals plot
        if 'y_test' in metrics and 'y_pred' in metrics:
            st.markdown("**📉 Residuals Plot:**")

            residuals = metrics['y_test'] - metrics['y_pred']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics['y_pred'],
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")

            fig.update_layout(
                title='Residuals Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals'
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_model_management(self):
        """Render model management interface"""
        st.subheader("💾 Model Management")

        # Import external model
        st.markdown("### 📥 Import Pre-trained Model")

        uploaded_model = st.file_uploader(
            "Upload Pre-trained Model",
            type=['pkl', 'joblib'],
            help="Upload a previously trained scikit-learn model"
        )

        if uploaded_model:
            try:
                if uploaded_model.name.endswith('.pkl'):
                    model = pickle.load(uploaded_model)
                else:
                    model = joblib.load(uploaded_model)

                # Add to session state
                if 'imported_models' not in st.session_state:
                    st.session_state.imported_models = []

                model_info = {
                    'name': uploaded_model.name,
                    'model': model,
                    'import_time': datetime.now().isoformat()
                }

                st.session_state.imported_models.append(model_info)
                st.success(f"✅ Successfully imported model: {uploaded_model.name}")

                # Show model info
                st.write(f"Model type: {type(model).__name__}")
                if hasattr(model, 'feature_importances_'):
                    st.write(f"Number of features: {len(model.feature_importances_)}")

            except Exception as e:
                st.error(f"❌ Error importing model: {str(e)}")

        # Current models summary
        if 'training_results' in st.session_state and st.session_state.training_results:
            st.markdown("### 📊 Current Models Summary")

            summary_data = []
            for result in st.session_state.training_results:
                summary_data.append({
                    'Model': result['model_name'],
                    'Task': result['task_type'],
                    'Dataset': result['dataset_name'],
                    'Features': len(result['feature_columns']),
                    'Training Time': f"{result['training_time']:.2f}s",
                    'Timestamp': result['timestamp'][:19]  # Remove microseconds
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Bulk operations
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🗑️ Clear All Models"):
                    st.session_state.training_results = []
                    st.success("All models cleared!")

            with col2:
                if st.button("📋 Export Model Summary"):
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=summary_csv,
                        file_name=f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
