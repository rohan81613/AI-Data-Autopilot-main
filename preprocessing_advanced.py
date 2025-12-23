"""
Advanced Preprocessing Module with Enhanced Outlier Detection and Feature Engineering
Includes: Standard Deviation, Modified Z-Score, Target Encoding, Feature Selection, Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from category_encoders import TargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False

try:
    from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
    HAS_FEATURE_SELECTION = True
except ImportError:
    HAS_FEATURE_SELECTION = False

from pipeline_history import PipelineHistory


class AdvancedPreprocessing:
    def __init__(self):
        self.history = PipelineHistory()
    
    def render_advanced_outlier_detection(self, df, dataset_name):
        """Enhanced outlier detection with visualization"""
        st.subheader("🔍 Advanced Outlier Detection & Visualization")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns available for outlier detection.")
            return
        
        selected_column = st.selectbox("Select Column for Outlier Analysis", numeric_cols, key="adv_outlier_col")
        
        if selected_column:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced outlier detection methods
                method = st.selectbox(
                    "Outlier Detection Method",
                    [
                        "IQR (Interquartile Range)",
                        "Z-Score",
                        "Standard Deviation",
                        "Modified Z-Score (Robust)",
                        "Isolation Forest",
                        "Percentile Trimming"
                    ],
                    key="adv_outlier_method"
                )
                
                # Method-specific parameters
                if method == "IQR (Interquartile Range)":
                    multiplier = st.slider("IQR Multiplier", 0.5, 3.0, 1.5, 0.1, key="iqr_mult")
                elif method == "Z-Score":
                    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh")
                elif method == "Standard Deviation":
                    std_threshold = st.slider("Std Dev Threshold", 1.0, 5.0, 3.0, 0.1, key="std_thresh")
                elif method == "Modified Z-Score (Robust)":
                    mod_z_threshold = st.slider("Modified Z-Score Threshold", 1.0, 5.0, 3.5, 0.1, key="mod_z_thresh")
                elif method == "Isolation Forest":
                    contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01, key="iso_contam")
                elif method == "Percentile Trimming":
                    lower_pct = st.slider("Lower Percentile", 0.0, 25.0, 5.0, 1.0, key="lower_pct")
                    upper_pct = st.slider("Upper Percentile", 75.0, 100.0, 95.0, 1.0, key="upper_pct")
                
                # Action for outliers
                action = st.selectbox(
                    "Action for Outliers",
                    ["Visualize Only", "Remove", "Cap (Winsorize)", "Transform (Log)", "Mark with Flag"],
                    key="adv_outlier_action"
                )
            
            with col2:
                # Current distribution visualization
                st.markdown("**Current Distribution:**")
                self._plot_distribution(df, selected_column)
            
            # Detect and visualize outliers
            if st.button("🔍 Detect & Visualize Outliers", type="primary", key="detect_adv_outliers"):
                self._detect_and_visualize_outliers(
                    df, dataset_name, selected_column, method, action, locals()
                )

    
    def _plot_distribution(self, df, column):
        """Plot distribution of a column"""
        try:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[column], name="Distribution", nbinsx=30))
            fig.update_layout(
                title=f"Distribution of {column}",
                xaxis_title=column,
                yaxis_title="Frequency",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key=f"dist_{column}")
        except Exception as e:
            st.error(f"Error plotting distribution: {str(e)}")
    
    def _detect_and_visualize_outliers(self, df, dataset_name, column, method, action, params):
        """Detect outliers using various methods and visualize"""
        try:
            data = df[column].dropna()
            outlier_mask = np.zeros(len(df), dtype=bool)
            outlier_indices = []
            lower_bound = None
            upper_bound = None
            
            # Detect outliers based on method
            if method == "IQR (Interquartile Range)":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = params.get('multiplier', 1.5)
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                method_info = f"IQR (multiplier={multiplier})"
            
            elif method == "Z-Score":
                threshold = params.get('threshold', 3.0)
                z_scores = np.abs(stats.zscore(data))
                outlier_indices_temp = np.where(z_scores > threshold)[0]
                outlier_mask[data.index[outlier_indices_temp]] = True
                method_info = f"Z-Score (threshold={threshold})"
            
            elif method == "Standard Deviation":
                std_threshold = params.get('std_threshold', 3.0)
                mean = data.mean()
                std = data.std()
                lower_bound = mean - std_threshold * std
                upper_bound = mean + std_threshold * std
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                method_info = f"Std Dev (threshold={std_threshold}σ)"
            
            elif method == "Modified Z-Score (Robust)":
                mod_z_threshold = params.get('mod_z_threshold', 3.5)
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros(len(data))
                outlier_indices_temp = np.where(np.abs(modified_z_scores) > mod_z_threshold)[0]
                outlier_mask[data.index[outlier_indices_temp]] = True
                method_info = f"Modified Z-Score (threshold={mod_z_threshold})"
            
            elif method == "Isolation Forest":
                contamination = params.get('contamination', 0.1)
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                predictions = iso_forest.fit_predict(df[[column]].dropna())
                outlier_indices_temp = np.where(predictions == -1)[0]
                outlier_mask[df[column].dropna().index[outlier_indices_temp]] = True
                method_info = f"Isolation Forest (contamination={contamination})"
            
            elif method == "Percentile Trimming":
                lower_pct = params.get('lower_pct', 5.0)
                upper_pct = params.get('upper_pct', 95.0)
                lower_bound = data.quantile(lower_pct / 100)
                upper_bound = data.quantile(upper_pct / 100)
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                method_info = f"Percentile ({lower_pct}%-{upper_pct}%)"
            
            outlier_count = outlier_mask.sum()
            
            # Visualize outliers
            st.markdown(f"### 📊 Outlier Visualization - {method_info}")
            st.markdown(f"**Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)**")
            
            # Create visualization
            self._create_outlier_visualization(df, column, outlier_mask, lower_bound, upper_bound, method)
            
            # Apply action if not just visualizing
            if action != "Visualize Only":
                df_modified = self._apply_outlier_action(
                    df, column, outlier_mask, action, lower_bound, upper_bound
                )
                
                # Save modified dataset
                new_dataset_name = f"{dataset_name}_outliers_{action.lower().replace(' ', '_')}"
                st.session_state.datasets[new_dataset_name] = df_modified
                
                # Show before/after comparison
                st.markdown("### 📊 Before/After Comparison")
                self._plot_before_after_comparison(df, df_modified, column)
                
                # Log operation
                self.history.log_step(
                    "Advanced Outlier Detection",
                    f"Applied {method} + {action} to {column}",
                    {
                        "column": column,
                        "method": method_info,
                        "outliers_detected": int(outlier_count),
                        "action": action,
                        "original_rows": len(df),
                        "final_rows": len(df_modified)
                    },
                    "success"
                )
                
                st.success(f"✅ {action} applied to {outlier_count} outliers")
                st.info(f"New dataset created: {new_dataset_name}")
        
        except Exception as e:
            st.error(f"Error in outlier detection: {str(e)}")
            self.history.log_step(
                "Advanced Outlier Detection",
                f"Failed outlier detection on {column}",
                {"error": str(e)},
                "error"
            )

    
    def _create_outlier_visualization(self, df, column, outlier_mask, lower_bound, upper_bound, method):
        """Create comprehensive outlier visualization"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Box Plot with Outliers", "Histogram with Outliers", 
                               "Scatter Plot", "Statistics Comparison"),
                specs=[[{"type": "box"}, {"type": "histogram"}],
                      [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[column], name="All Data", marker_color="lightblue"),
                row=1, col=1
            )
            
            # Histogram with outliers highlighted
            normal_data = df[~outlier_mask][column]
            outlier_data = df[outlier_mask][column]
            
            fig.add_trace(
                go.Histogram(x=normal_data, name="Normal", marker_color="lightblue", nbinsx=30),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(x=outlier_data, name="Outliers", marker_color="red", nbinsx=30),
                row=1, col=2
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df.index[~outlier_mask], 
                    y=df[~outlier_mask][column],
                    mode='markers',
                    name="Normal",
                    marker=dict(color='lightblue', size=5)
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index[outlier_mask], 
                    y=df[outlier_mask][column],
                    mode='markers',
                    name="Outliers",
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=2, col=1
            )
            
            # Add bounds if available
            if lower_bound is not None and upper_bound is not None:
                fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange", row=2, col=1)
                fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange", row=2, col=1)
            
            # Statistics table
            stats_data = [
                ["Metric", "Normal Data", "Outliers"],
                ["Count", f"{(~outlier_mask).sum()}", f"{outlier_mask.sum()}"],
                ["Mean", f"{df[~outlier_mask][column].mean():.2f}", 
                 f"{df[outlier_mask][column].mean():.2f}" if outlier_mask.sum() > 0 else "N/A"],
                ["Median", f"{df[~outlier_mask][column].median():.2f}", 
                 f"{df[outlier_mask][column].median():.2f}" if outlier_mask.sum() > 0 else "N/A"],
                ["Std Dev", f"{df[~outlier_mask][column].std():.2f}", 
                 f"{df[outlier_mask][column].std():.2f}" if outlier_mask.sum() > 0 else "N/A"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=stats_data[0], fill_color='paleturquoise', align='left'),
                    cells=dict(values=list(zip(*stats_data[1:])), fill_color='lavender', align='left')
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, title_text=f"Outlier Analysis: {column}")
            st.plotly_chart(fig, use_container_width=True, key=f"outlier_viz_{column}")
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

    
    def _apply_outlier_action(self, df, column, outlier_mask, action, lower_bound, upper_bound):
        """Apply action to outliers"""
        df_modified = df.copy()
        
        if action == "Remove":
            df_modified = df_modified[~outlier_mask]
        
        elif action == "Cap (Winsorize)":
            if lower_bound is not None and upper_bound is not None:
                df_modified[column] = df_modified[column].clip(lower=lower_bound, upper=upper_bound)
            else:
                # Use percentile-based capping
                lower_bound = df[column].quantile(0.05)
                upper_bound = df[column].quantile(0.95)
                df_modified[column] = df_modified[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif action == "Transform (Log)":
            min_val = df[column].min()
            if min_val <= 0:
                df_modified[column] = np.log1p(df_modified[column] - min_val + 1)
            else:
                df_modified[column] = np.log1p(df_modified[column])
        
        elif action == "Mark with Flag":
            df_modified[f'{column}_is_outlier'] = outlier_mask
        
        return df_modified
    
    def _plot_before_after_comparison(self, df_before, df_after, column):
        """Plot before/after comparison"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Before Treatment", "After Treatment")
            )
            
            # Before
            fig.add_trace(
                go.Histogram(x=df_before[column], name="Before", marker_color="lightcoral", nbinsx=30),
                row=1, col=1
            )
            
            # After
            fig.add_trace(
                go.Histogram(x=df_after[column], name="After", marker_color="lightgreen", nbinsx=30),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, key=f"before_after_{column}")
            
            # Statistics comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Before", f"{df_before[column].mean():.2f}")
                st.metric("Mean After", f"{df_after[column].mean():.2f}")
            with col2:
                st.metric("Std Before", f"{df_before[column].std():.2f}")
                st.metric("Std After", f"{df_after[column].std():.2f}")
            with col3:
                st.metric("Rows Before", len(df_before))
                st.metric("Rows After", len(df_after))
        
        except Exception as e:
            st.error(f"Error in before/after comparison: {str(e)}")

    
    def render_advanced_encoding(self, df, dataset_name):
        """Advanced encoding including Target Encoding"""
        st.subheader("🔤 Advanced Categorical Encoding")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            st.info("No categorical columns available for encoding.")
            return
        
        selected_cols = st.multiselect("Select Categorical Columns", categorical_cols, key="adv_enc_cols")
        
        if selected_cols:
            encoding_method = st.selectbox(
                "Encoding Method",
                [
                    "One-Hot Encoding",
                    "Label Encoding",
                    "Ordinal Encoding",
                    "Target Encoding (Supervised)",
                    "Frequency Encoding",
                    "Binary Encoding"
                ],
                key="adv_enc_method"
            )
            
            # Target encoding requires target column
            target_col = None
            if encoding_method == "Target Encoding (Supervised)":
                if not HAS_TARGET_ENCODER:
                    st.error("Target Encoder not available. Install: pip install category-encoders")
                    return
                
                all_cols = df.columns.tolist()
                target_col = st.selectbox("Select Target Column", all_cols, key="target_enc_col")
            
            # Visualization option
            show_viz = st.checkbox("Show encoding visualization", value=True, key="show_enc_viz")
            
            if st.button("Apply Advanced Encoding", type="primary", key="apply_adv_enc"):
                self._apply_advanced_encoding(
                    df, dataset_name, selected_cols, encoding_method, target_col, show_viz
                )

    
    def _apply_advanced_encoding(self, df, dataset_name, columns, method, target_col, show_viz):
        """Apply advanced encoding methods"""
        try:
            df_encoded = df.copy()
            encoding_info = []
            
            for col in columns:
                original_unique = df[col].nunique()
                
                if method == "One-Hot Encoding":
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    new_cols = len(dummies.columns)
                    encoding_info.append({
                        'column': col,
                        'method': method,
                        'original_unique': original_unique,
                        'new_columns': new_cols
                    })
                
                elif method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df[col].astype(str))
                    encoding_info.append({
                        'column': col,
                        'method': method,
                        'original_unique': original_unique,
                        'new_columns': 1
                    })
                
                elif method == "Ordinal Encoding":
                    from sklearn.preprocessing import OrdinalEncoder
                    oe = OrdinalEncoder()
                    df_encoded[col] = oe.fit_transform(df[[col]])
                    encoding_info.append({
                        'column': col,
                        'method': method,
                        'original_unique': original_unique,
                        'new_columns': 1
                    })
                
                elif method == "Target Encoding (Supervised)" and target_col:
                    if HAS_TARGET_ENCODER:
                        te = TargetEncoder()
                        df_encoded[col] = te.fit_transform(df[col], df[target_col])
                        encoding_info.append({
                            'column': col,
                            'method': method,
                            'original_unique': original_unique,
                            'new_columns': 1,
                            'target': target_col
                        })
                
                elif method == "Frequency Encoding":
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df_encoded[f'{col}_freq'] = df[col].map(freq_map)
                    encoding_info.append({
                        'column': col,
                        'method': method,
                        'original_unique': original_unique,
                        'new_columns': 1
                    })
                
                elif method == "Binary Encoding":
                    # Simple binary encoding for binary categories
                    if original_unique == 2:
                        df_encoded[col] = (df[col] == df[col].unique()[0]).astype(int)
                        encoding_info.append({
                            'column': col,
                            'method': method,
                            'original_unique': original_unique,
                            'new_columns': 1
                        })
            
            # Save encoded dataset
            new_dataset_name = f"{dataset_name}_advanced_encoded"
            st.session_state.datasets[new_dataset_name] = df_encoded
            
            # Show encoding summary
            st.success(f"✅ Advanced encoding completed!")
            st.info(f"New dataset created: {new_dataset_name}")
            
            encoding_df = pd.DataFrame(encoding_info)
            st.dataframe(encoding_df, use_container_width=True)
            
            # Visualization
            if show_viz:
                self._visualize_encoding(df, df_encoded, columns[0] if columns else None, method)
            
            # Log operation
            self.history.log_step(
                "Advanced Encoding",
                f"Applied {method} to {len(columns)} columns",
                {
                    "columns": columns,
                    "method": method,
                    "original_shape": df.shape,
                    "encoded_shape": df_encoded.shape
                },
                "success"
            )
        
        except Exception as e:
            st.error(f"Error in advanced encoding: {str(e)}")
            self.history.log_step(
                "Advanced Encoding",
                f"Failed encoding",
                {"error": str(e)},
                "error"
            )

    
    def _visualize_encoding(self, df_before, df_after, column, method):
        """Visualize encoding transformation"""
        try:
            if column and column in df_before.columns:
                st.markdown(f"### 📊 Encoding Visualization: {column}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before Encoding:**")
                    value_counts = df_before[column].value_counts().head(10)
                    fig1 = px.bar(x=value_counts.index.astype(str), y=value_counts.values,
                                 labels={'x': column, 'y': 'Count'},
                                 title=f"Original {column}")
                    st.plotly_chart(fig1, use_container_width=True, key=f"enc_before_{column}")
                
                with col2:
                    st.markdown("**After Encoding:**")
                    # Show encoded column distribution
                    if column in df_after.columns:
                        fig2 = px.histogram(df_after, x=column, title=f"Encoded {column}")
                        st.plotly_chart(fig2, use_container_width=True, key=f"enc_after_{column}")
                    else:
                        st.info("Column was one-hot encoded into multiple columns")
        
        except Exception as e:
            st.error(f"Error in encoding visualization: {str(e)}")

    
    def render_feature_selection(self, df, dataset_name):
        """Advanced feature selection methods"""
        st.subheader("🎯 Advanced Feature Selection")
        
        if not HAS_FEATURE_SELECTION:
            st.error("Feature selection libraries not available. Install scikit-learn.")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for feature selection.")
            return
        
        # Select target column
        target_col = st.selectbox("Select Target Column", df.columns.tolist(), key="fs_target")
        
        if target_col:
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            if not feature_cols:
                st.warning("No feature columns available after excluding target.")
                return
            
            st.markdown(f"**Available Features:** {len(feature_cols)}")
            
            # Feature selection method
            fs_method = st.selectbox(
                "Feature Selection Method",
                [
                    "Recursive Feature Elimination (RFE)",
                    "SelectKBest (F-statistic)",
                    "SelectKBest (Mutual Information)",
                    "Feature Importance (Random Forest)",
                    "Correlation-based Selection"
                ],
                key="fs_method"
            )
            
            # Number of features to select
            n_features = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=len(feature_cols),
                value=min(10, len(feature_cols)),
                key="fs_n_features"
            )
            
            # Task type
            task_type = st.radio("Task Type", ["Classification", "Regression"], key="fs_task")
            
            if st.button("🎯 Perform Feature Selection", type="primary", key="perform_fs"):
                self._perform_feature_selection(
                    df, dataset_name, feature_cols, target_col, fs_method, n_features, task_type
                )

    
    def _perform_feature_selection(self, df, dataset_name, feature_cols, target_col, method, n_features, task_type):
        """Perform feature selection"""
        try:
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col]
            
            selected_features = []
            feature_scores = {}
            
            if method == "Recursive Feature Elimination (RFE)":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if task_type == "Classification":
                    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
                rfe = RFE(estimator, n_features_to_select=n_features)
                rfe.fit(X, y)
                
                selected_features = [feature_cols[i] for i, selected in enumerate(rfe.support_) if selected]
                feature_scores = {feature_cols[i]: rfe.ranking_[i] for i in range(len(feature_cols))}
            
            elif method == "SelectKBest (F-statistic)":
                if task_type == "Classification":
                    selector = SelectKBest(f_classif, k=n_features)
                else:
                    selector = SelectKBest(f_regression, k=n_features)
                
                selector.fit(X, y)
                selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support()) if selected]
                feature_scores = {feature_cols[i]: selector.scores_[i] for i in range(len(feature_cols))}
            
            elif method == "SelectKBest (Mutual Information)":
                if task_type == "Classification":
                    selector = SelectKBest(mutual_info_classif, k=n_features)
                else:
                    selector = SelectKBest(mutual_info_regression, k=n_features)
                
                selector.fit(X, y)
                selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support()) if selected]
                feature_scores = {feature_cols[i]: selector.scores_[i] for i in range(len(feature_cols))}
            
            elif method == "Feature Importance (Random Forest)":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if task_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X, y)
                importances = model.feature_importances_
                feature_scores = {feature_cols[i]: importances[i] for i in range(len(feature_cols))}
                
                # Select top N features
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:n_features]]
            
            elif method == "Correlation-based Selection":
                correlations = X.corrwith(y).abs()
                feature_scores = correlations.to_dict()
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:n_features]]
            
            # Create new dataset with selected features
            df_selected = df[selected_features + [target_col]].copy()
            new_dataset_name = f"{dataset_name}_feature_selected"
            st.session_state.datasets[new_dataset_name] = df_selected
            
            # Display results
            st.success(f"✅ Feature selection completed!")
            st.info(f"New dataset created: {new_dataset_name} with {len(selected_features)} features")
            
            # Show selected features
            st.markdown("### 📊 Selected Features")
            st.write(selected_features)
            
            # Visualize feature scores
            self._visualize_feature_scores(feature_scores, selected_features, method)
            
            # Log operation
            self.history.log_step(
                "Feature Selection",
                f"Applied {method} to select {n_features} features",
                {
                    "method": method,
                    "original_features": len(feature_cols),
                    "selected_features": len(selected_features),
                    "features": selected_features
                },
                "success"
            )
        
        except Exception as e:
            st.error(f"Error in feature selection: {str(e)}")
            self.history.log_step(
                "Feature Selection",
                f"Failed feature selection",
                {"error": str(e)},
                "error"
            )

    
    def _visualize_feature_scores(self, feature_scores, selected_features, method):
        """Visualize feature importance scores"""
        try:
            st.markdown(f"### 📊 Feature Scores - {method}")
            
            # Sort by score
            sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            features = [f[0] for f in sorted_scores]
            scores = [f[1] for f in sorted_scores]
            
            # Color selected features differently
            colors = ['green' if f in selected_features else 'lightblue' for f in features]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=scores,
                    y=features,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{s:.4f}" for s in scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Feature Scores ({method})",
                xaxis_title="Score",
                yaxis_title="Feature",
                height=max(400, len(features) * 20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"fs_viz_{method}")
            
            # Show score table
            score_df = pd.DataFrame(sorted_scores, columns=['Feature', 'Score'])
            score_df['Selected'] = score_df['Feature'].isin(selected_features)
            st.dataframe(score_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error visualizing feature scores: {str(e)}")
    
    def render_scaling_visualization(self, df, dataset_name):
        """Visualize scaling transformations"""
        st.subheader("📊 Scaling & Encoding Visualization")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns available for scaling visualization.")
            return
        
        selected_col = st.selectbox("Select Column to Visualize Scaling", numeric_cols, key="scale_viz_col")
        
        if selected_col:
            scalers = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
                "QuantileTransformer": QuantileTransformer()
            }
            
            if st.button("📊 Show Scaling Comparison", key="show_scale_comp"):
                self._compare_scaling_methods(df, selected_col, scalers)

    
    def _compare_scaling_methods(self, df, column, scalers):
        """Compare different scaling methods visually"""
        try:
            st.markdown(f"### 📊 Scaling Comparison: {column}")
            
            # Create subplots for each scaler
            n_scalers = len(scalers) + 1  # +1 for original
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=["Original"] + list(scalers.keys()),
                specs=[[{"type": "histogram"}] * 3, [{"type": "histogram"}] * 3]
            )
            
            # Original data
            fig.add_trace(
                go.Histogram(x=df[column], name="Original", marker_color="lightblue", nbinsx=30),
                row=1, col=1
            )
            
            # Scaled data
            positions = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            for idx, (scaler_name, scaler) in enumerate(scalers.items()):
                if idx < len(positions):
                    scaled_data = scaler.fit_transform(df[[column]])
                    row, col = positions[idx]
                    
                    fig.add_trace(
                        go.Histogram(x=scaled_data.flatten(), name=scaler_name, 
                                   marker_color="lightgreen", nbinsx=30),
                        row=row, col=col
                    )
            
            fig.update_layout(height=800, showlegend=False, title_text=f"Scaling Methods Comparison: {column}")
            st.plotly_chart(fig, use_container_width=True, key=f"scale_comp_{column}")
            
            # Statistics comparison table
            st.markdown("### 📊 Statistics Comparison")
            
            stats_data = {
                'Method': ['Original'],
                'Mean': [df[column].mean()],
                'Std': [df[column].std()],
                'Min': [df[column].min()],
                'Max': [df[column].max()]
            }
            
            for scaler_name, scaler in scalers.items():
                scaled_data = scaler.fit_transform(df[[column]]).flatten()
                stats_data['Method'].append(scaler_name)
                stats_data['Mean'].append(scaled_data.mean())
                stats_data['Std'].append(scaled_data.std())
                stats_data['Min'].append(scaled_data.min())
                stats_data['Max'].append(scaled_data.max())
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error comparing scaling methods: {str(e)}")
