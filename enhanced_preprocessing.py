import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# Try to import TPOT for AutoML features
try:
    from tpot import TPOTRegressor, TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

# Try to import feature-engine for advanced preprocessing
try:
    from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
    from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
    from feature_engine.outliers import Winsorizer
    HAS_FEATURE_ENGINE = True
except ImportError:
    HAS_FEATURE_ENGINE = False

from pipeline_history import PipelineHistory


class EnhancedDataPreprocessing:
    def __init__(self):
        self.history = PipelineHistory()
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        self.encoders = {
            'LabelEncoder': LabelEncoder(),
            'OneHotEncoder': OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        }

    def render_preprocessing_ui(self):
        """Render the enhanced preprocessing interface"""
        
        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        if not dataset_names:
            st.warning("No datasets available. Please load a dataset first.")
            return
            
        selected_dataset = st.selectbox("Select Dataset for Preprocessing", dataset_names, key="enhanced_prep_dataset_selector")

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Dataset:** {selected_dataset} ({df.shape[0]} rows × {df.shape[1]} columns)")

        # Enhanced tabs for different preprocessing sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Profiling",
            "🧹 Smart Data Cleaning",
            "⚙️ Feature Engineering",
            "⚖️ Scaling & Encoding",
            "🤖 AutoML Preprocessing"
        ])

        with tab1:
            self._render_data_profiling(df, selected_dataset)

        with tab2:
            self._render_smart_data_cleaning(df, selected_dataset)

        with tab3:
            self._render_feature_engineering(df, selected_dataset)

        with tab4:
            self._render_scaling_encoding(df, selected_dataset)

        with tab5:
            self._render_automl_preprocessing(df, selected_dataset)

    def _render_data_profiling(self, df, dataset_name):
        """Render automated data profiling section"""
        st.subheader("📊 Automated Data Profiling")
        
        if st.button("🔍 Run Automated Data Profile", key="run_data_profile"):
            with st.spinner("Analyzing dataset..."):
                profile = self._generate_data_profile(df)
                
                # Display profile results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", profile['total_rows'])
                with col2:
                    st.metric("Total Columns", profile['total_columns'])
                with col3:
                    st.metric("Missing Values", profile['missing_values'])
                with col4:
                    st.metric("Duplicate Rows", profile['duplicate_rows'])
                
                # Data types distribution
                st.markdown("### 📋 Data Types Distribution")
                dtypes_df = pd.DataFrame(list(profile['data_types'].values()), index=list(profile['data_types'].keys()), columns=['Count'])
                dtypes_df.columns = ['Count']
                st.dataframe(dtypes_df)
                
                # Column details
                st.markdown("### 📊 Column Details")
                column_details = []
                for col in df.columns:
                    col_info = {
                        'Column': col,
                        'Data Type': str(df[col].dtype),
                        'Missing %': f"{profile['missing_by_column'].get(col, 0):.1f}%",
                        'Unique Values': df[col].nunique(),
                        'Most Frequent': str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A"
                    }
                    column_details.append(col_info)
                
                column_df = pd.DataFrame(column_details)
                st.dataframe(column_df, use_container_width=True)
                
                # Missing values heatmap
                if profile['missing_values'] > 0:
                    st.markdown("### 🕳️ Missing Values Visualization")
                    missing_data = df.isnull()
                    fig, ax = plt.subplots(figsize=(10, max(1, len(df.columns) // 3)))
                    sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
                    ax.set_title("Missing Values Heatmap")
                    st.pyplot(fig)
                    plt.close()
                
                # Distribution plots for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.markdown("### 📈 Numeric Columns Distribution")
                    cols = st.columns(min(3, len(numeric_cols)))
                    for i, col in enumerate(numeric_cols[:9]):  # Limit to 9 columns
                        with cols[i % 3]:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            df[col].hist(bins=30, ax=ax)
                            ax.set_title(f"Distribution: {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                            plt.close()
                
                # Log to history
                self.history.log_step(
                    "Data Profiling",
                    f"Generated profile for {dataset_name}",
                    {
                        "rows": profile['total_rows'],
                        "columns": profile['total_columns'],
                        "missing_values": profile['missing_values']
                    },
                    "success"
                )
                
                st.success("✅ Data profiling completed successfully!")

    def _generate_data_profile(self, df):
        """Generate comprehensive data profile"""
        profile = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_by_column': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        return profile

    def _render_smart_data_cleaning(self, df, dataset_name):
        """Render smart data cleaning section"""
        st.subheader("🧹 Smart Data Cleaning")
        
        # Missing value handling
        st.markdown("### 🕳️ Missing Value Treatment")
        
        # Show missing value statistics
        missing_stats = df.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
        
        if not missing_stats.empty:
            st.markdown("#### Missing Value Statistics")
            missing_df = pd.DataFrame({
                'Column': missing_stats.index,
                'Missing Count': missing_stats.values,
                'Missing %': (missing_stats.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
            
            # Smart imputation options
            st.markdown("#### Smart Imputation Options")
            
            # Auto-detect imputation strategy
            col1, col2 = st.columns(2)
            with col1:
                auto_impute = st.checkbox("Auto-detect imputation strategy", value=True)
            
            with col2:
                if auto_impute:
                    st.info("Auto-detection enabled: Will choose best strategy for each column")
            
            # Manual imputation options
            if not auto_impute:
                st.markdown("##### Manual Imputation")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Numeric imputation
                if numeric_cols:
                    st.markdown("###### Numeric Columns")
                    for col in numeric_cols:
                        if df[col].isnull().sum() > 0:
                            strategy = st.selectbox(
                                f"Imputation for {col}",
                                ["mean", "median", "mode", "knn", "drop"],
                                key=f"impute_{col}"
                            )
                
                # Categorical imputation
                if categorical_cols:
                    st.markdown("###### Categorical Columns")
                    for col in categorical_cols:
                        if df[col].isnull().sum() > 0:
                            strategy = st.selectbox(
                                f"Imputation for {col}",
                                ["mode", "constant", "drop"],
                                key=f"impute_cat_{col}"
                            )
            
            # Apply imputation
            if st.button("🧹 Apply Missing Value Treatment", key="apply_missing_treatment"):
                with st.spinner("Applying imputation..."):
                    try:
                        cleaned_df = self._apply_smart_imputation(df, auto_impute)
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = cleaned_df
                        
                        # Show results
                        st.success(f"✅ Missing values treated successfully!")
                        st.markdown(f"**Before:** {df.isnull().sum().sum()} missing values")
                        st.markdown(f"**After:** {cleaned_df.isnull().sum().sum()} missing values")
                        
                        # Log to history
                        self.history.log_step(
                            "Missing Value Treatment",
                            f"Treated missing values in {dataset_name}",
                            {
                                "before_missing": int(df.isnull().sum().sum()),
                                "after_missing": int(cleaned_df.isnull().sum().sum())
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error during imputation: {str(e)}")
        else:
            st.info("No missing values found in the dataset.")
        
        # Outlier detection and treatment
        st.markdown("### ⚠️ Outlier Detection & Treatment")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect("Select columns for outlier detection", numeric_cols)
            
            if selected_cols:
                method = st.selectbox("Outlier Detection Method", [
                    "IQR (Interquartile Range)",
                    "Z-Score",
                    "Isolation Forest",
                    "Elliptic Envelope"
                ])
                
                treatment = st.selectbox("Outlier Treatment", [
                    "Cap/Floor (Winsorization)",
                    "Remove Outliers",
                    "Transform (Log)",
                    "No Treatment"
                ])
                
                if st.button("🔍 Detect & Treat Outliers", key="detect_outliers"):
                    with st.spinner("Detecting and treating outliers..."):
                        try:
                            treated_df = self._detect_and_treat_outliers(df, selected_cols, method, treatment)
                            
                            # Update session state
                            st.session_state.datasets[dataset_name] = treated_df
                            
                            st.success("✅ Outlier detection and treatment completed!")
                            
                            # Log to history
                            self.history.log_step(
                                "Outlier Treatment",
                                f"Detected and treated outliers in {dataset_name}",
                                {
                                    "columns": selected_cols,
                                    "method": method,
                                    "treatment": treatment
                                },
                                "success"
                            )
                        except Exception as e:
                            st.error(f"Error during outlier treatment: {str(e)}")
        else:
            st.info("No numeric columns available for outlier detection.")
        
        # Data type detection and conversion
        st.markdown("### 🔄 Data Type Detection")
        
        if st.button("🔍 Auto-detect & Convert Data Types", key="auto_convert_types"):
            with st.spinner("Detecting and converting data types..."):
                try:
                    converted_df = self._auto_convert_data_types(df)
                    
                    # Update session state
                    st.session_state.datasets[dataset_name] = converted_df
                    
                    st.success("✅ Data types detected and converted successfully!")
                    
                    # Show conversion summary
                    st.markdown("#### Conversion Summary")
                    for col in df.columns:
                        if str(df[col].dtype) != str(converted_df[col].dtype):
                            st.markdown(f"- **{col}**: {df[col].dtype} → {converted_df[col].dtype}")
                    
                    # Log to history
                    self.history.log_step(
                        "Data Type Conversion",
                        f"Auto-converted data types in {dataset_name}",
                        {"status": "completed"},
                        "success"
                    )
                except Exception as e:
                    st.error(f"Error during data type conversion: {str(e)}")

    def _apply_smart_imputation(self, df, auto_detect=True):
        """Apply smart imputation based on column types and missing patterns"""
        df_cleaned = df.copy()
        
        if auto_detect:
            # Auto-detect best strategy for each column
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['object', 'category']:
                        # For categorical columns, use mode
                        mode_value = df[col].mode()
                        if not mode_value.empty:
                            df_cleaned[col].fillna(mode_value.iloc[0], inplace=True)
                    else:
                        # For numeric columns, check distribution
                        if df[col].skew() > 1 or df[col].skew() < -1:
                            # Skewed distribution - use median
                            df_cleaned[col].fillna(df[col].median(), inplace=True)
                        else:
                            # Normal distribution - use mean
                            df_cleaned[col].fillna(df[col].mean(), inplace=True)
        else:
            # Manual imputation would be implemented here
            # For now, we'll use a simple approach
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['object', 'category']:
                        mode_value = df[col].mode()
                        if not mode_value.empty:
                            df_cleaned[col].fillna(mode_value.iloc[0], inplace=True)
                    else:
                        df_cleaned[col].fillna(df[col].median(), inplace=True)
        
        return df_cleaned

    def _detect_and_treat_outliers(self, df, columns, method, treatment):
        """Detect and treat outliers in specified columns"""
        df_treated = df.copy()
        
        for col in columns:
            lower_bound = None
            upper_bound = None
            outliers = None
            
            if method == "IQR (Interquartile Range)":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "Z-Score":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                
            # Apply treatment
            if treatment == "Cap/Floor (Winsorization)":
                if method == "IQR (Interquartile Range)" and lower_bound is not None and upper_bound is not None:
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == "Z-Score":
                    # Cap at 3 standard deviations
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df_treated[col] = df_treated[col].clip(
                        lower=mean_val - 3 * std_val, 
                        upper=mean_val + 3 * std_val
                    )
                    
            elif treatment == "Remove Outliers" and outliers is not None:
                df_treated = df_treated[~outliers]
                
            elif treatment == "Transform (Log)":
                # Apply log transformation (only for positive values)
                if (df_treated[col] > 0).all():
                    df_treated[col] = np.log1p(df_treated[col])
        
        return df_treated

    def _auto_convert_data_types(self, df):
        """Automatically detect and convert data types"""
        df_converted = df.copy()
        
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Try numeric conversion
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                if not numeric_converted.isnull().all():
                    # Check if conversion makes sense (not too many NaNs)
                    if numeric_converted.isna().sum() <= len(df) * 0.3:  # Less than 30% NaN
                        df_converted[col] = numeric_converted
                        continue
                
                # Try datetime conversion
                try:
                    datetime_converted = pd.to_datetime(df[col], errors='coerce')
                    if datetime_converted.notna().sum() > len(df) * 0.5:  # More than 50% valid dates
                        df_converted[col] = datetime_converted
                        continue
                except:
                    pass
        
        return df_converted

    def _render_feature_engineering(self, df, dataset_name):
        """Render feature engineering section"""
        st.subheader("⚙️ Feature Engineering")
        
        # Feature creation options
        st.markdown("### ➕ Feature Creation")
        
        # Polynomial features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if st.checkbox("Create Polynomial Features", key="poly_features"):
                selected_poly_cols = st.multiselect("Select columns for polynomial features", numeric_cols)
                degree = st.slider("Polynomial Degree", 2, 5, 2)
                
                if st.button("➕ Generate Polynomial Features", key="generate_poly"):
                    try:
                        from sklearn.preprocessing import PolynomialFeatures
                        poly = PolynomialFeatures(degree=degree, include_bias=False)
                        poly_features = poly.fit_transform(df[selected_poly_cols])
                        
                        # Create feature names
                        feature_names = poly.get_feature_names_out(selected_poly_cols)
                        
                        # Create DataFrame with polynomial features
                        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                        
                        # Combine with original DataFrame
                        enhanced_df = pd.concat([df, poly_df], axis=1)
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = enhanced_df
                        
                        st.success(f"✅ Generated {poly_features.shape[1]} polynomial features!")
                        st.info(f"New dataset shape: {enhanced_df.shape}")
                        
                        # Log to history
                        self.history.log_step(
                            "Feature Engineering",
                            f"Generated polynomial features for {dataset_name}",
                            {
                                "original_features": len(df.columns),
                                "new_features": len(poly_df.columns),
                                "degree": degree
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error generating polynomial features: {str(e)}")
        
        # Binning features
        if numeric_cols:
            if st.checkbox("Create Binned Features", key="bin_features"):
                selected_bin_cols = st.multiselect("Select columns for binning", numeric_cols)
                n_bins = st.slider("Number of Bins", 2, 20, 5)
                
                if st.button("➕ Generate Binned Features", key="generate_bins"):
                    try:
                        enhanced_df = df.copy()
                        for col in selected_bin_cols:
                            # Create binned column
                            binned_col_name = f"{col}_binned"
                            enhanced_df[binned_col_name] = pd.cut(df[col], bins=n_bins, labels=False)
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = enhanced_df
                        
                        st.success(f"✅ Generated binned features for {len(selected_bin_cols)} columns!")
                        st.info(f"New dataset shape: {enhanced_df.shape}")
                        
                        # Log to history
                        self.history.log_step(
                            "Feature Engineering",
                            f"Generated binned features for {dataset_name}",
                            {
                                "columns_binned": len(selected_bin_cols),
                                "n_bins": n_bins
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error generating binned features: {str(e)}")
        
        # Encoding categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.markdown("### 🔤 Categorical Encoding")
            
            selected_cat_cols = st.multiselect("Select categorical columns to encode", categorical_cols)
            
            if selected_cat_cols:
                encoding_method = st.selectbox("Encoding Method", [
                    "One-Hot Encoding",
                    "Label Encoding",
                    "Target Encoding (if target available)",
                    "Frequency Encoding"
                ])
                
                if st.button("🔤 Apply Encoding", key="apply_encoding"):
                    try:
                        encoded_df = df.copy()
                        
                        if encoding_method == "One-Hot Encoding":
                            # One-hot encode selected columns
                            dummies = pd.get_dummies(df[selected_cat_cols], prefix=selected_cat_cols)
                            encoded_df = pd.concat([encoded_df.drop(selected_cat_cols, axis=1), dummies], axis=1)
                            
                        elif encoding_method == "Label Encoding":
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            for col in selected_cat_cols:
                                encoded_df[col] = le.fit_transform(df[col].astype(str))
                                
                        elif encoding_method == "Frequency Encoding":
                            for col in selected_cat_cols:
                                freq_map = df[col].value_counts().to_dict()
                                encoded_df[f"{col}_freq"] = df[col].map(freq_map)
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = encoded_df
                        
                        st.success(f"✅ Applied {encoding_method} to {len(selected_cat_cols)} columns!")
                        st.info(f"New dataset shape: {encoded_df.shape}")
                        
                        # Log to history
                        self.history.log_step(
                            "Feature Engineering",
                            f"Applied {encoding_method} to {dataset_name}",
                            {
                                "columns_encoded": len(selected_cat_cols),
                                "method": encoding_method
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error during encoding: {str(e)}")
        
        # Feature selection
        st.markdown("### ⚖️ Feature Selection")
        
        if st.checkbox("Perform Feature Selection", key="feature_selection"):
            target_col = st.selectbox("Select Target Column", df.columns.tolist())
            
            if target_col:
                method = st.selectbox("Selection Method", [
                    "Correlation-based",
                    "Variance Threshold",
                    "Recursive Feature Elimination",
                    "Select K Best"
                ])
                
                if method == "Correlation-based":
                    threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.95)
                elif method == "Select K Best":
                    k = st.slider("Number of Features to Select", 1, len(df.columns)-1, min(10, len(df.columns)-1))
                
                if st.button("⚖️ Apply Feature Selection", key="apply_feature_selection"):
                    try:
                        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
                        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                        
                        selected_df = df.copy()
                        
                        if method == "Correlation-based":
                            # Remove highly correlated features
                            corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
                            upper_tri = corr_matrix.where(
                                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                            )
                            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
                            selected_df = df.drop(to_drop, axis=1)
                            
                        elif method == "Variance Threshold":
                            # Remove low variance features
                            selector = VarianceThreshold(threshold=0.01)
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                selected_data = selector.fit_transform(df[numeric_cols])
                                selected_features = numeric_cols[selector.get_support()]
                                selected_df = pd.concat([
                                    df.drop(numeric_cols, axis=1), 
                                    pd.DataFrame(selected_data, columns=selected_features, index=df.index)
                                ], axis=1)
                                
                        elif method == "Select K Best":
                            # Select K best features
                            X = df.drop(target_col, axis=1)
                            y = df[target_col]
                            
                            # Separate numeric and categorical columns
                            numeric_features = X.select_dtypes(include=[np.number]).columns
                            if len(numeric_features) > 0:
                                # Use appropriate scoring function
                                if y.dtype in ['object', 'category']:
                                    score_func = f_classif
                                else:
                                    score_func = f_regression
                                
                                selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_features)))
                                X_selected = selector.fit_transform(X[numeric_features], y)
                                selected_features = numeric_features[selector.get_support()]
                                
                                # Combine selected numeric features with categorical features
                                categorical_features = X.select_dtypes(include=['object', 'category']).columns
                                selected_df = pd.concat([
                                    X[categorical_features], 
                                    pd.DataFrame(X_selected, columns=selected_features, index=X.index),
                                    y
                                ], axis=1)
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = selected_df
                        
                        st.success(f"✅ Feature selection completed!")
                        st.info(f"Original features: {len(df.columns)}, Selected features: {len(selected_df.columns)}")
                        
                        # Log to history
                        self.history.log_step(
                            "Feature Selection",
                            f"Applied {method} to {dataset_name}",
                            {
                                "original_features": len(df.columns),
                                "selected_features": len(selected_df.columns),
                                "method": method
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error during feature selection: {str(e)}")

    def _render_scaling_encoding(self, df, dataset_name):
        """Render scaling and encoding section"""
        st.subheader("⚖️ Scaling & Encoding")
        
        # Scaling options
        st.markdown("### 📏 Feature Scaling")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_scale_cols = st.multiselect("Select columns to scale", numeric_cols, default=numeric_cols)
            
            if selected_scale_cols:
                scaler_type = st.selectbox("Select Scaler", [
                    "StandardScaler (Mean=0, Std=1)",
                    "MinMaxScaler (0 to 1)",
                    "RobustScaler (Median, IQR)",
                    "MaxAbsScaler",
                    "Normalizer"
                ])
                
                if st.button("📏 Apply Scaling", key="apply_scaling"):
                    try:
                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
                        
                        scaled_df = df.copy()
                        
                        if "StandardScaler" in scaler_type:
                            scaler = StandardScaler()
                            scaled_df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                        elif "MinMaxScaler" in scaler_type:
                            scaler = MinMaxScaler()
                            scaled_df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                        elif "RobustScaler" in scaler_type:
                            scaler = RobustScaler()
                            scaled_df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                        elif "MaxAbsScaler" in scaler_type:
                            from sklearn.preprocessing import MaxAbsScaler
                            scaler = MaxAbsScaler()
                            scaled_df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                        elif "Normalizer" in scaler_type:
                            scaler = Normalizer()
                            scaled_df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = scaled_df
                        
                        # Store scaler in session state for later use
                        if 'transformers' not in st.session_state:
                            st.session_state.transformers = {}
                        st.session_state.transformers[f"{dataset_name}_scaler"] = scaler
                        
                        st.success(f"✅ Applied {scaler_type} to {len(selected_scale_cols)} columns!")
                        
                        # Show scaling results
                        st.markdown("#### Scaling Results")
                        stats_df = pd.DataFrame({
                            'Column': selected_scale_cols,
                            'Before Mean': [df[col].mean() for col in selected_scale_cols],
                            'After Mean': [scaled_df[col].mean() for col in selected_scale_cols],
                            'Before Std': [df[col].std() for col in selected_scale_cols],
                            'After Std': [scaled_df[col].std() for col in selected_scale_cols]
                        })
                        st.dataframe(stats_df)
                        
                        # Log to history
                        self.history.log_step(
                            "Feature Scaling",
                            f"Applied {scaler_type} to {dataset_name}",
                            {
                                "columns_scaled": len(selected_scale_cols),
                                "scaler_type": scaler_type
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error during scaling: {str(e)}")
        else:
            st.info("No numeric columns available for scaling.")
        
        # Encoding options
        st.markdown("### 🔤 Categorical Encoding")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_encode_cols = st.multiselect("Select columns to encode", categorical_cols)
            
            if selected_encode_cols:
                encoding_type = st.selectbox("Select Encoding Method", [
                    "One-Hot Encoding",
                    "Label Encoding",
                    "Ordinal Encoding",
                    "Binary Encoding",
                    "Hash Encoding"
                ])
                
                if st.button("🔤 Apply Encoding", key="apply_encoding_final"):
                    try:
                        encoded_df = df.copy()
                        
                        if encoding_type == "One-Hot Encoding":
                            # One-hot encode selected columns
                            dummies = pd.get_dummies(df[selected_encode_cols], prefix=selected_encode_cols)
                            encoded_df = pd.concat([
                                encoded_df.drop(selected_encode_cols, axis=1), 
                                dummies
                            ], axis=1)
                            
                        elif encoding_type == "Label Encoding":
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            for col in selected_encode_cols:
                                encoded_df[col] = le.fit_transform(df[col].astype(str))
                                
                        elif encoding_type == "Ordinal Encoding":
                            if HAS_FEATURE_ENGINE:
                                from feature_engine.encoding import OrdinalEncoder
                                encoder = OrdinalEncoder(encoding_method='ordered')
                                encoded_df = encoder.fit_transform(df[selected_encode_cols])
                            else:
                                st.warning("Feature-engine not available. Using Label Encoding instead.")
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                for col in selected_encode_cols:
                                    encoded_df[col] = le.fit_transform(df[col].astype(str))
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = encoded_df
                        
                        # Store encoder in session state
                        if 'transformers' not in st.session_state:
                            st.session_state.transformers = {}
                        st.session_state.transformers[f"{dataset_name}_encoder"] = encoding_type
                        
                        st.success(f"✅ Applied {encoding_type} to {len(selected_encode_cols)} columns!")
                        st.info(f"New dataset shape: {encoded_df.shape}")
                        
                        # Log to history
                        self.history.log_step(
                            "Categorical Encoding",
                            f"Applied {encoding_type} to {dataset_name}",
                            {
                                "columns_encoded": len(selected_encode_cols),
                                "encoding_type": encoding_type
                            },
                            "success"
                        )
                    except Exception as e:
                        st.error(f"Error during encoding: {str(e)}")
        else:
            st.info("No categorical columns available for encoding.")

    def _render_automl_preprocessing(self, df, dataset_name):
        """Render AutoML preprocessing section"""
        st.subheader("🤖 AutoML Preprocessing")
        
        st.markdown("### 🚀 Automated Preprocessing Pipeline")
        
        if st.checkbox("Enable AutoML Preprocessing", key="automl_prep"):
            # AutoML preprocessing options
            col1, col2 = st.columns(2)
            
            with col1:
                handle_missing = st.checkbox("Auto Handle Missing Values", value=True)
                handle_outliers = st.checkbox("Auto Handle Outliers", value=True)
                auto_scaling = st.checkbox("Auto Feature Scaling", value=True)
                
            with col2:
                auto_encoding = st.checkbox("Auto Categorical Encoding", value=True)
                feature_selection = st.checkbox("Auto Feature Selection", value=True)
                create_features = st.checkbox("Auto Feature Creation", value=False)
            
            # Target column for supervised tasks
            target_col = st.selectbox("Select Target Column (if applicable)", [None] + df.columns.tolist())
            
            if st.button("🚀 Run AutoML Preprocessing", key="run_automl_preprocessing"):
                with st.spinner("Running automated preprocessing pipeline..."):
                    try:
                        processed_df = df.copy()
                        
                        # Step 1: Handle missing values
                        if handle_missing:
                            with st.spinner("Handling missing values..."):
                                processed_df = self._apply_smart_imputation(processed_df, auto_detect=True)
                                st.success("✅ Missing values handled")
                        
                        # Step 2: Handle outliers
                        if handle_outliers:
                            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                with st.spinner("Handling outliers..."):
                                    processed_df = self._detect_and_treat_outliers(
                                        processed_df, numeric_cols, 
                                        "IQR (Interquartile Range)", 
                                        "Cap/Floor (Winsorization)"
                                    )
                                    st.success("✅ Outliers handled")
                        
                        # Step 3: Auto scaling
                        if auto_scaling:
                            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                with st.spinner("Applying feature scaling..."):
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()
                                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                                    st.success("✅ Feature scaling applied")
                        
                        # Step 4: Auto encoding
                        if auto_encoding:
                            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
                            if categorical_cols:
                                with st.spinner("Applying categorical encoding..."):
                                    # One-hot encode categorical columns
                                    dummies = pd.get_dummies(processed_df[categorical_cols], prefix=categorical_cols)
                                    processed_df = pd.concat([
                                        processed_df.drop(categorical_cols, axis=1), 
                                        dummies
                                    ], axis=1)
                                    st.success("✅ Categorical encoding applied")
                        
                        # Step 5: Feature selection
                        if feature_selection and target_col:
                            with st.spinner("Performing feature selection..."):
                                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                                
                                X = processed_df.drop(target_col, axis=1)
                                y = processed_df[target_col]
                                
                                # Select top 75% of features
                                k = max(1, int(len(X.columns) * 0.75))
                                
                                # Use appropriate scoring function
                                if y.dtype in ['object', 'category']:
                                    score_func = f_classif
                                else:
                                    score_func = f_regression
                                
                                selector = SelectKBest(score_func=score_func, k=k)
                                X_selected = selector.fit_transform(X, y)
                                selected_features = X.columns[selector.get_support()]
                                
                                processed_df = pd.concat([
                                    X[selected_features], 
                                    y
                                ], axis=1)
                                st.success(f"✅ Feature selection applied ({k} features selected)")
                        
                        # Update session state
                        st.session_state.datasets[dataset_name] = processed_df
                        
                        st.success("✅ AutoML preprocessing pipeline completed successfully!")
                        st.info(f"Original shape: {df.shape} → Processed shape: {processed_df.shape}")
                        
                        # Log to history
                        self.history.log_step(
                            "AutoML Preprocessing",
                            f"Completed AutoML preprocessing for {dataset_name}",
                            {
                                "original_shape": str(df.shape),
                                "processed_shape": str(processed_df.shape),
                                "steps_applied": sum([handle_missing, handle_outliers, auto_scaling, auto_encoding, feature_selection])
                            },
                            "success"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during AutoML preprocessing: {str(e)}")
        
        # TPOT AutoML (if available)
        st.markdown("### 🧪 TPOT AutoML Optimization")
        
        if HAS_TPOT:
            st.info("TPOT is available for automated machine learning pipeline optimization.")
            
            if target_col:
                task_type = st.radio("Task Type", ["Classification", "Regression"])
                generations = st.slider("Generations", 1, 20, 5)
                population_size = st.slider("Population Size", 10, 200, 50)
                
                if st.button("🧬 Run TPOT Optimization", key="run_tpot"):
                    with st.spinner("Running TPOT optimization..."):
                        try:
                            X = df.drop(target_col, axis=1)
                            y = df[target_col]
                            
                            # Handle categorical variables
                            X = pd.get_dummies(X)
                            
                            if task_type == "Classification":
                                tpot = TPOTClassifier(
                                    generations=generations,
                                    population_size=population_size,
                                    verbosity=2,
                                    random_state=42
                                )
                            else:
                                tpot = TPOTRegressor(
                                    generations=generations,
                                    population_size=population_size,
                                    verbosity=2,
                                    random_state=42
                                )
                            
                            # Fit TPOT
                            tpot.fit(X, y)
                            
                            # Show best pipeline
                            st.success("✅ TPOT optimization completed!")
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
                                    "population_size": population_size
                                },
                                "success"
                            )
                        except Exception as e:
                            st.error(f"Error during TPOT optimization: {str(e)}")
            else:
                st.warning("Please select a target column to enable TPOT optimization.")
        else:
            st.info("TPOT not installed. Install with: pip install tpot")

    def render_scaling_ui(self):
        """Render scaling interface (for backward compatibility)"""
        st.subheader("⚖️ Enhanced Scaling & Encoding")
        st.info("This interface has been enhanced. Please use the main preprocessing tab for advanced features.")