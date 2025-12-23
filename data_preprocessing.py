import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import sklearn

# Try to import IterativeImputer - handle sklearn version differences
try:
    from sklearn.impute import IterativeImputer

    HAS_ITERATIVE_IMPUTER = True
except ImportError:
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        HAS_ITERATIVE_IMPUTER = True
    except ImportError:
        HAS_ITERATIVE_IMPUTER = False

from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib
import pickle
from datetime import datetime
from pipeline_history import PipelineHistory
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessing:
    def __init__(self):
        self.history = PipelineHistory()

    def _safe_duplicated_check(self, df):
        """Safely check for duplicated rows, handling unhashable types"""
        try:
            return df.duplicated().sum()
        except TypeError:
            # Handle unhashable types by converting to string representations
            try:
                # Convert all columns to string for duplicate checking
                df_str = df.astype(str)
                return df_str.duplicated().sum()
            except Exception:
                # If all else fails, return 0 (no duplicates detected)
                return 0

    def _safe_drop_duplicates(self, df):
        """Safely drop duplicates, handling unhashable types"""
        try:
            return df.drop_duplicates()
        except TypeError:
            # Handle unhashable types by converting to string representations
            try:
                # Create a copy and convert problematic columns
                df_clean = df.copy()

                # Identify columns that might contain unhashable types
                for col in df_clean.columns:
                    try:
                        # Test if column can be used in duplicate checking
                        df_clean[col].duplicated().head(1)
                    except TypeError:
                        # Convert problematic column to string
                        df_clean[col] = df_clean[col].astype(str)

                return df_clean.drop_duplicates()
            except Exception:
                # If conversion fails, return original dataframe
                st.warning("Could not remove duplicates due to complex data types in the dataset.")
                return df

    def _convert_unhashable_columns(self, df):
        """Convert columns with unhashable types to strings"""
        df_converted = df.copy()

        for col in df_converted.columns:
            try:
                # Test if column contains hashable values
                pd.Series(df_converted[col]).nunique()
            except TypeError:
                # Convert unhashable column to string
                df_converted[col] = df_converted[col].astype(str)
                st.info(f"Converted column '{col}' to string due to complex data types.")

        return df_converted

    def render_preprocessing_ui(self):
        """Render the preprocessing interface"""

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Preprocessing", dataset_names)

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        # Convert unhashable columns to strings upfront
        df = self._convert_unhashable_columns(df)

        # Preprocessing mode selector
        mode = st.radio(
            "Preprocessing Mode",
            ["Automatic Clean", "Manual Clean"],
            horizontal=True,
            help="Choose between automated cleaning or manual step-by-step cleaning"
        )

        if mode == "Automatic Clean":
            self._render_automatic_clean(df, selected_dataset)
        else:
            self._render_manual_clean(df, selected_dataset)

    def _render_automatic_clean(self, df, dataset_name):
        """Render automatic cleaning interface"""
        st.subheader("🤖 Automatic Data Cleaning")

        # Preview what will be done
        st.markdown("**Preview of Automatic Cleaning Operations:**")

        operations = []

        # Check for duplicates (safely)
        duplicate_count = self._safe_duplicated_check(df)
        if duplicate_count > 0:
            operations.append(f"✅ Remove {duplicate_count} duplicate rows")

        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            numeric_missing = [col for col in missing_cols if df[col].dtype in ['int64', 'float64']]
            categorical_missing = [col for col in missing_cols if col not in numeric_missing]

            if numeric_missing:
                operations.append(f"✅ Impute missing values in numeric columns {numeric_missing} with median")
            if categorical_missing:
                operations.append(f"✅ Impute missing values in categorical columns {categorical_missing} with mode")

        # Check for whitespace
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            operations.append(f"✅ Trim whitespace from {len(object_cols)} text columns")

        # Check for data type inference
        potential_numeric = []
        for col in object_cols:
            try:
                sample = df[col].dropna().head(100)
                pd.to_numeric(sample)
                potential_numeric.append(col)
            except:
                pass

        if potential_numeric:
            operations.append(f"✅ Convert {potential_numeric} to numeric type")

        if operations:
            for op in operations:
                st.write(op)
        else:
            st.info("No cleaning operations needed - data appears clean!")

        col1, col2 = st.columns(2)

        with col1:
            new_dataset_name = st.text_input("Cleaned Dataset Name", value=f"{dataset_name}_cleaned")

        with col2:
            if st.button("🧹 Execute Automatic Cleaning", type="primary"):
                if operations:
                    self._execute_automatic_cleaning(df, new_dataset_name, operations)
                else:
                    st.info("No cleaning needed!")

    def _execute_automatic_cleaning(self, df, new_dataset_name, operations):
        """Execute automatic cleaning operations"""

        with st.spinner("Performing automatic cleaning..."):
            cleaned_df = df.copy()
            executed_operations = []

            try:
                # Remove duplicates (safely)
                duplicate_count = self._safe_duplicated_check(cleaned_df)
                if duplicate_count > 0:
                    before_rows = len(cleaned_df)
                    cleaned_df = self._safe_drop_duplicates(cleaned_df)
                    after_rows = len(cleaned_df)
                    executed_operations.append(f"Removed {before_rows - after_rows} duplicate rows")

                # Handle missing values
                missing_cols = cleaned_df.columns[cleaned_df.isnull().any()].tolist()

                for col in missing_cols:
                    if cleaned_df[col].dtype in ['int64', 'float64']:
                        # Numeric - use median
                        median_val = cleaned_df[col].median()
                        cleaned_df[col].fillna(median_val, inplace=True)
                        executed_operations.append(f"Imputed {col} with median ({median_val:.2f})")
                    else:
                        # Categorical - use mode
                        try:
                            mode_val = cleaned_df[col].mode().iloc[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown'
                            cleaned_df[col].fillna(mode_val, inplace=True)
                            executed_operations.append(f"Imputed {col} with mode ({mode_val})")
                        except Exception:
                            # Fallback for problematic columns
                            cleaned_df[col].fillna('Unknown', inplace=True)
                            executed_operations.append(f"Imputed {col} with 'Unknown'")

                # Trim whitespace
                object_cols = cleaned_df.select_dtypes(include=['object']).columns
                for col in object_cols:
                    try:
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                            executed_operations.append(f"Trimmed whitespace from {col}")
                    except Exception:
                        # Skip problematic columns
                        pass

                # Infer data types
                for col in object_cols:
                    try:
                        # Try to convert to numeric
                        numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                        if numeric_series.notna().sum() / len(cleaned_df) > 0.8:  # 80% conversion success
                            cleaned_df[col] = numeric_series
                            executed_operations.append(f"Converted {col} to numeric")
                    except Exception:
                        # Skip problematic columns
                        pass

                # Save cleaned dataset
                st.session_state.datasets[new_dataset_name] = cleaned_df
                st.session_state.current_dataset = new_dataset_name

                # Log to history
                self.history.log_step(
                    "Automatic Cleaning",
                    f"Applied automatic cleaning to create {new_dataset_name}",
                    {
                        "original_rows": len(df),
                        "cleaned_rows": len(cleaned_df),
                        "operations": executed_operations
                    },
                    "success"
                )

                st.success(f"✅ Automatic cleaning completed! Created dataset: {new_dataset_name}")

                # Show summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", len(df))
                    st.metric("Original Columns", len(df.columns))

                with col2:
                    st.metric("Cleaned Rows", len(cleaned_df))
                    st.metric("Cleaned Columns", len(cleaned_df.columns))

                # Show executed operations
                st.markdown("**Executed Operations:**")
                for op in executed_operations:
                    st.write(f"✅ {op}")

                # Show sample of cleaned data
                st.markdown("**Sample of Cleaned Data:**")
                st.dataframe(cleaned_df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error during automatic cleaning: {str(e)}")
                self.history.log_step(
                    "Automatic Cleaning",
                    "Failed automatic cleaning",
                    {"error": str(e)},
                    "error"
                )

    def _render_manual_clean(self, df, dataset_name):
        """Render manual cleaning interface"""
        st.subheader("🔧 Manual Data Cleaning")

        # Create tabs for different cleaning operations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Missing Values", "Duplicates", "Outliers", "Data Types", "Custom Transformations"
        ])

        with tab1:
            self._render_missing_values_ui(df, dataset_name)

        with tab2:
            self._render_duplicates_ui(df, dataset_name)

        with tab3:
            self._render_outliers_ui(df, dataset_name)

        with tab4:
            self._render_data_types_ui(df, dataset_name)

        with tab5:
            self._render_custom_transform_ui(df, dataset_name)

    def _render_missing_values_ui(self, df, dataset_name):
        """Render missing values handling UI"""
        st.markdown("### 🕳️ Missing Values Handling")

        # Show missing values summary
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]

        if missing_summary.empty:
            st.success("✅ No missing values found!")
            return

        st.markdown("**Missing Values Summary:**")
        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing Count': missing_summary.values,
            'Percentage': (missing_summary.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)

        # Column selector
        selected_column = st.selectbox("Select Column to Handle", missing_summary.index)

        if selected_column:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Current Status of {selected_column}:**")
                st.write(f"Data Type: {df[selected_column].dtype}")
                st.write(f"Missing: {df[selected_column].isnull().sum()}")
                st.write(f"Non-null: {df[selected_column].notna().sum()}")

                # Show sample values (safely)
                st.markdown("**Sample Non-null Values:**")
                try:
                    sample_values = df[selected_column].dropna().head(10)
                    st.write(sample_values.tolist())
                except Exception:
                    st.write("Unable to display sample values")

            with col2:
                # Imputation method selector
                imputation_methods = [
                    "Mean (numeric only)",
                    "Median (numeric only)",
                    "Mode (most frequent)",
                    "Forward Fill",
                    "Backward Fill",
                    "Constant Value",
                    "Linear Interpolation",
                    "KNN Imputer",
                    "Custom Expression"
                ]

                # Add IterativeImputer if available
                if HAS_ITERATIVE_IMPUTER:
                    imputation_methods.insert(-1, "Iterative Imputer")

                method = st.selectbox("Imputation Method", imputation_methods)

                # Additional parameters based on method
                if method == "Constant Value":
                    fill_value = st.text_input("Fill Value", value="0")
                elif method == "KNN Imputer":
                    n_neighbors = st.number_input("Number of Neighbors", value=5, min_value=1)
                elif method == "Iterative Imputer" and HAS_ITERATIVE_IMPUTER:
                    max_iter = st.number_input("Max Iterations", value=10, min_value=1)
                    random_state = st.number_input("Random State", value=42)
                elif method == "Custom Expression":
                    custom_expr = st.text_area(
                        "Custom Expression (pandas syntax)",
                        value="df[column].fillna(df[column].median())",
                        help="Use 'df' for dataframe and 'column' for column name"
                    )

                if st.button(f"Apply {method}", type="primary"):
                    self._apply_missing_value_treatment(df, dataset_name, selected_column, method, locals())

    def _apply_missing_value_treatment(self, df, dataset_name, column, method, params):
        """Apply missing value treatment"""
        try:
            df_modified = df.copy()

            if method == "Mean (numeric only)":
                if df[column].dtype in ['int64', 'float64']:
                    fill_value = df[column].mean()
                    df_modified[column].fillna(fill_value, inplace=True)
                    action = f"Filled with mean: {fill_value:.2f}"
                else:
                    st.error("Mean imputation only works for numeric columns")
                    return

            elif method == "Median (numeric only)":
                if df[column].dtype in ['int64', 'float64']:
                    fill_value = df[column].median()
                    df_modified[column].fillna(fill_value, inplace=True)
                    action = f"Filled with median: {fill_value:.2f}"
                else:
                    st.error("Median imputation only works for numeric columns")
                    return

            elif method == "Mode (most frequent)":
                try:
                    mode_value = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else 'Unknown'
                    df_modified[column].fillna(mode_value, inplace=True)
                    action = f"Filled with mode: {mode_value}"
                except Exception:
                    df_modified[column].fillna('Unknown', inplace=True)
                    action = "Filled with 'Unknown'"

            elif method == "Forward Fill":
                df_modified[column].fillna(method='ffill', inplace=True)
                action = "Applied forward fill"

            elif method == "Backward Fill":
                df_modified[column].fillna(method='bfill', inplace=True)
                action = "Applied backward fill"

            elif method == "Constant Value":
                fill_value = params.get('fill_value', '0')
                # Try to convert to appropriate type
                if df[column].dtype in ['int64', 'float64']:
                    try:
                        fill_value = float(fill_value)
                    except:
                        st.error("Invalid numeric value")
                        return
                df_modified[column].fillna(fill_value, inplace=True)
                action = f"Filled with constant: {fill_value}"

            elif method == "Linear Interpolation":
                if df[column].dtype in ['int64', 'float64']:
                    df_modified[column] = df_modified[column].interpolate(method='linear')
                    action = "Applied linear interpolation"
                else:
                    st.error("Interpolation only works for numeric columns")
                    return

            elif method == "KNN Imputer":
                if df[column].dtype in ['int64', 'float64']:
                    n_neighbors = params.get('n_neighbors', 5)
                    imputer = KNNImputer(n_neighbors=n_neighbors)

                    # Get numeric columns for KNN
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    df_numeric = df[numeric_cols]

                    imputed_data = imputer.fit_transform(df_numeric)
                    df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
                    df_modified[column] = df_imputed[column]
                    action = f"Applied KNN imputation (k={n_neighbors})"
                else:
                    st.error("KNN imputation only works for numeric columns")
                    return

            elif method == "Iterative Imputer" and HAS_ITERATIVE_IMPUTER:
                if df[column].dtype in ['int64', 'float64']:
                    max_iter = params.get('max_iter', 10)
                    random_state = params.get('random_state', 42)
                    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)

                    # Get numeric columns for iterative imputation
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    df_numeric = df[numeric_cols]

                    imputed_data = imputer.fit_transform(df_numeric)
                    df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
                    df_modified[column] = df_imputed[column]
                    action = f"Applied iterative imputation (max_iter={max_iter})"
                else:
                    st.error("Iterative imputation only works for numeric columns")
                    return

            elif method == "Custom Expression":
                custom_expr = params.get('custom_expr', '')
                if custom_expr:
                    # Replace placeholders
                    custom_expr = custom_expr.replace('column', f"'{column}'")
                    custom_expr = custom_expr.replace('df', 'df_modified')

                    # Execute custom expression (with warning)
                    st.warning("⚠️ Executing custom expression. Please ensure it's safe!")
                    exec(custom_expr)
                    action = f"Applied custom expression: {custom_expr}"
                else:
                    st.error("Please provide a custom expression")
                    return

            # Handle case where IterativeImputer is not available
            elif method == "Iterative Imputer" and not HAS_ITERATIVE_IMPUTER:
                st.error(
                    "IterativeImputer is not available in your scikit-learn version. Please upgrade or use another method.")
                return

            # Update dataset
            new_dataset_name = f"{dataset_name}_mv_treated"
            st.session_state.datasets[new_dataset_name] = df_modified

            # Log the operation
            self.history.log_step(
                "Missing Value Treatment",
                f"Applied {method} to column {column}",
                {
                    "column": column,
                    "method": method,
                    "action": action,
                    "before_missing": df[column].isnull().sum(),
                    "after_missing": df_modified[column].isnull().sum()
                },
                "success"
            )

            st.success(f"✅ Applied {method} to {column}")
            st.info(f"New dataset created: {new_dataset_name}")

            # Show before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Before - Missing", df[column].isnull().sum())
            with col2:
                st.metric("After - Missing", df_modified[column].isnull().sum())

        except Exception as e:
            st.error(f"Error applying missing value treatment: {str(e)}")
            self.history.log_step(
                "Missing Value Treatment",
                f"Failed to apply {method} to column {column}",
                {"error": str(e)},
                "error"
            )

    def _render_duplicates_ui(self, df, dataset_name):
        """Render duplicates handling UI"""
        st.markdown("### 🔄 Duplicate Rows Handling")

        duplicate_count = self._safe_duplicated_check(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Duplicate Rows", duplicate_count)
        with col3:
            if duplicate_count > 0:
                st.metric("Duplicate %", f"{(duplicate_count / len(df) * 100):.1f}%")
            else:
                st.metric("Duplicate %", "0%")

        if duplicate_count > 0:
            # Show some duplicate examples (safely)
            st.markdown("**Sample Duplicate Information:**")
            st.info(
                f"Found {duplicate_count} duplicate rows. Due to data complexity, individual duplicates are not displayed.")

            # Options for handling duplicates
            col1, col2 = st.columns(2)

            with col1:
                keep_option = st.selectbox(
                    "Keep which duplicate?",
                    ["first", "last"],
                    help="first: keep first occurrence, last: keep last occurrence"
                )

            with col2:
                subset_cols = st.multiselect(
                    "Consider subset of columns (optional)",
                    df.columns.tolist(),
                    help="If selected, only these columns will be used to identify duplicates"
                )

            if st.button("Remove Duplicates", type="primary"):
                try:
                    df_deduped = self._safe_drop_duplicates(df.copy())
                    removed_count = len(df) - len(df_deduped)

                    # Save modified dataset
                    new_dataset_name = f"{dataset_name}_deduped"
                    st.session_state.datasets[new_dataset_name] = df_deduped

                    # Log operation
                    self.history.log_step(
                        "Duplicate Removal",
                        f"Removed {removed_count} duplicate rows",
                        {
                            "original_rows": len(df),
                            "final_rows": len(df_deduped),
                            "removed": removed_count,
                            "keep": keep_option,
                            "subset_columns": subset_cols
                        },
                        "success"
                    )

                    st.success(f"✅ Removed {removed_count} duplicate rows")
                    st.info(f"New dataset created: {new_dataset_name}")

                except Exception as e:
                    st.error(f"Error removing duplicates: {str(e)}")

        else:
            st.success("✅ No duplicate rows found!")

    def _render_outliers_ui(self, df, dataset_name):
        """Render outlier handling UI"""
        st.markdown("### 📊 Outlier Detection & Handling")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns available for outlier detection.")
            return

        selected_column = st.selectbox("Select Column for Outlier Analysis", numeric_cols)

        if selected_column:
            col1, col2 = st.columns(2)

            with col1:
                # Outlier detection method
                method = st.selectbox(
                    "Outlier Detection Method",
                    ["IQR Method", "Z-Score", "Isolation Forest", "Percentile Trimming"]
                )

                # Method-specific parameters
                if method == "IQR Method":
                    multiplier = st.number_input("IQR Multiplier", value=1.5, min_value=0.1, step=0.1)
                elif method == "Z-Score":
                    threshold = st.number_input("Z-Score Threshold", value=3.0, min_value=0.1, step=0.1)
                elif method == "Isolation Forest":
                    contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1)
                elif method == "Percentile Trimming":
                    lower_percentile = st.number_input("Lower Percentile", value=5.0, min_value=0.0, max_value=50.0)
                    upper_percentile = st.number_input("Upper Percentile", value=95.0, min_value=50.0, max_value=100.0)

                # Action for outliers
                action = st.selectbox(
                    "Action for Outliers",
                    ["Remove", "Cap (Winsorize)", "Transform (Log)", "Keep (Mark only)"]
                )

            with col2:
                # Current distribution
                st.markdown("**Current Distribution:**")
                try:
                    import plotly.express as px

                    fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    st.info("Plotly not available for visualization")
                except Exception as e:
                    st.info(f"Unable to create histogram: {str(e)}")

                # Statistics
                try:
                    stats_df = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max'],
                        'Value': [
                            df[selected_column].count(),
                            df[selected_column].mean(),
                            df[selected_column].std(),
                            df[selected_column].min(),
                            df[selected_column].quantile(0.25),
                            df[selected_column].median(),
                            df[selected_column].quantile(0.75),
                            df[selected_column].max()
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")

            if st.button("Detect & Handle Outliers", type="primary"):
                self._handle_outliers(df, dataset_name, selected_column, method, action, locals())

    def _handle_outliers(self, df, dataset_name, column, method, action, params):
        """Handle outliers in the specified column"""
        try:
            df_modified = df.copy()
            outlier_indices = []

            # Detect outliers based on method
            if method == "IQR Method":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = params.get('multiplier', 1.5)

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
                method_info = f"IQR method (multiplier={multiplier}), bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"

            elif method == "Z-Score":
                threshold = params.get('threshold', 3.0)
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outlier_indices = df.iloc[np.where(z_scores > threshold)[0]].index
                method_info = f"Z-score method (threshold={threshold})"

            elif method == "Isolation Forest":
                contamination = params.get('contamination', 0.1)
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[column]].dropna())
                outlier_indices = df.dropna(subset=[column]).iloc[np.where(outlier_pred == -1)[0]].index
                method_info = f"Isolation Forest (contamination={contamination})"

            elif method == "Percentile Trimming":
                lower_percentile = params.get('lower_percentile', 5.0)
                upper_percentile = params.get('upper_percentile', 95.0)

                lower_bound = df[column].quantile(lower_percentile / 100)
                upper_bound = df[column].quantile(upper_percentile / 100)

                outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
                method_info = f"Percentile trimming ({lower_percentile}%-{upper_percentile}%), bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"

            outlier_count = len(outlier_indices)

            # Apply action for outliers
            if action == "Remove":
                df_modified = df_modified.drop(outlier_indices)
                action_info = f"Removed {outlier_count} outlier rows"

            elif action == "Cap (Winsorize)":
                if method in ["IQR Method", "Percentile Trimming"]:
                    # Use the bounds calculated above
                    df_modified[column] = df_modified[column].clip(lower=lower_bound, upper=upper_bound)
                    action_info = f"Capped {outlier_count} outlier values to bounds"
                else:
                    st.error("Capping only available for IQR and Percentile methods")
                    return

            elif action == "Transform (Log)":
                # Apply log transformation (add 1 to handle zeros)
                min_val = df[column].min()
                if min_val <= 0:
                    df_modified[column] = np.log1p(df_modified[column] - min_val + 1)
                else:
                    df_modified[column] = np.log1p(df_modified[column])
                action_info = f"Applied log transformation to {column}"

            elif action == "Keep (Mark only)":
                # Add outlier flag column
                df_modified[f'{column}_outlier'] = df_modified.index.isin(outlier_indices)
                action_info = f"Added outlier flag column for {outlier_count} outliers"

            # Save modified dataset
            new_dataset_name = f"{dataset_name}_outliers_handled"
            st.session_state.datasets[new_dataset_name] = df_modified

            # Log operation
            self.history.log_step(
                "Outlier Handling",
                f"Applied {method} + {action} to {column}",
                {
                    "column": column,
                    "method": method_info,
                    "action": action_info,
                    "outliers_detected": outlier_count,
                    "original_rows": len(df),
                    "final_rows": len(df_modified)
                },
                "success"
            )

            st.success(f"✅ {action_info}")
            st.info(f"New dataset created: {new_dataset_name}")

            # Show before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Rows", len(df))
                st.metric("Outliers Detected", outlier_count)
            with col2:
                st.metric("Final Rows", len(df_modified))
                st.metric("Outlier %", f"{(outlier_count / len(df) * 100):.1f}%")

        except Exception as e:
            st.error(f"Error handling outliers: {str(e)}")
            self.history.log_step(
                "Outlier Handling",
                f"Failed to handle outliers in {column}",
                {"error": str(e)},
                "error"
            )

    def _render_data_types_ui(self, df, dataset_name):
        """Render data type conversion UI"""
        st.markdown("### 🔧 Data Type Conversion")

        # Show current data types
        st.markdown("**Current Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Current Type': df.dtypes.astype(str),
            'Non-null Count': df.count(),
            'Sample Values': [str(df[col].dropna().iloc[:3].tolist()) for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)

        # Column selector for conversion
        selected_columns = st.multiselect("Select Columns to Convert", df.columns.tolist())

        if selected_columns:
            target_type = st.selectbox(
                "Target Data Type",
                ["int64", "float64", "object", "category", "datetime64[ns]", "bool"]
            )

            # Show preview of conversion
            st.markdown("**Conversion Preview:**")

            for col in selected_columns:
                try:
                    if target_type == "int64":
                        converted = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif target_type == "float64":
                        converted = pd.to_numeric(df[col], errors='coerce')
                    elif target_type == "object":
                        converted = df[col].astype(str)
                    elif target_type == "category":
                        converted = df[col].astype('category')
                    elif target_type == "datetime64[ns]":
                        converted = pd.to_datetime(df[col], errors='coerce')
                    elif target_type == "bool":
                        converted = df[col].astype('bool')

                    success_rate = (converted.notna().sum() / len(df)) * 100
                    st.write(f"**{col}**: {success_rate:.1f}% successful conversion")

                    if success_rate < 100:
                        failed_count = converted.isna().sum() - df[col].isna().sum()
                        st.warning(f"⚠️ {failed_count} values would become NaN")

                except Exception as e:
                    st.error(f"❌ {col}: Conversion preview failed - {str(e)}")

            if st.button("Apply Data Type Conversion", type="primary"):
                self._apply_dtype_conversion(df, dataset_name, selected_columns, target_type)

    def _apply_dtype_conversion(self, df, dataset_name, columns, target_type):
        """Apply data type conversion"""
        try:
            df_modified = df.copy()
            conversion_results = []

            for col in columns:
                original_type = str(df[col].dtype)

                try:
                    if target_type == "int64":
                        df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce').astype('Int64')
                    elif target_type == "float64":
                        df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce')
                    elif target_type == "object":
                        df_modified[col] = df_modified[col].astype(str)
                    elif target_type == "category":
                        df_modified[col] = df_modified[col].astype('category')
                    elif target_type == "datetime64[ns]":
                        df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')
                    elif target_type == "bool":
                        df_modified[col] = df_modified[col].astype('bool')

                    conversion_results.append({
                        'column': col,
                        'from': original_type,
                        'to': target_type,
                        'success': True
                    })

                except Exception as e:
                    conversion_results.append({
                        'column': col,
                        'from': original_type,
                        'to': target_type,
                        'success': False,
                        'error': str(e)
                    })

            # Save modified dataset
            new_dataset_name = f"{dataset_name}_dtypes_converted"
            st.session_state.datasets[new_dataset_name] = df_modified

            # Log operation
            self.history.log_step(
                "Data Type Conversion",
                f"Converted {len(columns)} columns to {target_type}",
                {
                    "columns": columns,
                    "target_type": target_type,
                    "conversions": conversion_results
                },
                "success"
            )

            st.success(f"✅ Data type conversion completed")
            st.info(f"New dataset created: {new_dataset_name}")

            # Show conversion results
            results_df = pd.DataFrame(conversion_results)
            st.dataframe(results_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error in data type conversion: {str(e)}")
            self.history.log_step(
                "Data Type Conversion",
                f"Failed data type conversion",
                {"error": str(e)},
                "error"
            )

    def _render_custom_transform_ui(self, df, dataset_name):
        """Render custom transformation UI"""
        st.markdown("### 🧠 Custom Transformations")

        st.warning("⚠️ **Safety Warning**: Custom expressions will be executed. Ensure they are safe!")

        # Preset transformations
        st.markdown("**Preset Transformations:**")
        presets = {
            "Normalize Column": "df['column_name'] = (df['column_name'] - df['column_name'].mean()) / df['column_name'].std()",
            "Create Age Groups": "df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])",
            "Extract Year from Date": "df['year'] = pd.to_datetime(df['date_column']).dt.year",
            "Create Binary Flag": "df['high_value'] = df['value_column'] > df['value_column'].median()",
            "Combine Columns": "df['full_name'] = df['first_name'].str.cat(df['last_name'], sep=' ')"
        }

        selected_preset = st.selectbox("Choose Preset (optional)", ["None"] + list(presets.keys()))

        if selected_preset != "None":
            custom_expression = st.text_area(
                "Custom Expression (Python/Pandas)",
                value=presets[selected_preset],
                height=100,
                help="Use standard pandas operations. 'df' refers to the current dataset."
            )
        else:
            custom_expression = st.text_area(
                "Custom Expression (Python/Pandas)",
                value="# Example: df['new_column'] = df['existing_column'] * 2",
                height=100,
                help="Use standard pandas operations. 'df' refers to the current dataset."
            )

        # Validation checkbox
        validate = st.checkbox("Validate expression (recommended)", value=True)

        if st.button("Execute Custom Transformation", type="primary"):
            if custom_expression.strip() and not custom_expression.strip().startswith('#'):
                self._execute_custom_transform(df, dataset_name, custom_expression, validate)
            else:
                st.error("Please provide a valid custom expression")

    def _execute_custom_transform(self, df, dataset_name, expression, validate=True):
        """Execute custom transformation"""
        try:
            df_modified = df.copy()

            # Basic validation if enabled
            if validate:
                # Check for potentially dangerous operations
                dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__', 'globals', 'locals']
                expression_lower = expression.lower()

                for keyword in dangerous_keywords:
                    if keyword in expression_lower:
                        st.error(
                            f"⚠️ Potentially dangerous keyword detected: '{keyword}'. Please review your expression.")
                        return

            # Execute the transformation
            # Replace 'df' with 'df_modified' in the expression
            safe_expression = expression.replace('df[', 'df_modified[')

            # Execute with limited scope
            exec(safe_expression, {"df_modified": df_modified, "pd": pd, "np": np})

            # Save modified dataset
            new_dataset_name = f"{dataset_name}_custom_transformed"
            st.session_state.datasets[new_dataset_name] = df_modified

            # Log operation
            self.history.log_step(
                "Custom Transformation",
                "Applied custom transformation",
                {
                    "expression": expression,
                    "original_shape": df.shape,
                    "final_shape": df_modified.shape
                },
                "success"
            )

            st.success("✅ Custom transformation applied successfully!")
            st.info(f"New dataset created: {new_dataset_name}")

            # Show shape comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
            with col2:
                st.metric("Final Shape", f"{df_modified.shape[0]} × {df_modified.shape[1]}")

            # Show sample of modified data
            st.markdown("**Sample of Transformed Data:**")
            st.dataframe(df_modified.head(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error executing custom transformation: {str(e)}")
            st.code(f"Expression: {expression}")

            self.history.log_step(
                "Custom Transformation",
                "Failed custom transformation",
                {"expression": expression, "error": str(e)},
                "error"
            )

    def render_scaling_ui(self):
        """Render scaling and encoding interface"""
        st.subheader("⚖️ Feature Scaling & Encoding")

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Scaling/Encoding", dataset_names)

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        # Convert unhashable columns to strings upfront
        df = self._convert_unhashable_columns(df)

        # Mode selector
        mode = st.radio(
            "Scaling Mode",
            ["Automatic", "Manual"],
            horizontal=True
        )

        if mode == "Automatic":
            self._render_automatic_scaling(df, selected_dataset)
        else:
            self._render_manual_scaling(df, selected_dataset)

    def _render_automatic_scaling(self, df, dataset_name):
        """Render automatic scaling interface"""
        st.markdown("### 🤖 Automatic Scaling & Encoding")

        # Analyze columns and suggest transformations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        st.markdown("**Automatic Analysis & Recommendations:**")

        recommendations = []

        # Numeric column recommendations
        for col in numeric_cols:
            data = df[col].dropna()

            if len(data) == 0:
                continue

            # Analyze distribution
            skewness = data.skew()
            has_outliers = len(data[(data < data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))) |
                                    (data > data.quantile(0.75) + 1.5 * (
                                                data.quantile(0.75) - data.quantile(0.25)))]) > 0

            # Recommend scaler
            if abs(skewness) > 2:
                recommended_scaler = "QuantileTransformer"
                reason = f"High skewness ({skewness:.2f})"
            elif has_outliers:
                recommended_scaler = "RobustScaler"
                reason = "Contains outliers"
            elif data.min() >= 0 and data.max() <= 1:
                recommended_scaler = "None"
                reason = "Already normalized"
            else:
                recommended_scaler = "StandardScaler"
                reason = "Normal distribution"

            recommendations.append({
                'column': col,
                'type': 'numeric',
                'recommended_transformer': recommended_scaler,
                'reason': reason
            })

        # Categorical column recommendations
        for col in categorical_cols:
            try:
                unique_count = df[col].nunique()

                if unique_count <= 10:
                    recommended_encoder = "OneHotEncoder"
                    reason = f"Low cardinality ({unique_count} categories)"
                elif unique_count <= 50:
                    recommended_encoder = "OrdinalEncoder"
                    reason = f"Medium cardinality ({unique_count} categories)"
                else:
                    recommended_encoder = "TargetEncoder"
                    reason = f"High cardinality ({unique_count} categories)"

                recommendations.append({
                    'column': col,
                    'type': 'categorical',
                    'recommended_transformer': recommended_encoder,
                    'reason': reason
                })
            except Exception:
                # Skip problematic columns
                pass

        # Display recommendations
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)

            if st.button("Apply Automatic Scaling/Encoding", type="primary"):
                self._apply_automatic_scaling(df, dataset_name, recommendations)
        else:
            st.info("No scaling/encoding recommendations generated.")

    def _get_onehot_encoder(self, drop='first'):
        """Get OneHotEncoder with version-appropriate parameters"""
        # Check scikit-learn version and use appropriate parameter
        sklearn_version = sklearn.__version__
        major, minor = map(int, sklearn_version.split('.')[:2])

        if major >= 1 and minor >= 2:
            # For sklearn >= 1.2.0, use sparse_output
            return OneHotEncoder(sparse_output=False, drop=drop)
        else:
            # For sklearn < 1.2.0, use sparse
            return OneHotEncoder(sparse=False, drop=drop)

    def _apply_automatic_scaling(self, df, dataset_name, recommendations):
        """Apply automatic scaling and encoding"""
        try:
            df_transformed = df.copy()
            applied_transformers = {}

            with st.spinner("Applying automatic transformations..."):

                for rec in recommendations:
                    col = rec['column']
                    transformer_name = rec['recommended_transformer']

                    if transformer_name == "None":
                        continue

                    if rec['type'] == 'numeric':
                        # Apply numeric scaling
                        if transformer_name == "StandardScaler":
                            scaler = StandardScaler()
                        elif transformer_name == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        elif transformer_name == "RobustScaler":
                            scaler = RobustScaler()
                        elif transformer_name == "QuantileTransformer":
                            scaler = QuantileTransformer()

                        # Fit and transform
                        df_transformed[col] = scaler.fit_transform(df_transformed[[col]])
                        applied_transformers[col] = scaler

                    else:  # categorical
                        # Apply categorical encoding
                        if transformer_name == "OneHotEncoder":
                            encoder = self._get_onehot_encoder(drop='first')
                            encoded = encoder.fit_transform(df_transformed[[col]])

                            # Create new column names
                            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]

                            # Add encoded columns and remove original
                            for i, name in enumerate(feature_names):
                                df_transformed[name] = encoded[:, i]
                            df_transformed = df_transformed.drop(columns=[col])

                        elif transformer_name == "OrdinalEncoder":
                            encoder = OrdinalEncoder()
                            df_transformed[col] = encoder.fit_transform(df_transformed[[col]])

                        elif transformer_name == "TargetEncoder":
                            # For target encoding, we need a target column
                            # For now, use label encoding as fallback
                            encoder = LabelEncoder()
                            df_transformed[col] = encoder.fit_transform(df_transformed[col].astype(str))

                        applied_transformers[col] = encoder

                # Save transformed dataset and transformers
                new_dataset_name = f"{dataset_name}_auto_scaled"
                st.session_state.datasets[new_dataset_name] = df_transformed

                # Initialize transformers dict if it doesn't exist
                if 'transformers' not in st.session_state:
                    st.session_state.transformers = {}
                st.session_state.transformers[new_dataset_name] = applied_transformers

                # Log operation
                self.history.log_step(
                    "Automatic Scaling/Encoding",
                    f"Applied automatic transformations to create {new_dataset_name}",
                    {
                        "original_shape": df.shape,
                        "transformed_shape": df_transformed.shape,
                        "transformers": list(applied_transformers.keys())
                    },
                    "success"
                )

                st.success(f"✅ Automatic scaling/encoding completed!")
                st.info(f"New dataset created: {new_dataset_name}")

                # Show transformation summary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Columns", len(df.columns))
                with col2:
                    st.metric("Transformed Columns", len(df_transformed.columns))

                # Show sample of transformed data
                st.markdown("**Sample of Transformed Data:**")
                st.dataframe(df_transformed.head(), use_container_width=True)

                # Export transformers option
                if st.button("💾 Export Transformers"):
                    transformer_data = {}
                    for col, transformer in applied_transformers.items():
                        # Convert to serializable format
                        transformer_data[col] = {
                            'type': type(transformer).__name__,
                            'params': transformer.get_params() if hasattr(transformer, 'get_params') else {}
                        }

                    import json
                    transformer_json = json.dumps(transformer_data, indent=2)

                    st.download_button(
                        label="Download Transformers (JSON)",
                        data=transformer_json,
                        file_name=f"transformers_{dataset_name}.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"Error in automatic scaling/encoding: {str(e)}")
            self.history.log_step(
                "Automatic Scaling/Encoding",
                "Failed automatic scaling/encoding",
                {"error": str(e)},
                "error"
            )

    def _render_manual_scaling(self, df, dataset_name):
        """Render manual scaling interface"""
        st.markdown("### 🔧 Manual Scaling & Encoding")

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Tabs for numeric and categorical
        tab1, tab2 = st.tabs(["Numeric Scaling", "Categorical Encoding"])

        with tab1:
            if numeric_cols:
                st.markdown("**Numeric Column Scaling:**")

                selected_numeric_cols = st.multiselect("Select Numeric Columns", numeric_cols)

                if selected_numeric_cols:
                    scaler_type = st.selectbox(
                        "Select Scaler",
                        ["StandardScaler", "MinMaxScaler", "RobustScaler", "QuantileTransformer"]
                    )

                    # Scaler-specific parameters
                    scaler_params = {}
                    if scaler_type == "QuantileTransformer":
                        output_distribution = st.selectbox("Output Distribution", ["uniform", "normal"])
                        scaler_params['output_distribution'] = output_distribution

                    # Preview scaling
                    if st.button("Preview Scaling"):
                        self._preview_scaling(df, selected_numeric_cols, scaler_type, scaler_params)

                    if st.button("Apply Numeric Scaling", type="primary"):
                        self._apply_manual_scaling(df, dataset_name, selected_numeric_cols, scaler_type, scaler_params)
            else:
                st.info("No numeric columns available for scaling.")

        with tab2:
            if categorical_cols:
                st.markdown("**Categorical Column Encoding:**")

                selected_categorical_cols = st.multiselect("Select Categorical Columns", categorical_cols)

                if selected_categorical_cols:
                    encoder_type = st.selectbox(
                        "Select Encoder",
                        ["OneHotEncoder", "OrdinalEncoder", "LabelEncoder"]
                    )

                    # Encoder-specific parameters
                    encoder_params = {}
                    if encoder_type == "OneHotEncoder":
                        drop_first = st.checkbox("Drop First Category", value=True)
                        encoder_params['drop_first'] = drop_first

                    if st.button("Apply Categorical Encoding", type="primary"):
                        self._apply_manual_encoding(df, dataset_name, selected_categorical_cols, encoder_type,
                                                    encoder_params)
            else:
                st.info("No categorical columns available for encoding.")

    def _preview_scaling(self, df, columns, scaler_type, params):
        """Preview the effect of scaling"""
        try:
            # Create scaler
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaler_type == "RobustScaler":
                scaler = RobustScaler()
            elif scaler_type == "QuantileTransformer":
                scaler = QuantileTransformer(**params)

            # Apply scaling to preview
            df_preview = df[columns].copy()
            scaled_data = scaler.fit_transform(df_preview)
            df_scaled = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in columns])

            # Show before/after comparison
            st.markdown("**Before/After Comparison:**")

            for i, col in enumerate(columns):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**{col} (Original):**")
                    st.write(df[col].describe())

                with col2:
                    st.write(f"**{col} (Scaled):**")
                    st.write(df_scaled[f"{col}_scaled"].describe())

        except Exception as e:
            st.error(f"Error previewing scaling: {str(e)}")

    def _apply_manual_scaling(self, df, dataset_name, columns, scaler_type, params):
        """Apply manual scaling to selected columns"""
        try:
            df_scaled = df.copy()

            # Create scaler
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaler_type == "RobustScaler":
                scaler = RobustScaler()
            elif scaler_type == "QuantileTransformer":
                scaler = QuantileTransformer(**params)

            # Apply scaling
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

            # Save scaled dataset and scaler
            new_dataset_name = f"{dataset_name}_scaled"
            st.session_state.datasets[new_dataset_name] = df_scaled

            # Initialize transformers dict if it doesn't exist
            if 'transformers' not in st.session_state:
                st.session_state.transformers = {}
            if new_dataset_name not in st.session_state.transformers:
                st.session_state.transformers[new_dataset_name] = {}
            st.session_state.transformers[new_dataset_name][f'{scaler_type}_scaler'] = scaler

            # Log operation
            self.history.log_step(
                "Manual Scaling",
                f"Applied {scaler_type} to {len(columns)} columns",
                {
                    "scaler_type": scaler_type,
                    "columns": columns,
                    "parameters": params
                },
                "success"
            )

            st.success(f"✅ {scaler_type} applied to {len(columns)} columns")
            st.info(f"New dataset created: {new_dataset_name}")

        except Exception as e:
            st.error(f"Error applying scaling: {str(e)}")
            self.history.log_step(
                "Manual Scaling",
                f"Failed to apply {scaler_type}",
                {"error": str(e)},
                "error"
            )

    def _apply_manual_encoding(self, df, dataset_name, columns, encoder_type, params):
        """Apply manual encoding to selected columns"""
        try:
            df_encoded = df.copy()
            applied_encoders = {}

            for col in columns:
                if encoder_type == "OneHotEncoder":
                    drop_option = 'first' if params.get('drop_first', True) else None
                    encoder = self._get_onehot_encoder(drop=drop_option)
                    encoded = encoder.fit_transform(df_encoded[[col]])

                    # Create new column names
                    if params.get('drop_first', True):
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    else:
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                    # Add encoded columns and remove original
                    for i, name in enumerate(feature_names):
                        df_encoded[name] = encoded[:, i]
                    df_encoded = df_encoded.drop(columns=[col])

                elif encoder_type == "OrdinalEncoder":
                    encoder = OrdinalEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[[col]])

                elif encoder_type == "LabelEncoder":
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

                applied_encoders[col] = encoder

            # Save encoded dataset and encoders
            new_dataset_name = f"{dataset_name}_encoded"
            st.session_state.datasets[new_dataset_name] = df_encoded

            # Initialize transformers dict if it doesn't exist
            if 'transformers' not in st.session_state:
                st.session_state.transformers = {}
            if new_dataset_name not in st.session_state.transformers:
                st.session_state.transformers[new_dataset_name] = {}
            st.session_state.transformers[new_dataset_name].update(applied_encoders)

            # Log operation
            self.history.log_step(
                "Manual Encoding",
                f"Applied {encoder_type} to {len(columns)} columns",
                {
                    "encoder_type": encoder_type,
                    "columns": columns,
                    "parameters": params
                },
                "success"
            )

            st.success(f"✅ {encoder_type} applied to {len(columns)} columns")
            st.info(f"New dataset created: {new_dataset_name}")

            # Show shape change
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Columns", len(df.columns))
            with col2:
                st.metric("Encoded Columns", len(df_encoded.columns))

        except Exception as e:
            st.error(f"Error applying encoding: {str(e)}")
            self.history.log_step(
                "Manual Encoding",
                f"Failed to apply {encoder_type}",
                {"error": str(e)},
                "error"
            )