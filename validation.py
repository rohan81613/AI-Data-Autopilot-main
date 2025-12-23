# Fixed validation.py - Resolved duplicate plotly_chart ID error

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pipeline_history import PipelineHistory
import warnings

warnings.filterwarnings('ignore')


class DataValidation:
    def __init__(self):
        self.history = PipelineHistory()

    def render_validation_ui(self):
        """Render the validation interface"""

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Validation", dataset_names)

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Validating Dataset:** {selected_dataset}")
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # Validation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Column-Level Validation",
            "📊 Dataset-Level Validation",
            "🎯 Robustness Scoring",
            "💡 Recommendations"
        ])

        with tab1:
            self._render_column_validation(df)

        with tab2:
            self._render_dataset_validation(df)

        with tab3:
            self._render_robustness_scoring(df)

        with tab4:
            self._render_validation_recommendations(df)

    def _render_column_validation(self, df):
        """Render column-level validation"""
        st.subheader("🔍 Column-Level Validation")

        # Column selector
        columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select Columns to Validate (leave empty for all)",
            columns,
            default=columns[:5] if len(columns) > 5 else columns
        )

        if not selected_columns:
            selected_columns = columns

        validation_results = []

        for col in selected_columns:
            col_validation = self._validate_column(df, col)
            validation_results.append(col_validation)

        # Display validation results
        if validation_results:
            results_df = pd.DataFrame(validation_results)

            # Format the results for display
            display_df = results_df[
                ['column', 'data_type', 'completeness_pct', 'uniqueness_pct', 'validity_score', 'overall_score']]

            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "completeness_pct": st.column_config.ProgressColumn(
                        "Completeness %",
                        help="Percentage of non-null values",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%"
                    ),
                    "uniqueness_pct": st.column_config.ProgressColumn(
                        "Uniqueness %",
                        help="Percentage of unique values",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%"
                    ),
                    "validity_score": st.column_config.ProgressColumn(
                        "Validity Score",
                        help="Data validity score (0-100)",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    ),
                    "overall_score": st.column_config.ProgressColumn(
                        "Overall Score",
                        help="Overall column quality score (0-100)",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    )
                }
            )

            # Detailed view for selected column
            st.markdown("### 🔎 Detailed Column Analysis")
            detailed_col = st.selectbox("Select Column for Detailed Analysis", selected_columns)

            if detailed_col:
                self._render_detailed_column_analysis(df, detailed_col, validation_results)

    def _validate_column(self, df, column):
        """Validate a single column"""
        col_data = df[column]

        # Basic metrics
        total_count = len(col_data)
        non_null_count = col_data.notna().sum()
        null_count = col_data.isna().sum()
        unique_count = col_data.nunique()

        # Calculate percentages
        completeness_pct = (non_null_count / total_count) * 100 if total_count > 0 else 0
        uniqueness_pct = (unique_count / non_null_count) * 100 if non_null_count > 0 else 0

        # Data type analysis
        data_type = str(col_data.dtype)
        inferred_type = self._infer_data_type(col_data)
        type_consistency = data_type == inferred_type

        # Value range analysis (for numeric columns)
        if col_data.dtype in ['int64', 'float64']:
            valid_range = self._check_numeric_range(col_data)
            outlier_count = self._count_outliers(col_data)
            outlier_impact = (outlier_count / non_null_count) * 100 if non_null_count > 0 else 0
        else:
            valid_range = True
            outlier_count = 0
            outlier_impact = 0

        # Distribution analysis
        distribution_health = self._assess_distribution_health(col_data)

        # Calculate validity score
        validity_components = [
            completeness_pct * 0.3,  # 30% weight for completeness
            (100 - outlier_impact) * 0.2,  # 20% weight for outlier control
            (100 if type_consistency else 50) * 0.2,  # 20% weight for type consistency
            (100 if valid_range else 0) * 0.1,  # 10% weight for valid range
            distribution_health * 0.2  # 20% weight for distribution health
        ]
        validity_score = sum(validity_components)

        # Overall quality score
        overall_score = (completeness_pct * 0.4 + uniqueness_pct * 0.2 + validity_score * 0.4)

        return {
            'column': column,
            'data_type': data_type,
            'inferred_type': inferred_type,
            'total_count': total_count,
            'non_null_count': non_null_count,
            'null_count': null_count,
            'unique_count': unique_count,
            'completeness_pct': completeness_pct,
            'uniqueness_pct': uniqueness_pct,
            'type_consistency': type_consistency,
            'valid_range': valid_range,
            'outlier_count': outlier_count,
            'outlier_impact': outlier_impact,
            'distribution_health': distribution_health,
            'validity_score': validity_score,
            'overall_score': overall_score
        }

    def _infer_data_type(self, series):
        """Infer the appropriate data type for a series"""
        non_null_series = series.dropna()

        if len(non_null_series) == 0:
            return str(series.dtype)

        # Try numeric conversion
        try:
            pd.to_numeric(non_null_series)
            return 'float64'
        except:
            pass

        # Try datetime conversion
        try:
            pd.to_datetime(non_null_series)
            return 'datetime64[ns]'
        except:
            pass

        # Check if boolean
        unique_values = set(non_null_series.astype(str).str.lower())
        if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no'}):
            return 'bool'

        return 'object'

    def _check_numeric_range(self, series):
        """Check if numeric values are in a reasonable range"""
        if series.dtype not in ['int64', 'float64']:
            return True

        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return True

        # Check for extreme values
        q1 = non_null_series.quantile(0.25)
        q3 = non_null_series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:  # No variation
            return True

        # Define reasonable bounds (more generous than outlier detection)
        lower_bound = q1 - 5 * iqr  # More generous than typical 1.5 * IQR
        upper_bound = q3 + 5 * iqr

        out_of_range = ((non_null_series < lower_bound) | (non_null_series > upper_bound)).sum()

        # If more than 5% of values are out of reasonable range, flag as invalid
        return (out_of_range / len(non_null_series)) <= 0.05

    def _count_outliers(self, series):
        """Count outliers using IQR method"""
        if series.dtype not in ['int64', 'float64']:
            return 0

        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return 0

        q1 = non_null_series.quantile(0.25)
        q3 = non_null_series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return 0

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = ((non_null_series < lower_bound) | (non_null_series > upper_bound)).sum()
        return outliers

    def _assess_distribution_health(self, series):
        """Assess the health of data distribution"""
        non_null_series = series.dropna()

        if len(non_null_series) == 0:
            return 0

        if series.dtype in ['int64', 'float64']:
            # For numeric data, check skewness and kurtosis
            if len(non_null_series) < 3:
                return 50  # Not enough data

            skewness = abs(non_null_series.skew())
            kurtosis = abs(non_null_series.kurtosis())

            # Score based on normality (lower skewness and kurtosis are better)
            skew_score = max(0, 100 - skewness * 10)  # Penalize high skewness
            kurt_score = max(0, 100 - kurtosis * 5)  # Penalize high kurtosis

            return (skew_score + kurt_score) / 2

        else:
            # For categorical data, check balance
            value_counts = non_null_series.value_counts()
            if len(value_counts) == 0:
                return 0

            # Calculate balance (entropy-based)
            proportions = value_counts / len(non_null_series)
            entropy = -np.sum(proportions * np.log2(proportions))
            max_entropy = np.log2(len(value_counts))

            if max_entropy == 0:
                return 100  # Single category

            balance_score = (entropy / max_entropy) * 100
            return balance_score

    def _render_detailed_column_analysis(self, df, column, validation_results):
        """Render detailed analysis for a specific column"""

        # Find the validation result for this column
        col_result = next((r for r in validation_results if r['column'] == column), None)
        if not col_result:
            return

        col1, col2 = st.columns(2)

        with col1:
            # Basic metrics
            st.markdown("**📊 Basic Metrics:**")
            metrics_data = {
                'Metric': ['Total Count', 'Non-Null', 'Null', 'Unique Values', 'Data Type'],
                'Value': [
                    col_result['total_count'],
                    col_result['non_null_count'],
                    col_result['null_count'],
                    col_result['unique_count'],
                    col_result['data_type']
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

            # Quality scores
            st.markdown("**🎯 Quality Scores:**")
            scores_data = {
                'Metric': ['Completeness', 'Uniqueness', 'Validity', 'Overall'],
                'Score': [
                    f"{col_result['completeness_pct']:.1f}%",
                    f"{col_result['uniqueness_pct']:.1f}%",
                    f"{col_result['validity_score']:.1f}",
                    f"{col_result['overall_score']:.1f}"
                ]
            }
            st.dataframe(pd.DataFrame(scores_data), use_container_width=True, hide_index=True)

        with col2:
            # Visualization based on data type
            col_data = df[column]

            if col_data.dtype in ['int64', 'float64']:
                # Histogram for numeric data - ADD UNIQUE KEY
                fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                st.plotly_chart(fig, use_container_width=True, key=f"histogram_{column}")
            else:
                # Bar chart for categorical data - ADD UNIQUE KEY
                value_counts = col_data.value_counts().head(10)  # Top 10 categories
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f"Top Categories in {column}"
                )
                st.plotly_chart(fig, use_container_width=True, key=f"bar_{column}")

        # Issues and recommendations
        issues = []
        recommendations = []

        if col_result['completeness_pct'] < 95:
            issues.append(
                f"High missing values: {col_result['null_count']} ({100 - col_result['completeness_pct']:.1f}%)")
            recommendations.append("Consider imputation or investigate data collection issues")

        if col_result['outlier_impact'] > 5:
            issues.append(
                f"High outlier rate: {col_result['outlier_count']} outliers ({col_result['outlier_impact']:.1f}%)")
            recommendations.append("Review outliers - consider removal or transformation")

        if not col_result['type_consistency']:
            issues.append(
                f"Type inconsistency: Current={col_result['data_type']}, Inferred={col_result['inferred_type']}")
            recommendations.append("Consider converting to the inferred data type")

        if col_data.dtype == 'object' and col_result['uniqueness_pct'] > 90:
            issues.append("Very high cardinality for categorical column")
            recommendations.append("Consider if this should be treated as an identifier rather than a feature")

        if issues:
            st.markdown("**⚠️ Issues Identified:**")
            for issue in issues:
                st.warning(issue)

        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in recommendations:
                st.info(rec)

        if not issues:
            st.success("✅ No major issues detected for this column!")

    def _render_dataset_validation(self, df):
        """Render dataset-level validation"""
        st.subheader("📊 Dataset-Level Validation")

        # Basic dataset metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_cells = df.size
            null_cells = df.isnull().sum().sum()
            null_rate = (null_cells / total_cells) * 100 if total_cells > 0 else 0
            st.metric("Overall Null Rate", f"{null_rate:.1f}%")

        with col2:
            duplicate_rows = df.duplicated().sum()
            duplicate_rate = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Duplicate Rows", f"{duplicate_rate:.1f}%")

        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

        with col4:
            cardinality_score = self._calculate_cardinality_health(df)
            st.metric("Cardinality Health", f"{cardinality_score:.1f}/100")

        # Class imbalance analysis (if applicable)
        st.markdown("### ⚖️ Class Balance Analysis")

        # Try to identify potential target columns
        potential_targets = self._identify_potential_targets(df)

        if potential_targets:
            target_col = st.selectbox("Select Potential Target Column", potential_targets)
            if target_col:
                self._analyze_class_balance(df, target_col)
        else:
            st.info("No clear target columns identified for class balance analysis.")

        # Multicollinearity analysis
        st.markdown("### 🔗 Multicollinearity Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            self._analyze_multicollinearity(df, numeric_cols)
        else:
            st.info("Need at least 2 numeric columns for multicollinearity analysis.")

        # Correlation matrix
        if len(numeric_cols) >= 2:
            st.markdown("### 📈 Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()

            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig.update_layout(height=max(400, len(numeric_cols) * 30))
            st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")

            # Highlight high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })

            if high_corr_pairs:
                st.warning("⚠️ High Correlations Detected (>0.8):")
                high_corr_df = pd.DataFrame(high_corr_pairs)
                st.dataframe(high_corr_df, use_container_width=True, hide_index=True)

    def _calculate_cardinality_health(self, df):
        """Calculate overall cardinality health score"""
        scores = []

        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)

            if df[col].dtype in ['int64', 'float64']:
                # For numeric columns, moderate uniqueness is good
                if 0.7 <= unique_ratio <= 0.95:
                    score = 100
                elif unique_ratio < 0.1:
                    score = 30  # Too few unique values
                elif unique_ratio > 0.99:
                    score = 70  # Might be an ID column
                else:
                    score = 80
            else:
                # For categorical columns, moderate cardinality is preferred
                if 0.01 <= unique_ratio <= 0.1:
                    score = 100
                elif unique_ratio > 0.8:
                    score = 30  # Too high cardinality
                else:
                    score = 70

            scores.append(score)

        return np.mean(scores) if scores else 0

    def _identify_potential_targets(self, df):
        """Identify potential target columns"""
        potential_targets = []

        for col in df.columns:
            # Check for common target column names
            if any(keyword in col.lower() for keyword in
                   ['target', 'label', 'class', 'outcome', 'result', 'prediction']):
                potential_targets.append(col)
                continue

            # Check for binary columns (potential classification targets)
            if df[col].nunique() == 2 and df[col].dtype in ['int64', 'float64', 'object', 'bool']:
                potential_targets.append(col)
                continue

            # Check for small number of categories (potential classification targets)
            if 2 <= df[col].nunique() <= 10 and df[col].dtype in ['object', 'category', 'int64']:
                potential_targets.append(col)

        return potential_targets

    def _analyze_class_balance(self, df, target_col):
        """Analyze class balance for a target column"""
        value_counts = df[target_col].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            # Class distribution table
            balance_df = pd.DataFrame({
                'Class': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(balance_df, use_container_width=True, hide_index=True)

            # Calculate imbalance ratio
            min_class_count = value_counts.min()
            max_class_count = value_counts.max()
            imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

            st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")

            if imbalance_ratio > 5:
                st.warning("⚠️ Significant class imbalance detected!")
            elif imbalance_ratio > 2:
                st.info("ℹ️ Moderate class imbalance present")
            else:
                st.success("✅ Classes are relatively balanced")

        with col2:
            # Pie chart - ADD UNIQUE KEY
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Class Distribution: {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"class_balance_{target_col}")

    def _analyze_multicollinearity(self, df, numeric_cols):
        """Analyze multicollinearity using VIF"""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Calculate VIF for each numeric column
            df_numeric = df[numeric_cols].dropna()

            if len(df_numeric) < 10:
                st.info("Not enough data for reliable multicollinearity analysis.")
                return

            vif_data = []
            for i, col in enumerate(df_numeric.columns):
                vif_value = variance_inflation_factor(df_numeric.values, i)
                vif_data.append({
                    'Feature': col,
                    'VIF': vif_value,
                    'Status': 'High' if vif_value > 10 else 'Moderate' if vif_value > 5 else 'Low'
                })

            vif_df = pd.DataFrame(vif_data)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(vif_df, use_container_width=True, hide_index=True)

                # Warnings
                high_vif = vif_df[vif_df['VIF'] > 10]
                if not high_vif.empty:
                    st.warning(f"⚠️ High multicollinearity detected in {len(high_vif)} features (VIF > 10)")
                    st.write("Consider removing or combining these features:")
                    for _, row in high_vif.iterrows():
                        st.write(f"- {row['Feature']}: VIF = {row['VIF']:.2f}")

            with col2:
                # VIF chart - ADD UNIQUE KEY
                fig = px.bar(
                    vif_df,
                    x='VIF',
                    y='Feature',
                    orientation='h',
                    title="Variance Inflation Factor (VIF)",
                    color='Status',
                    color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
                )
                fig.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
                fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="High Threshold")
                st.plotly_chart(fig, use_container_width=True, key="vif_chart")

        except Exception as e:
            st.error(f"Error calculating VIF: {str(e)}")

    def _render_robustness_scoring(self, df):
        """Render robustness scoring interface"""
        st.subheader("🎯 Data Robustness Scoring")

        # Calculate overall robustness scores
        column_scores = []

        for col in df.columns:
            col_validation = self._validate_column(df, col)
            column_scores.append({
                'Column': col,
                'Robustness Score': col_validation['overall_score']
            })

        scores_df = pd.DataFrame(column_scores)

        # Overall dataset robustness
        overall_robustness = scores_df['Robustness Score'].mean()

        # Display overall score with color coding
        col1, col2, col3 = st.columns(3)

        with col1:
            if overall_robustness >= 80:
                st.success(f"🟢 Overall Robustness: {overall_robustness:.1f}/100")
            elif overall_robustness >= 60:
                st.warning(f"🟡 Overall Robustness: {overall_robustness:.1f}/100")
            else:
                st.error(f"🔴 Overall Robustness: {overall_robustness:.1f}/100")

        with col2:
            high_quality_cols = len(scores_df[scores_df['Robustness Score'] >= 80])
            st.metric("High Quality Columns", f"{high_quality_cols}/{len(df.columns)}")

        with col3:
            low_quality_cols = len(scores_df[scores_df['Robustness Score'] < 60])
            st.metric("Needs Attention", f"{low_quality_cols}/{len(df.columns)}")

        # Column-wise robustness chart - ADD UNIQUE KEY
        fig = px.bar(
            scores_df.sort_values('Robustness Score', ascending=True),
            x='Robustness Score',
            y='Column',
            orientation='h',
            title="Column Robustness Scores",
            color='Robustness Score',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Needs Attention")
        fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="High Quality")
        st.plotly_chart(fig, use_container_width=True, key="robustness_scores")

        # Detailed robustness breakdown
        st.markdown("### 📊 Detailed Robustness Breakdown")

        # Create detailed breakdown
        detailed_scores = []
        for col in df.columns:
            col_validation = self._validate_column(df, col)
            detailed_scores.append({
                'Column': col,
                'Completeness': col_validation['completeness_pct'],
                'Uniqueness': col_validation['uniqueness_pct'],
                'Validity': col_validation['validity_score'],
                'Overall': col_validation['overall_score']
            })

        detailed_df = pd.DataFrame(detailed_scores)

        st.dataframe(
            detailed_df,
            use_container_width=True,
            column_config={
                "Completeness": st.column_config.ProgressColumn(
                    "Completeness",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "Uniqueness": st.column_config.ProgressColumn(
                    "Uniqueness",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "Validity": st.column_config.ProgressColumn(
                    "Validity",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "Overall": st.column_config.ProgressColumn(
                    "Overall Score",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                )
            },
            hide_index=True
        )

    def _render_validation_recommendations(self, df):
        """Render validation recommendations"""
        st.subheader("💡 Validation Recommendations")

        recommendations = []

        # Analyze each column and generate recommendations
        for col in df.columns:
            col_validation = self._validate_column(df, col)
            col_recommendations = self._generate_column_recommendations(df, col, col_validation)
            recommendations.extend(col_recommendations)

        # Dataset-level recommendations
        dataset_recommendations = self._generate_dataset_recommendations(df)
        recommendations.extend(dataset_recommendations)

        # Sort recommendations by priority
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }

                with st.expander(
                        f"{priority_color.get(rec['priority'], '🔵')} {rec['priority']} Priority: {rec['title']}",
                        expanded=(i < 3)):  # Expand first 3 recommendations
                    st.write(rec['description'])

                    if rec['action']:
                        st.markdown("**Suggested Action:**")
                        st.code(rec['action'])

                    if rec['impact']:
                        st.markdown("**Expected Impact:**")
                        st.info(rec['impact'])

            # Log validation recommendations
            self.history.log_step(
                "Data Validation",
                f"Generated {len(recommendations)} validation recommendations",
                {
                    "total_recommendations": len(recommendations),
                    "high_priority": sum(1 for r in recommendations if r['priority'] == 'High'),
                    "medium_priority": sum(1 for r in recommendations if r['priority'] == 'Medium'),
                    "low_priority": sum(1 for r in recommendations if r['priority'] == 'Low')
                },
                "success"
            )
        else:
            st.success("🎉 No major validation issues found! Your dataset appears to be in good shape.")

    def _generate_column_recommendations(self, df, column, col_validation):
        """Generate recommendations for a specific column"""
        recommendations = []

        # High missing values
        if col_validation['completeness_pct'] < 50:
            recommendations.append({
                'priority': 'High',
                'title': f"High Missing Values in {column}",
                'description': f"Column '{column}' has {col_validation['null_count']} missing values ({100 - col_validation['completeness_pct']:.1f}%). This severely impacts data quality.",
                'action': f"# Consider dropping the column or investigating data collection\\ndf.drop(columns=['{column}'], inplace=True)\\n# OR\\n# Investigate and fix data collection process",
                'impact': "Removing high-missing columns or fixing data collection will significantly improve model performance."
            })
        elif col_validation['completeness_pct'] < 80:
            recommendations.append({
                'priority': 'Medium',
                'title': f"Moderate Missing Values in {column}",
                'description': f"Column '{column}' has {col_validation['null_count']} missing values ({100 - col_validation['completeness_pct']:.1f}%).",
                'action': f"# Apply appropriate imputation\\nfrom sklearn.impute import SimpleImputer\\nimputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'\\ndf['{column}'] = imputer.fit_transform(df[['{column}']])",
                'impact': "Proper imputation will allow you to retain the column while improving completeness."
            })

        # High outlier impact
        if col_validation['outlier_impact'] > 10:
            recommendations.append({
                'priority': 'Medium',
                'title': f"High Outlier Rate in {column}",
                'description': f"Column '{column}' has {col_validation['outlier_count']} outliers ({col_validation['outlier_impact']:.1f}% of data).",
                'action': f"# Review outliers and consider treatment\\n# Option 1: Remove outliers\\nQ1 = df['{column}'].quantile(0.25)\\nQ3 = df['{column}'].quantile(0.75)\\nIQR = Q3 - Q1\\ndf = df[(df['{column}'] >= Q1 - 1.5*IQR) & (df['{column}'] <= Q3 + 1.5*IQR)]\\n# Option 2: Cap outliers\\ndf['{column}'] = df['{column}'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)",
                'impact': "Outlier treatment will improve model stability and performance."
            })

        # Type inconsistency
        if not col_validation['type_consistency']:
            recommendations.append({
                'priority': 'Medium',
                'title': f"Data Type Issue in {column}",
                'description': f"Column '{column}' is stored as {col_validation['data_type']} but appears to be {col_validation['inferred_type']}.",
                'action': f"# Convert to appropriate data type\\ndf['{column}'] = pd.to_numeric(df['{column}'], errors='coerce')  # for numeric\\n# OR\\ndf['{column}'] = pd.to_datetime(df['{column}'], errors='coerce')  # for datetime",
                'impact': "Correct data types enable proper analysis and modeling."
            })

        # High cardinality categorical
        if (df[column].dtype == 'object' and
                col_validation['uniqueness_pct'] > 80 and
                col_validation['unique_count'] > 50):
            recommendations.append({
                'priority': 'Low',
                'title': f"High Cardinality in Categorical Column {column}",
                'description': f"Column '{column}' has {col_validation['unique_count']} unique values ({col_validation['uniqueness_pct']:.1f}% unique). This may cause issues in modeling.",
                'action': f"# Consider treating as identifier or applying dimensionality reduction\\n# Option 1: Drop if it's an ID column\\ndf.drop(columns=['{column}'], inplace=True)\\n# Option 2: Group rare categories\\nvalue_counts = df['{column}'].value_counts()\\nrare_categories = value_counts[value_counts < 10].index\\ndf['{column}'] = df['{column}'].replace(rare_categories, 'Other')",
                'impact': "Reducing cardinality will improve model training efficiency and prevent overfitting."
            })

        return recommendations

    def _generate_dataset_recommendations(self, df):
        """Generate dataset-level recommendations"""
        recommendations = []

        # Overall null rate
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_rate = (null_cells / total_cells) * 100 if total_cells > 0 else 0

        if null_rate > 20:
            recommendations.append({
                'priority': 'High',
                'title': "High Overall Missing Value Rate",
                'description': f"Dataset has {null_rate:.1f}% missing values overall. This indicates systematic data quality issues.",
                'action': "# Comprehensive data cleaning needed\\n# 1. Identify patterns in missing data\\n# 2. Implement systematic imputation strategy\\n# 3. Consider data collection improvements",
                'impact': "Comprehensive missing value treatment will significantly improve dataset quality."
            })

        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_rate = (duplicate_count / len(df)) * 100
            priority = 'High' if duplicate_rate > 10 else 'Medium'

            recommendations.append({
                'priority': priority,
                'title': "Duplicate Rows Present",
                'description': f"Found {duplicate_count} duplicate rows ({duplicate_rate:.1f}% of dataset).",
                'action': "# Remove duplicate rows\\ndf_cleaned = df.drop_duplicates()\\nprint(f'Removed {len(df) - len(df_cleaned)} duplicate rows')",
                'impact': "Removing duplicates will prevent data leakage and improve model reliability."
            })

        # Memory usage warning
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > 100:
            recommendations.append({
                'priority': 'Low',
                'title': "High Memory Usage",
                'description': f"Dataset uses {memory_mb:.1f} MB of memory. Consider optimization for large-scale processing.",
                'action': "# Optimize data types\\n# Convert int64 to int32 where possible\\n# Use categorical dtype for low-cardinality string columns\\ndf['category_col'] = df['category_col'].astype('category')",
                'impact': "Memory optimization will improve processing speed and reduce resource requirements."
            })

        # Class imbalance (if binary target detected)
        potential_targets = self._identify_potential_targets(df)
        for target in potential_targets:
            if df[target].nunique() == 2:
                value_counts = df[target].value_counts()
                imbalance_ratio = value_counts.max() / value_counts.min()

                if imbalance_ratio > 5:
                    recommendations.append({
                        'priority': 'Medium',
                        'title': f"Class Imbalance in {target}",
                        'description': f"Target column '{target}' has significant class imbalance (ratio: {imbalance_ratio:.1f}:1).",
                        'action': f"# Address class imbalance\\nfrom imblearn.over_sampling import SMOTE\\nfrom imblearn.under_sampling import RandomUnderSampler\\n# Apply SMOTE or undersampling based on dataset size",
                        'impact': "Balancing classes will improve model performance on minority class."
                    })

        return recommendations