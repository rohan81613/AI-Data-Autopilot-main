import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Try to import WordCloud - handle if not installed
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Try to import kaleido for export functionality
try:
    import plotly.io as pio
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

from pipeline_history import PipelineHistory


class EnhancedVisualization:
    def __init__(self):
        self.history = PipelineHistory()
        # Dashboard templates
        self.dashboard_templates = {
            "Sales Analytics": ["Bar Chart", "Line Chart", "Pie Chart", "Histogram"],
            "Marketing Performance": ["Scatter Plot", "Bubble Chart", "Heatmap", "Funnel Chart"],
            "Financial Overview": ["Candlestick Chart", "Waterfall Chart", "Gauge Chart", "KPI Cards"],
            "Customer Analysis": ["Treemap", "Sunburst", "Sankey Diagram", "Radar Chart"]
        }

    def render_visualization_ui(self):
        """Render the enhanced visualization interface"""
        
        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Visualization", dataset_names, key="enhanced_viz_dataset_selector")

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Dataset:** {selected_dataset} ({df.shape[0]} rows × {df.shape[1]} columns)")

        # Enhanced tabs for different visualization sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "📈 Interactive Chart Builder",
            "🔍 Auto Audit & Recommendations",
            "🎨 Dashboard Templates",
            "🛠️ Custom Visualization Builder"
        ])

        with tab1:
            self._render_enhanced_data_overview(df)

        with tab2:
            self._render_interactive_chart_builder(df, selected_dataset)

        with tab3:
            self._render_auto_audit(df, selected_dataset)

        with tab4:
            self._render_dashboard_templates(df, selected_dataset)

        with tab5:
            self._render_custom_visualization_builder(df, selected_dataset)

    def _safe_nunique(self, series):
        """Safely get number of unique values, handling unhashable types"""
        try:
            return int(series.nunique())
        except TypeError:
            # Handle unhashable types (like lists)
            try:
                # Convert to string representation and count unique strings
                return int(series.astype(str).nunique())
            except Exception:
                return "N/A"

    def _safe_sample_values(self, series, n=3):
        """Safely get sample values, handling unhashable types"""
        try:
            sample_values = series.dropna().head(n).tolist()
            return str(sample_values)[:50] + "..."
        except Exception:
            try:
                # Convert to string and sample
                sample_values = series.dropna().astype(str).head(n).tolist()
                return str(sample_values)[:50] + "..."
            except Exception:
                return "Unable to display"

    def _convert_for_plotly(self, data):
        """Convert data types for Plotly compatibility"""
        if isinstance(data, pd.DataFrame):
            # Convert all dtypes to native Python types
            converted_data = data.copy()
            for col in converted_data.columns:
                try:
                    if converted_data[col].dtype == 'object':
                        # Check if column contains lists or other unhashable types
                        try:
                            converted_data[col] = converted_data[col].astype(str)
                        except Exception:
                            # If conversion fails, convert each value individually
                            converted_data[col] = converted_data[col].apply(lambda x: str(x) if x is not None else '')
                    elif converted_data[col].dtype.name.startswith('int'):
                        converted_data[col] = converted_data[col].astype(int)
                    elif converted_data[col].dtype.name.startswith('float'):
                        converted_data[col] = converted_data[col].astype(float)
                except Exception:
                    # If all else fails, convert to string
                    converted_data[col] = converted_data[col].astype(str)
            return converted_data
        return data

    def _render_enhanced_data_overview(self, df):
        """Render enhanced data overview section with interactive filtering"""
        st.subheader("📊 Enhanced Data Overview")

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            st.metric("Categorical Columns", len(categorical_cols))

        # Interactive filtering controls
        st.markdown("### 🔍 Interactive Data Filtering")
        
        # Column-based filtering
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            filter_columns = st.multiselect("Select columns to filter", df.columns.tolist())
        
        with filter_col2:
            if filter_columns:
                selected_filter_col = st.selectbox("Select column to filter", filter_columns)
                if selected_filter_col:
                    unique_values = df[selected_filter_col].dropna().unique()
                    if len(unique_values) <= 100:  # Only show if not too many values
                        filter_values = st.multiselect(f"Select values for {selected_filter_col}", unique_values)
                    else:
                        st.info(f"Too many unique values in {selected_filter_col} to display. Use search below.")
                        search_value = st.text_input(f"Search values in {selected_filter_col}")
                        if search_value:
                            filter_values = [v for v in unique_values if search_value.lower() in str(v).lower()]
                        else:
                            filter_values = []
        
        with filter_col3:
            # Numeric range filtering
            numeric_filter_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_filter_cols:
                numeric_filter_col = st.selectbox("Select numeric column to filter by range", numeric_filter_cols)
                if numeric_filter_col:
                    min_val = float(df[numeric_filter_col].min())
                    max_val = float(df[numeric_filter_col].max())
                    range_vals = st.slider(f"Select range for {numeric_filter_col}", 
                                         min_val, max_val, (min_val, max_val))
        
        # Apply filters
        filtered_df = df.copy()
        if filter_columns and 'filter_values' in locals() and filter_values:
            filtered_df = filtered_df[filtered_df[selected_filter_col].isin(filter_values)]
        
        if 'numeric_filter_col' in locals() and 'range_vals' in locals():
            filtered_df = filtered_df[
                (filtered_df[numeric_filter_col] >= range_vals[0]) & 
                (filtered_df[numeric_filter_col] <= range_vals[1])
            ]
        
        st.markdown(f"**Filtered Data:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
        
        # Export filtered data
        if st.button("💾 Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Column information with safe handling
        st.markdown("### 📋 Column Information")

        column_info = []
        for col in filtered_df.columns:
            try:
                col_info = {
                    'Column': str(col),  # Convert to string
                    'Data Type': str(filtered_df[col].dtype),  # Convert to string
                    'Non-Null Count': int(filtered_df[col].count()),  # Convert to int
                    'Null Count': int(filtered_df[col].isnull().sum()),  # Convert to int
                    'Unique Values': self._safe_nunique(filtered_df[col]),  # Safe unique count
                    'Sample Values': self._safe_sample_values(filtered_df[col])  # Safe sample values
                }
                column_info.append(col_info)
            except Exception as e:
                # Fallback for problematic columns
                col_info = {
                    'Column': str(col),
                    'Data Type': str(filtered_df[col].dtype),
                    'Non-Null Count': "Error",
                    'Null Count': "Error",
                    'Unique Values': "Error",
                    'Sample Values': f"Error: {str(e)[:30]}..."
                }
                column_info.append(col_info)

        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)

        # Missing values chart
        if filtered_df.isnull().sum().sum() > 0:
            st.markdown("### 🕳️ Missing Values Analysis")

            missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]

            if len(missing_data) > 0:
                # Convert to native Python types
                missing_df = pd.DataFrame({
                    'Column': [str(col) for col in missing_data.index],  # Convert to string
                    'Missing_Count': [int(count) for count in missing_data.values]  # Convert to int
                })

                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing_Count',
                    title="Missing Values by Column",
                    labels={'Missing_Count': 'Missing Count'}
                )
                
                # Add export button for chart
                self._add_chart_export_options(fig, "missing_values_chart")
                
                st.plotly_chart(fig, use_container_width=True, key="enhanced_overview_missing_values")

        # Statistical summary for numeric columns
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("### 📈 Statistical Summary (Numeric Columns)")
            try:
                summary_df = filtered_df[numeric_cols].describe()
                # Convert all values to native Python types
                for col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(float)
                st.dataframe(summary_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating statistical summary: {str(e)}")

        # Value counts for categorical columns (safe handling)
        categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            # Filter out columns that might contain unhashable types
            safe_categorical_cols = []
            for col in categorical_cols:
                try:
                    # Test if we can get value counts
                    test_counts = filtered_df[col].value_counts().head(1)
                    safe_categorical_cols.append(col)
                except Exception:
                    # Skip columns with unhashable types
                    continue

            if safe_categorical_cols:
                st.markdown("### 🏷️ Category Distributions")

                selected_cat_col = st.selectbox("Select Categorical Column", safe_categorical_cols,
                                                key="enhanced_overview_cat_selector")
                if selected_cat_col:
                    try:
                        value_counts = filtered_df[selected_cat_col].value_counts().head(10)

                        # Convert to DataFrame with native Python types
                        cat_df = pd.DataFrame({
                            'Category': [str(cat) for cat in value_counts.index],
                            'Count': [int(count) for count in value_counts.values]
                        })

                        fig = px.bar(
                            cat_df,
                            x='Count',
                            y='Category',
                            orientation='h',
                            title=f"Top 10 Categories in {selected_cat_col}"
                        )
                        
                        # Add export button for chart
                        self._add_chart_export_options(fig, "category_distribution_chart")
                        
                        st.plotly_chart(fig, use_container_width=True, key="enhanced_overview_category_dist")
                    except Exception as e:
                        st.error(f"Error creating category distribution: {str(e)}")
            else:
                st.info("No suitable categorical columns found for distribution analysis.")

    def _add_chart_export_options(self, fig, chart_name):
        """Add export options for charts"""
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"💾 Save as PNG ({chart_name})", key=f"save_png_{chart_name}"):
                if HAS_KALEIDO:
                    buf = io.BytesIO()
                    fig.write_image(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download PNG",
                        data=buf,
                        file_name=f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Export functionality requires kaleido. Install with: pip install kaleido")
        with col2:
            if st.button(f"💾 Save as SVG ({chart_name})", key=f"save_svg_{chart_name}"):
                if HAS_KALEIDO:
                    buf = io.BytesIO()
                    fig.write_image(buf, format="svg")
                    buf.seek(0)
                    st.download_button(
                        label="Download SVG",
                        data=buf,
                        file_name=f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning("Export functionality requires kaleido. Install with: pip install kaleido")
        with col3:
            if st.button(f"📋 Copy Chart Data ({chart_name})", key=f"copy_data_{chart_name}"):
                # This would copy chart data to clipboard in a real implementation
                st.info("Chart data copied to clipboard (simulated)")

    def _render_interactive_chart_builder(self, df, dataset_name):
        """Render interactive chart builder with enhanced features"""
        st.subheader("📈 Enhanced Interactive Chart Builder")

        # Get safe columns for visualization
        numeric_cols = self._get_safe_columns(df, [np.number])
        categorical_cols = self._get_safe_columns(df, ['object', 'category'])

        if not numeric_cols and not categorical_cols:
            st.error("No suitable columns found for visualization. The dataset may contain complex data types.")
            return

        # Chart type categorization with more options
        categorical_charts = {
            'Basic': ['Bar Chart', 'Column Chart', 'Pie Chart', 'Donut Chart'],
            'Intermediate': ['Stacked Bar', 'Grouped Bar', 'Lollipop Chart'],
            'Advanced': ['Treemap', 'Sunburst', 'Sankey Diagram', 'Funnel Chart', 'Waterfall Chart']
        }

        numerical_charts = {
            'Basic': ['Histogram', 'Line Chart', 'Scatter Plot', 'Box Plot'],
            'Intermediate': ['Area Chart', 'Violin Plot', 'Heatmap', 'Bubble Chart'],
            'Advanced': ['3D Scatter', 'Contour Plot', 'Parallel Coordinates', 'Radar Chart', 
                        'Candlestick Chart', 'Gauge Chart']
        }

        # Add WordCloud conditionally
        if HAS_WORDCLOUD and categorical_cols:
            categorical_charts['Basic'].append('Word Cloud')

        # Chart type selection
        col1, col2 = st.columns(2)

        with col1:
            available_data_types = []
            if categorical_cols:
                available_data_types.append("Categorical")
            if numeric_cols:
                available_data_types.append("Numerical")
            if categorical_cols and numeric_cols:
                available_data_types.append("Mixed")

            if not available_data_types:
                st.error("No suitable data types found for visualization.")
                return

            data_type = st.selectbox("Data Type", available_data_types, key="enhanced_chart_data_type")

            if data_type == "Categorical":
                available_charts = categorical_charts
            elif data_type == "Numerical":
                available_charts = numerical_charts
            else:  # Mixed
                available_charts = {**categorical_charts, **numerical_charts}

        with col2:
            complexity = st.selectbox("Complexity Level", ["Basic", "Intermediate", "Advanced"], key="enhanced_chart_complexity")
            chart_options = available_charts.get(complexity, [])

            if not chart_options:
                st.warning(f"No {complexity} charts available for {data_type} data type.")
                return

            selected_chart = st.selectbox("Chart Type", chart_options, key="enhanced_chart_type_selector")

        # Column selection based on chart type
        self._render_enhanced_chart_configuration(df, selected_chart, dataset_name, numeric_cols, categorical_cols)

    def _get_safe_columns(self, df, include_types):
        """Get columns that are safe for visualization (no unhashable types)"""
        safe_cols = []
        candidate_cols = df.select_dtypes(include=include_types).columns.tolist()

        for col in candidate_cols:
            try:
                # Test if column is safe for operations
                if include_types == ['object', 'category']:
                    # Test value_counts for categorical
                    df[col].value_counts().head(1)
                elif include_types == [np.number]:
                    # Test basic operations for numeric
                    df[col].mean()
                safe_cols.append(col)
            except Exception:
                # Skip problematic columns
                continue

        return safe_cols

    def _render_enhanced_chart_configuration(self, df, chart_type, dataset_name, numeric_cols, categorical_cols):
        """Render enhanced chart configuration interface"""

        all_cols = numeric_cols + categorical_cols

        if not all_cols:
            st.error("No suitable columns available for chart configuration.")
            return

        # Chart-specific configuration
        if chart_type in ['Bar Chart', 'Column Chart', 'Pie Chart', 'Donut Chart']:
            if not categorical_cols:
                st.error("Categorical columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Select Category Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                if chart_type in ['Bar Chart', 'Column Chart']:
                    y_column = st.selectbox("Select Value Column (optional)", [None] + numeric_cols,
                                            key=f"enhanced_{chart_type}_y_col")
                    aggregation = st.selectbox("Aggregation", ["count", "sum", "mean", "median"],
                                               key=f"enhanced_{chart_type}_agg")
                else:
                    y_column = None
                    aggregation = "count"

        elif chart_type in ['Stacked Bar', 'Grouped Bar']:
            if not categorical_cols or not numeric_cols:
                st.error("Both categorical and numeric columns are required for this chart type.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            with col3:
                color_column = st.selectbox("Stack/Group by", categorical_cols, key=f"enhanced_{chart_type}_color_col")
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"], key=f"enhanced_{chart_type}_agg")

        elif chart_type == 'Lollipop Chart':
            if not categorical_cols or not numeric_cols:
                st.error("Both categorical and numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Category Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = None

        elif chart_type in ['Histogram', 'Box Plot', 'Violin Plot']:
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            x_column = st.selectbox("Select Numeric Column", numeric_cols, key=f"enhanced_{chart_type}_x_col")
            y_column = None
            aggregation = None

        elif chart_type in ['Line Chart', 'Scatter Plot', 'Bubble Chart']:
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = None

        elif chart_type == 'Area Chart':
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = None

        elif chart_type == 'Heatmap':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for heatmap.")
                return

            x_column = st.selectbox("X-axis Column", all_cols, key=f"enhanced_{chart_type}_x_col")
            y_column = st.selectbox("Y-axis Column", all_cols, key=f"enhanced_{chart_type}_y_col")
            z_column = st.selectbox("Value Column", numeric_cols, key=f"enhanced_{chart_type}_z_col")
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"], key=f"enhanced_{chart_type}_agg")

        elif chart_type == '3D Scatter':
            if len(numeric_cols) < 3:
                st.error("At least 3 numeric columns are required for 3D scatter plot.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis", numeric_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            with col3:
                z_column = st.selectbox("Z-axis", numeric_cols, key=f"enhanced_{chart_type}_z_col")
            aggregation = None

        elif chart_type == 'Contour Plot':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for contour plot.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis", numeric_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            with col3:
                z_column = st.selectbox("Z-axis (optional)", [None] + numeric_cols, key=f"enhanced_{chart_type}_z_col")
            aggregation = None

        elif chart_type == 'Parallel Coordinates':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for parallel coordinates.")
                return

            selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, default=numeric_cols[:4],
                                           key=f"enhanced_{chart_type}_cols")
            if not selected_cols:
                st.warning("Please select at least 2 columns.")
                return
            x_column = selected_cols
            y_column = None
            aggregation = None

        elif chart_type == 'Radar Chart':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for radar chart.")
                return

            selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, default=numeric_cols[:5],
                                           key=f"enhanced_{chart_type}_cols")
            if not selected_cols:
                st.warning("Please select at least 2 columns.")
                return
            category_col = st.selectbox("Category Column (optional)", [None] + categorical_cols,
                                        key=f"enhanced_{chart_type}_cat_col")
            x_column = selected_cols
            y_column = category_col
            aggregation = None

        elif chart_type in ['Treemap', 'Sunburst']:
            if not categorical_cols:
                st.error("Categorical columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Category Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column (optional)", [None] + numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = "sum" if y_column else "count"

        elif chart_type == 'Sankey Diagram':
            if len(categorical_cols) < 2:
                st.error("At least 2 categorical columns are required for Sankey diagram.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                source_col = st.selectbox("Source Column", categorical_cols, key=f"enhanced_{chart_type}_source_col")
            with col2:
                target_col = st.selectbox("Target Column", categorical_cols, key=f"enhanced_{chart_type}_target_col")
            with col3:
                value_col = st.selectbox("Value Column (optional)", [None] + numeric_cols,
                                         key=f"enhanced_{chart_type}_value_col")
            x_column = source_col
            y_column = target_col
            z_column = value_col
            aggregation = "sum" if value_col else "count"

        elif chart_type == 'Word Cloud' and HAS_WORDCLOUD:
            if not categorical_cols:
                st.error("Text columns are required for word cloud.")
                return

            x_column = st.selectbox("Text Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            y_column = None
            aggregation = None

        elif chart_type == 'Funnel Chart':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for funnel chart.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Stage Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = "sum" if y_column else "count"

        elif chart_type == 'Waterfall Chart':
            if not numeric_cols:
                st.error("Numeric columns are required for waterfall chart.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Category Column", categorical_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column", numeric_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = "sum" if y_column else "count"

        elif chart_type == 'Candlestick Chart':
            if len(numeric_cols) < 4:
                st.error("At least 4 numeric columns are required for candlestick chart (Open, High, Low, Close).")
                return

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                open_col = st.selectbox("Open Column", numeric_cols, key=f"enhanced_{chart_type}_open_col")
            with col2:
                high_col = st.selectbox("High Column", numeric_cols, key=f"enhanced_{chart_type}_high_col")
            with col3:
                low_col = st.selectbox("Low Column", numeric_cols, key=f"enhanced_{chart_type}_low_col")
            with col4:
                close_col = st.selectbox("Close Column", numeric_cols, key=f"enhanced_{chart_type}_close_col")
            x_column = open_col
            y_column = [open_col, high_col, low_col, close_col]
            aggregation = None

        elif chart_type == 'Gauge Chart':
            if not numeric_cols:
                st.error("Numeric columns are required for gauge chart.")
                return

            x_column = st.selectbox("Value Column", numeric_cols, key=f"enhanced_{chart_type}_x_col")
            y_column = None
            aggregation = None

        else:
            # Default configuration for other chart types
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"enhanced_{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column (optional)", [None] + all_cols, key=f"enhanced_{chart_type}_y_col")
            aggregation = None

        # Additional chart options
        with st.expander("🎨 Advanced Chart Options"):
            if chart_type not in ['Parallel Coordinates', 'Radar Chart', 'Sankey Diagram']:
                color_column = st.selectbox("Color by Column (optional)", [None] + all_cols,
                                            key=f"enhanced_{chart_type}_color_opt")
            else:
                color_column = None

            size_column = st.selectbox("Size by Column (optional)", [None] + numeric_cols,
                                       key=f"enhanced_{chart_type}_size_opt") if chart_type in ['Scatter Plot',
                                                                                       'Bubble Chart'] else None

            # Chart styling options
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                chart_title = st.text_input("Chart Title", value=f"{chart_type} - {x_column}",
                                            key=f"enhanced_{chart_type}_title")
            with col2:
                chart_height = st.number_input("Chart Height (px)", value=500, min_value=300, max_value=1000,
                                               key=f"enhanced_{chart_type}_height")
            with col3:
                show_legend = st.checkbox("Show Legend", value=True, key=f"enhanced_{chart_type}_legend")
            with col4:
                chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                                          key=f"enhanced_{chart_type}_theme")

        # Real-time update options
        with st.expander("⏱️ Real-time Updates"):
            enable_realtime = st.checkbox("Enable Real-time Updates", key=f"enhanced_{chart_type}_realtime")
            if enable_realtime:
                update_interval = st.slider("Update Interval (seconds)", 1, 60, 5, key=f"enhanced_{chart_type}_interval")
                st.info(f"Chart will update every {update_interval} seconds")

        # Generate chart button
        if st.button("🎨 Generate Chart", type="primary", key=f"enhanced_{chart_type}_generate"):
            if x_column:
                self._generate_enhanced_chart(
                    df, chart_type, x_column, y_column,
                    color_column, size_column, aggregation,
                    chart_title, chart_height, show_legend,
                    chart_theme, dataset_name, locals()
                )
            else:
                st.error("Please select at least one column for the chart.")

    def _generate_enhanced_chart(self, df, chart_type, x_col, y_col, color_col, size_col,
                        aggregation, title, height, show_legend, theme, dataset_name, params={}):
        """Generate the specified enhanced chart"""

        try:
            fig = None

            # Prepare data and clean for Plotly compatibility
            chart_data = df.copy()
            chart_data = self._convert_for_plotly(chart_data)

            # Handle missing values for basic charts
            if y_col and isinstance(x_col, str):
                chart_data = chart_data.dropna(subset=[x_col, y_col])
            elif isinstance(x_col, str):
                chart_data = chart_data.dropna(subset=[x_col])

            # Generate chart based on type
            if chart_type == "Bar Chart":
                fig = self._create_bar_chart(chart_data, x_col, y_col, color_col, aggregation, title, height)

            elif chart_type == "Column Chart":
                fig = self._create_column_chart(chart_data, x_col, y_col, color_col, aggregation, title, height)

            elif chart_type == "Stacked Bar":
                fig = self._create_stacked_bar_chart(chart_data, x_col, y_col, color_col, aggregation, title, height)

            elif chart_type == "Grouped Bar":
                fig = self._create_grouped_bar_chart(chart_data, x_col, y_col, color_col, aggregation, title, height)

            elif chart_type == "Lollipop Chart":
                fig = self._create_lollipop_chart(chart_data, x_col, y_col, title, height)

            elif chart_type == "Pie Chart":
                fig = self._create_pie_chart(chart_data, x_col, title, height)

            elif chart_type == "Donut Chart":
                fig = self._create_donut_chart(chart_data, x_col, title, height)

            elif chart_type == "Histogram":
                fig = self._create_histogram(chart_data, x_col, color_col, title, height)

            elif chart_type == "Line Chart":
                fig = self._create_line_chart(chart_data, x_col, y_col, color_col, title, height)

            elif chart_type == "Area Chart":
                fig = self._create_area_chart(chart_data, x_col, y_col, color_col, title, height)

            elif chart_type == "Scatter Plot":
                fig = self._create_scatter_plot(chart_data, x_col, y_col, color_col, size_col, title, height)

            elif chart_type == "Bubble Chart":
                fig = self._create_bubble_chart(chart_data, x_col, y_col, size_col, color_col, title, height)

            elif chart_type == "Box Plot":
                fig = self._create_box_plot(chart_data, x_col, color_col, title, height)

            elif chart_type == "Violin Plot":
                fig = self._create_violin_plot(chart_data, x_col, color_col, title, height)

            elif chart_type == "Heatmap":
                fig = self._create_heatmap(chart_data, x_col, y_col, params.get('z_column'), aggregation, title, height)

            elif chart_type == "3D Scatter":
                fig = self._create_3d_scatter_plot(chart_data, x_col, y_col, params.get('z_column'), color_col, title,
                                                   height)

            elif chart_type == "Contour Plot":
                fig = self._create_contour_plot(chart_data, x_col, y_col, params.get('z_column'), title, height)

            elif chart_type == "Parallel Coordinates":
                fig = self._create_parallel_coordinates(chart_data, x_col, color_col, title, height)

            elif chart_type == "Radar Chart":
                fig = self._create_radar_chart(chart_data, x_col, y_col, title, height)

            elif chart_type == "Treemap":
                fig = self._create_treemap(chart_data, x_col, y_col, title, height)

            elif chart_type == "Sunburst":
                fig = self._create_sunburst(chart_data, x_col, y_col, title, height)

            elif chart_type == "Sankey Diagram":
                fig = self._create_sankey_diagram(chart_data, x_col, y_col, params.get('z_column'), title, height)

            elif chart_type == "Word Cloud" and HAS_WORDCLOUD:
                self._create_word_cloud(chart_data, x_col, title)
                return  # Word cloud handled separately

            elif chart_type == "Funnel Chart":
                fig = self._create_funnel_chart(chart_data, x_col, y_col, title, height)

            elif chart_type == "Waterfall Chart":
                fig = self._create_waterfall_chart(chart_data, x_col, y_col, title, height)

            elif chart_type == "Candlestick Chart":
                fig = self._create_candlestick_chart(chart_data, x_col, y_col, title, height)

            elif chart_type == "Gauge Chart":
                fig = self._create_gauge_chart(chart_data, x_col, title, height)

            else:
                st.warning(f"Chart type '{chart_type}' is not yet implemented.")
                return

            # Display the chart
            if fig:
                fig.update_layout(
                    title=title,
                    height=height,
                    showlegend=show_legend,
                    template=theme
                )
                # Use unique key for each chart
                chart_key = f"enhanced_{chart_type.lower().replace(' ', '_')}_{x_col}_{dataset_name}_{id(fig)}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)

                # Add export options
                self._add_chart_export_options(fig, f"{chart_type.lower().replace(' ', '_')}_{x_col}")

                # Store chart in session state
                if 'charts' not in st.session_state:
                    st.session_state.charts = []

                chart_info = {
                    'chart_type': chart_type,
                    'dataset': dataset_name,
                    'x_column': x_col,
                    'y_column': y_col,
                    'color_column': color_col,
                    'size_column': size_col,
                    'title': title,
                    'figure': fig
                }
                st.session_state.charts.append(chart_info)

                # Log to history
                self.history.log_step(
                    "Chart Generation",
                    f"Created {chart_type} for {x_col}",
                    {
                        "chart_type": chart_type,
                        "x_column": x_col,
                        "y_column": y_col,
                        "dataset": dataset_name
                    },
                    "success"
                )

                st.success(f"✅ {chart_type} generated successfully!")

        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")
            st.error(f"Debug info: Chart type: {chart_type}, X column: {x_col}, Y column: {y_col}")
            self.history.log_step(
                "Chart Generation",
                f"Failed to create {chart_type}",
                {"error": str(e)},
                "error"
            )

    # === BASIC CHART IMPLEMENTATIONS ===

    def _create_bar_chart(self, df, x_col, y_col, color_col, aggregation, title, height):
        """Create a bar chart"""
        try:
            if y_col and aggregation and aggregation != "count":
                data = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
            elif y_col:
                data = df
            else:
                data = df[x_col].value_counts().reset_index()
                data.columns = ['Category', 'Count']
                x_col = 'Category'
                y_col = 'Count'
                color_col = None

            fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None

    def _create_column_chart(self, df, x_col, y_col, color_col, aggregation, title, height):
        """Create a column chart (same as bar but vertical)"""
        return self._create_bar_chart(df, x_col, y_col, color_col, aggregation, title, height)

    def _create_stacked_bar_chart(self, df, x_col, y_col, color_col, aggregation, title, height):
        """Create a stacked bar chart"""
        try:
            if aggregation == "count":
                data = df.groupby([x_col, color_col]).size().reset_index(name='count')
                y_col = 'count'
            else:
                data = df.groupby([x_col, color_col])[y_col].agg(aggregation).reset_index()

            fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating stacked bar chart: {str(e)}")
            return None

    def _create_grouped_bar_chart(self, df, x_col, y_col, color_col, aggregation, title, height):
        """Create a grouped bar chart"""
        try:
            if aggregation == "count":
                data = df.groupby([x_col, color_col]).size().reset_index(name='count')
                y_col = 'count'
            else:
                data = df.groupby([x_col, color_col])[y_col].agg(aggregation).reset_index()

            fig = px.bar(data, x=x_col, y=y_col, color=color_col, barmode='group', title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating grouped bar chart: {str(e)}")
            return None

    def _create_lollipop_chart(self, df, x_col, y_col, title, height):
        """Create a lollipop chart"""
        try:
            # Aggregate data
            data = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False)

            fig = go.Figure()

            # Add lines (sticks)
            for i, row in data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row[x_col], row[x_col]],
                    y=[0, row[y_col]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))

            # Add circles (lollipops)
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(size=10, color='red'),
                showlegend=False
            ))

            fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
            return fig
        except Exception as e:
            st.error(f"Error creating lollipop chart: {str(e)}")
            return None

    def _create_pie_chart(self, df, x_col, title, height):
        """Create a pie chart"""
        try:
            data = df[x_col].value_counts()
            # Convert to native Python types
            values = [int(v) for v in data.values]
            names = [str(n) for n in data.index]

            fig = px.pie(values=values, names=names, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None

    def _create_donut_chart(self, df, x_col, title, height):
        """Create a donut chart"""
        try:
            data = df[x_col].value_counts()
            # Convert to native Python types
            values = [int(v) for v in data.values]
            names = [str(n) for n in data.index]

            fig = px.pie(values=values, names=names, title=title, hole=0.4)
            return fig
        except Exception as e:
            st.error(f"Error creating donut chart: {str(e)}")
            return None

    def _create_histogram(self, df, x_col, color_col, title, height):
        """Create a histogram"""
        try:
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return None

    def _create_line_chart(self, df, x_col, y_col, color_col, title, height):
        """Create a line chart"""
        try:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating line chart: {str(e)}")
            return None

    def _create_area_chart(self, df, x_col, y_col, color_col, title, height):
        """Create an area chart"""
        try:
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating area chart: {str(e)}")
            return None

    def _create_scatter_plot(self, df, x_col, y_col, color_col, size_col, title, height):
        """Create a scatter plot"""
        try:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            return None

    def _create_bubble_chart(self, df, x_col, y_col, size_col, color_col, title, height):
        """Create a bubble chart"""
        try:
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title=title,
                             size_max=60)
            return fig
        except Exception as e:
            st.error(f"Error creating bubble chart: {str(e)}")
            return None

    def _create_box_plot(self, df, x_col, color_col, title, height):
        """Create a box plot"""
        try:
            fig = px.box(df, y=x_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
            return None

    def _create_violin_plot(self, df, x_col, color_col, title, height):
        """Create a violin plot"""
        try:
            fig = px.violin(df, y=x_col, color=color_col, title=title, box=True)
            return fig
        except Exception as e:
            st.error(f"Error creating violin plot: {str(e)}")
            return None

    def _create_heatmap(self, df, x_col, y_col, z_col, aggregation, title, height):
        """Create a heatmap"""
        try:
            if z_col:
                pivot_data = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc=aggregation)
                # Convert to native Python types
                pivot_data = pivot_data.astype(float)
                fig = px.imshow(pivot_data, title=title, aspect="auto")
            else:
                # Correlation heatmap for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr().astype(float)
                    fig = px.imshow(corr_matrix, title=title, aspect="auto", text_auto=True)
                else:
                    raise ValueError("Not enough numeric columns for correlation heatmap")
            return fig
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None

    # === ADVANCED CHART IMPLEMENTATIONS ===

    def _create_3d_scatter_plot(self, df, x_col, y_col, z_col, color_col, title, height):
        """Create a 3D scatter plot"""
        try:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating 3D scatter plot: {str(e)}")
            return None

    def _create_contour_plot(self, df, x_col, y_col, z_col, title, height):
        """Create a contour plot"""
        try:
            if z_col:
                # Use provided z column
                fig = go.Figure(data=go.Contour(
                    x=df[x_col],
                    y=df[y_col],
                    z=df[z_col],
                    colorscale='Viridis'
                ))
            else:
                # Create density contour
                fig = px.density_contour(df, x=x_col, y=y_col, title=title)

            fig.update_layout(title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating contour plot: {str(e)}")
            return None

    def _create_parallel_coordinates(self, df, selected_cols, color_col, title, height):
        """Create a parallel coordinates plot"""
        try:
            if not selected_cols:
                st.error("Please select columns for parallel coordinates")
                return None

            # Sample data if too large
            if len(df) > 1000:
                df_sample = df.sample(1000)
            else:
                df_sample = df

            fig = px.parallel_coordinates(
                df_sample,
                dimensions=selected_cols,
                color=color_col,
                title=title
            )
            return fig
        except Exception as e:
            st.error(f"Error creating parallel coordinates: {str(e)}")
            return None

    def _create_radar_chart(self, df, selected_cols, category_col, title, height):
        """Create a radar chart"""
        try:
            if not selected_cols:
                st.error("Please select columns for radar chart")
                return None

            fig = go.Figure()

            if category_col:
                # Multiple categories
                categories = df[category_col].unique()[:5]  # Limit to 5 categories
                for cat in categories:
                    cat_data = df[df[category_col] == cat][selected_cols].mean()

                    fig.add_trace(go.Scatterpolar(
                        r=cat_data.values,
                        theta=selected_cols,
                        fill='toself',
                        name=str(cat)
                    ))
            else:
                # Single radar for mean values
                mean_values = df[selected_cols].mean()

                fig.add_trace(go.Scatterpolar(
                    r=mean_values.values,
                    theta=selected_cols,
                    fill='toself',
                    name='Mean Values'
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, df[selected_cols].max().max()]
                    )),
                title=title
            )
            return fig
        except Exception as e:
            st.error(f"Error creating radar chart: {str(e)}")
            return None

    def _create_treemap(self, df, x_col, y_col, title, height):
        """Create a treemap"""
        try:
            if y_col:
                data = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.treemap(data, path=[x_col], values=y_col, title=title)
            else:
                data = df[x_col].value_counts().reset_index()
                data.columns = ['Category', 'Count']
                fig = px.treemap(data, path=['Category'], values='Count', title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating treemap: {str(e)}")
            return None

    def _create_sunburst(self, df, x_col, y_col, title, height):
        """Create a sunburst chart"""
        try:
            if y_col:
                data = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.sunburst(data, path=[x_col], values=y_col, title=title)
            else:
                data = df[x_col].value_counts().reset_index()
                data.columns = ['Category', 'Count']
                fig = px.sunburst(data, path=['Category'], values='Count', title=title)
            return fig
        except Exception as e:
            st.error(f"Error creating sunburst: {str(e)}")
            return None

    def _create_sankey_diagram(self, df, source_col, target_col, value_col, title, height):
        """Create a Sankey diagram"""
        try:
            # Prepare data for Sankey
            if value_col:
                sankey_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
            else:
                sankey_data = df.groupby([source_col, target_col]).size().reset_index(name='count')
                value_col = 'count'

            # Create node labels
            all_nodes = list(set(df[source_col].unique().tolist() + df[target_col].unique().tolist()))
            node_dict = {node: i for i, node in enumerate(all_nodes)}

            # Create links
            source_indices = [node_dict[source] for source in sankey_data[source_col]]
            target_indices = [node_dict[target] for target in sankey_data[target_col]]
            values = sankey_data[value_col].tolist()

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values
                )
            )])

            fig.update_layout(title_text=title, font_size=10)
            return fig
        except Exception as e:
            st.error(f"Error creating Sankey diagram: {str(e)}")
            return None

    def _create_word_cloud(self, df, x_col, title):
        """Create a word cloud"""
        if not HAS_WORDCLOUD:
            st.error("WordCloud library is not available. Install with: pip install wordcloud")
            return

        try:
            # Combine all text in the column
            text = ' '.join(df[x_col].astype(str).tolist())

            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

            # Display using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title)

            st.pyplot(fig)
            plt.close()  # Close the figure to free memory
        except Exception as e:
            st.error(f"Error creating word cloud: {str(e)}")

    def _create_funnel_chart(self, df, x_col, y_col, title, height):
        """Create a funnel chart"""
        try:
            if y_col:
                data = df.groupby(x_col)[y_col].sum().reset_index()
            else:
                data = df[x_col].value_counts().reset_index()
                data.columns = ['Stage', 'Count']

            fig = go.Figure(go.Funnel(
                y=data[x_col],
                x=data[y_col] if y_col else data['Count'],
                textinfo="value+percent initial"
            ))

            fig.update_layout(title=title, height=height)
            return fig
        except Exception as e:
            st.error(f"Error creating funnel chart: {str(e)}")
            return None

    def _create_waterfall_chart(self, df, x_col, y_col, title, height):
        """Create a waterfall chart"""
        try:
            if y_col:
                data = df.groupby(x_col)[y_col].sum().reset_index()
            else:
                data = df[x_col].value_counts().reset_index()
                data.columns = ['Category', 'Count']

            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(data),
                x=data[x_col],
                y=data[y_col] if y_col else data['Count'],
                textposition="outside",
                text=df[y_col] if y_col else data['Count'],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))

            fig.update_layout(title=title, height=height)
            return fig
        except Exception as e:
            st.error(f"Error creating waterfall chart: {str(e)}")
            return None

    def _create_candlestick_chart(self, df, x_col, y_col, title, height):
        """Create a candlestick chart"""
        try:
            if isinstance(y_col, list) and len(y_col) == 4:
                open_col, high_col, low_col, close_col = y_col
                
                fig = go.Figure(data=go.Candlestick(
                    x=df[x_col],
                    open=df[open_col],
                    high=df[high_col],
                    low=df[low_col],
                    close=df[close_col]
                ))
                
                fig.update_layout(title=title, height=height)
                return fig
            else:
                st.error("Candlestick chart requires 4 numeric columns (Open, High, Low, Close)")
                return None
        except Exception as e:
            st.error(f"Error creating candlestick chart: {str(e)}")
            return None

    def _create_gauge_chart(self, df, x_col, title, height):
        """Create a gauge chart"""
        try:
            value = df[x_col].mean()
            max_value = df[x_col].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, max_value*0.5], 'color': "lightgray"},
                        {'range': [max_value*0.5, max_value*0.75], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value}
                }
            ))
            
            fig.update_layout(height=height)
            return fig
        except Exception as e:
            st.error(f"Error creating gauge chart: {str(e)}")
            return None

    # === AUTO AUDIT AND ADVANCED FEATURES ===

    def _render_auto_audit(self, df, dataset_name):
        """Render auto audit and recommendations"""
        st.subheader("🔍 Auto Data Audit & Recommendations")

        if st.button("🤖 Run Automated Audit", type="primary", key="enhanced_auto_audit_btn"):
            with st.spinner("Analyzing dataset..."):
                recommendations = self._perform_data_audit(df)

                if recommendations:
                    st.markdown("### 📋 Audit Results & Recommendations")

                    for i, rec in enumerate(recommendations, 1):
                        priority_color = {
                            'High': '🔴',
                            'Medium': '🟡',
                            'Low': '🟢'
                        }

                        with st.expander(
                                f"{priority_color.get(rec['priority'], '🔵')} {rec['priority']} Priority: {rec['issue']}",
                                expanded=i <= 3):
                            st.write(f"**Description:** {rec['description']}")
                            st.write(f"**Recommendation:** {rec['recommendation']}")

                            if 'code' in rec:
                                st.markdown("**Suggested Code:**")
                                st.code(rec['code'])

                    # Log audit results
                    self.history.log_step(
                        "Auto Audit & Recommendation",
                        f"Generated {len(recommendations)} recommendations for {dataset_name}",
                        {
                            "total_recommendations": len(recommendations),
                            "high_priority": sum(1 for r in recommendations if r['priority'] == 'High'),
                            "medium_priority": sum(1 for r in recommendations if r['priority'] == 'Medium'),
                            "low_priority": sum(1 for r in recommendations if r['priority'] == 'Low')
                        },
                        "success"
                    )
                else:
                    st.success("🎉 No major issues found! Your dataset looks good.")

    def _perform_data_audit(self, df):
        """Perform automated data audit with safe handling"""
        recommendations = []

        try:
            # Check missing values
            missing_pct = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_pct[missing_pct > 20]

            if not high_missing.empty:
                for col in high_missing.index:
                    recommendations.append({
                        'priority': 'High',
                        'issue': f"High missing values in {col}",
                        'description': f"Column {col} has {missing_pct[col]:.1f}% missing values",
                        'recommendation': f"Consider imputation or removal of column {col}",
                        'code': f"# Impute missing values\ndf['{col}'].fillna(df['{col}'].median(), inplace=True)\n# OR remove column\ndf.drop('{col}', axis=1, inplace=True)"
                    })

            # Check duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                duplicate_pct = (duplicate_count / len(df)) * 100
                priority = 'High' if duplicate_pct > 10 else 'Medium'

                recommendations.append({
                    'priority': priority,
                    'issue': "Duplicate rows detected",
                    'description': f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                    'recommendation': "Remove duplicate rows to avoid bias",
                    'code': "df.drop_duplicates(inplace=True)"
                })

            # Check data types (safely)
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    # Test if column can be converted to numeric
                    test_numeric = pd.to_numeric(df[col].dropna().head(100), errors='coerce')
                    if not test_numeric.isna().all():
                        recommendations.append({
                            'priority': 'Medium',
                            'issue': f"Potential data type issue in {col}",
                            'description': f"Column {col} is object type but appears to contain numeric data",
                            'recommendation': f"Consider converting {col} to numeric type",
                            'code': f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
                        })
                except Exception:
                    # Skip columns that cause issues
                    pass

            # Check for outliers in numeric columns (safely)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:  # Avoid division by zero
                        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

                        if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                            recommendations.append({
                                'priority': 'Medium',
                                'issue': f"High outlier rate in {col}",
                                'description': f"Column {col} has {len(outliers)} outliers ({len(outliers) / len(df) * 100:.1f}%)",
                                'recommendation': f"Consider outlier treatment for {col}",
                                'code': f"# Remove outliers using IQR method\nQ1 = df['{col}'].quantile(0.25)\nQ3 = df['{col}'].quantile(0.75)\nIQR = Q3 - Q1\ndf = df[(df['{col}'] >= Q1 - 1.5*IQR) & (df['{col}'] <= Q3 + 1.5*IQR)]"
                            })
                except Exception:
                    # Skip problematic columns
                    pass

        except Exception as e:
            recommendations.append({
                'priority': 'High',
                'issue': "Data audit error",
                'description': f"Error during data audit: {str(e)}",
                'recommendation': "Check data format and types",
                'code': "# Check data info\ndf.info()\ndf.describe()"
            })

        return recommendations

    def _render_dashboard_templates(self, df, dataset_name):
        """Render dashboard templates"""
        st.subheader("🎨 Dashboard Templates")
        
        st.markdown("Select a pre-built dashboard template for common use cases:")
        
        # Template selection
        template_names = list(self.dashboard_templates.keys())
        selected_template = st.selectbox("Choose Dashboard Template", template_names)
        
        if selected_template:
            st.markdown(f"### {selected_template} Dashboard")
            st.markdown(f"This template includes: {', '.join(self.dashboard_templates[selected_template])}")
            
            # Show template preview
            if st.button("📊 Generate Template Dashboard", key=f"template_{selected_template}"):
                self._generate_template_dashboard(df, selected_template, dataset_name)

    def _generate_template_dashboard(self, df, template_name, dataset_name):
        """Generate a dashboard based on template"""
        st.markdown(f"### 📊 {template_name} Dashboard Preview")
        
        # Get chart types for this template
        chart_types = self.dashboard_templates[template_name]
        
        # Create a grid layout for the dashboard
        num_charts = len(chart_types)
        cols = 2 if num_charts > 1 else 1
        rows = (num_charts + cols - 1) // cols
        
        # Get available columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numeric_cols + categorical_cols
        
        if not all_cols:
            st.error("No suitable columns found for dashboard generation.")
            return
            
        # Generate charts for the template
        for i, chart_type in enumerate(chart_types):
            with st.container():
                st.markdown(f"#### {chart_type}")
                
                try:
                    # Select appropriate columns for each chart type
                    if chart_type in ['Bar Chart', 'Pie Chart', 'Donut Chart', 'Treemap', 'Sunburst']:
                        x_col = categorical_cols[0] if categorical_cols else all_cols[0]
                        y_col = numeric_cols[0] if numeric_cols else None
                        fig = self._generate_simple_chart(df, chart_type, x_col, y_col)
                    elif chart_type in ['Line Chart', 'Scatter Plot', 'Histogram']:
                        x_col = numeric_cols[0] if numeric_cols else all_cols[0]
                        y_col = numeric_cols[1] if len(numeric_cols) > 1 else None
                        fig = self._generate_simple_chart(df, chart_type, x_col, y_col)
                    elif chart_type == 'Heatmap':
                        fig = self._create_heatmap(df, numeric_cols[0] if numeric_cols else all_cols[0], 
                                                  numeric_cols[1] if len(numeric_cols) > 1 else None, 
                                                  numeric_cols[2] if len(numeric_cols) > 2 else None, 
                                                  "correlation", f"{template_name} - Correlation Heatmap", 400)
                    else:
                        # Default to simple bar chart for other types
                        x_col = categorical_cols[0] if categorical_cols else all_cols[0]
                        y_col = numeric_cols[0] if numeric_cols else None
                        fig = self._generate_simple_chart(df, "Bar Chart", x_col, y_col)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        # Add export options
                        self._add_chart_export_options(fig, f"{template_name}_{chart_type}")
                except Exception as e:
                    st.error(f"Error generating {chart_type}: {str(e)}")
            
            # Add spacing between charts
            if i < len(chart_types) - 1:
                st.markdown("---")

    def _generate_simple_chart(self, df, chart_type, x_col, y_col):
        """Generate a simple chart for dashboard templates"""
        try:
            if chart_type == "Bar Chart":
                data = df.groupby(x_col).size().reset_index(name='count')
                fig = px.bar(data, x=x_col, y='count', title=f"{chart_type} - {x_col}")
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type} - {x_col} vs {y_col}")
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type} - {x_col} vs {y_col}")
            elif chart_type == "Pie Chart":
                data = df[x_col].value_counts()
                fig = px.pie(values=data.values, names=data.index, title=f"{chart_type} - {x_col}")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, title=f"{chart_type} - {x_col}")
            elif chart_type == "Donut Chart":
                data = df[x_col].value_counts()
                fig = px.pie(values=data.values, names=data.index, title=f"{chart_type} - {x_col}", hole=0.4)
            else:
                # Default to bar chart
                data = df.groupby(x_col).size().reset_index(name='count')
                fig = px.bar(data, x=x_col, y='count', title=f"{chart_type} - {x_col}")
            
            return fig
        except Exception as e:
            st.error(f"Error generating simple chart: {str(e)}")
            return None

    def _render_custom_visualization_builder(self, df, dataset_name):
        """Render custom visualization builder"""
        st.subheader("🛠️ Custom Visualization Builder")
        
        st.markdown("Create your own custom visualizations with advanced configuration options.")
        
        # Visualization type selection
        viz_type = st.selectbox("Select Visualization Type", [
            "Custom Chart",
            "Multi-Chart Dashboard",
            "Interactive Dashboard",
            "Statistical Analysis Plot"
        ])
        
        if viz_type == "Custom Chart":
            self._render_custom_chart_builder(df, dataset_name)
        elif viz_type == "Multi-Chart Dashboard":
            self._render_multi_chart_dashboard(df, dataset_name)
        elif viz_type == "Interactive Dashboard":
            self._render_interactive_dashboard(df, dataset_name)
        elif viz_type == "Statistical Analysis Plot":
            self._render_statistical_analysis(df, dataset_name)

    def _render_custom_chart_builder(self, df, dataset_name):
        """Render custom chart builder interface"""
        st.markdown("### Custom Chart Builder")
        
        # Chart configuration
        col1, col2 = st.columns(2)
        
        with col1:
            chart_lib = st.selectbox("Chart Library", ["Plotly", "Matplotlib", "Seaborn"])
            chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter", "Area", "Histogram", "Box Plot"])
            
        with col2:
            x_col = st.selectbox("X-axis Column", df.columns.tolist())
            y_col = st.selectbox("Y-axis Column", [None] + df.columns.tolist())
            
        # Advanced options
        with st.expander("Advanced Options"):
            color_col = st.selectbox("Color Column", [None] + df.columns.tolist())
            size_col = st.selectbox("Size Column", [None] + df.columns.tolist())
            title = st.text_input("Chart Title", f"Custom {chart_type} Chart")
            height = st.slider("Chart Height", 300, 1000, 500)
            
        # Generate button
        if st.button("🎨 Generate Custom Chart", key="custom_chart_generate"):
            try:
                if chart_lib == "Plotly":
                    if chart_type == "Line":
                        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
                    elif chart_type == "Bar":
                        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
                    elif chart_type == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
                    elif chart_type == "Area":
                        fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_col, color=color_col, title=title)
                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
                        
                    if fig:
                        fig.update_layout(height=height)
                        st.plotly_chart(fig, use_container_width=True)
                        self._add_chart_export_options(fig, "custom_chart")
                        
                st.success("✅ Custom chart generated successfully!")
            except Exception as e:
                st.error(f"Error generating custom chart: {str(e)}")

    def _render_multi_chart_dashboard(self, df, dataset_name):
        """Render multi-chart dashboard builder"""
        st.markdown("### Multi-Chart Dashboard Builder")
        
        st.markdown("Create a dashboard with multiple charts arranged in a grid.")
        
        # Number of charts
        num_charts = st.slider("Number of Charts", 1, 6, 2)
        
        charts_config = []
        
        # Configure each chart
        for i in range(num_charts):
            st.markdown(f"#### Chart {i+1}")
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(f"Chart Type {i+1}", 
                                        ["Bar", "Line", "Scatter", "Pie", "Histogram"], 
                                        key=f"multi_chart_type_{i}")
                x_col = st.selectbox(f"X-axis Column {i+1}", 
                                   df.columns.tolist(), 
                                   key=f"multi_x_col_{i}")
                
            with col2:
                y_col = st.selectbox(f"Y-axis Column {i+1}", 
                                   [None] + df.columns.tolist(), 
                                   key=f"multi_y_col_{i}")
                title = st.text_input(f"Chart Title {i+1}", 
                                    f"{chart_type} Chart {i+1}", 
                                    key=f"multi_title_{i}")
            
            charts_config.append({
                'type': chart_type,
                'x_col': x_col,
                'y_col': y_col,
                'title': title
            })
        
        # Layout configuration
        st.markdown("#### Layout Configuration")
        layout_cols = st.slider("Columns in Dashboard", 1, 3, 2)
        
        # Generate dashboard
        if st.button("📊 Generate Multi-Chart Dashboard", key="multi_chart_dashboard_generate"):
            try:
                # Calculate rows needed
                rows = (num_charts + layout_cols - 1) // layout_cols
                
                # Create the dashboard
                for row in range(rows):
                    cols = st.columns(layout_cols)
                    for col_idx in range(layout_cols):
                        chart_idx = row * layout_cols + col_idx
                        if chart_idx < num_charts:
                            with cols[col_idx]:
                                config = charts_config[chart_idx]
                                try:
                                    if config['type'] == "Bar":
                                        data = df.groupby(config['x_col']).size().reset_index(name='count')
                                        fig = px.bar(data, x=config['x_col'], y='count', title=config['title'])
                                    elif config['type'] == "Line":
                                        fig = px.line(df, x=config['x_col'], y=config['y_col'], title=config['title'])
                                    elif config['type'] == "Scatter":
                                        fig = px.scatter(df, x=config['x_col'], y=config['y_col'], title=config['title'])
                                    elif config['type'] == "Pie":
                                        data = df[config['x_col']].value_counts()
                                        fig = px.pie(values=data.values, names=data.index, title=config['title'])
                                    elif config['type'] == "Histogram":
                                        fig = px.histogram(df, x=config['x_col'], title=config['title'])
                                    else:
                                        data = df.groupby(config['x_col']).size().reset_index(name='count')
                                        fig = px.bar(data, x=config['x_col'], y='count', title=config['title'])
                                    
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                        self._add_chart_export_options(fig, f"multi_chart_{chart_idx}")
                                except Exception as e:
                                    st.error(f"Error generating chart {chart_idx+1}: {str(e)}")
                
                st.success("✅ Multi-chart dashboard generated successfully!")
            except Exception as e:
                st.error(f"Error generating multi-chart dashboard: {str(e)}")

    def _render_interactive_dashboard(self, df, dataset_name):
        """Render interactive dashboard builder"""
        st.markdown("### Interactive Dashboard Builder")
        
        st.markdown("Create an interactive dashboard with filters and dynamic updates.")
        
        # Select key metrics to display
        st.markdown("#### Key Metrics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_metrics = st.multiselect("Select Key Metrics", numeric_cols, 
                                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        
        # Create metric cards
        if selected_metrics:
            cols = st.columns(len(selected_metrics))
            for i, metric in enumerate(selected_metrics):
                with cols[i]:
                    value = df[metric].mean()
                    st.metric(metric, f"{value:.2f}")
        
        # Select charts for dashboard
        st.markdown("#### Interactive Charts")
        chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"]
        selected_charts = st.multiselect("Select Charts", chart_types, default=chart_types[:2])
        
        # Add filters
        st.markdown("#### Filters")
        filter_cols = st.multiselect("Select Filter Columns", df.columns.tolist())
        
        # Generate interactive dashboard
        if st.button("📊 Generate Interactive Dashboard", key="interactive_dashboard_generate"):
            try:
                # Create filter widgets
                filter_values = {}
                if filter_cols:
                    filter_cols_container = st.columns(len(filter_cols))
                    for i, col in enumerate(filter_cols):
                        with filter_cols_container[i]:
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) <= 20:  # Only show dropdown if not too many values
                                filter_values[col] = st.multiselect(f"Filter by {col}", unique_vals)
                            else:
                                search_term = st.text_input(f"Search {col}")
                                if search_term:
                                    filter_values[col] = [v for v in unique_vals if search_term.lower() in str(v).lower()]
                
                # Apply filters
                filtered_df = df.copy()
                for col, values in filter_values.items():
                    if values:
                        filtered_df = filtered_df[filtered_df[col].isin(values)]
                
                st.markdown("### Interactive Dashboard Preview")
                
                # Update metrics based on filters
                if selected_metrics:
                    cols = st.columns(len(selected_metrics))
                    for i, metric in enumerate(selected_metrics):
                        with cols[i]:
                            if not filtered_df.empty:
                                value = filtered_df[metric].mean()
                                st.metric(metric, f"{value:.2f}")
                            else:
                                st.metric(metric, "N/A")
                
                # Update charts based on filters
                for chart_type in selected_charts:
                    st.markdown(f"#### {chart_type}")
                    try:
                        # Select appropriate columns for the chart
                        x_col = df.columns[0]  # Default to first column
                        y_col = numeric_cols[0] if numeric_cols else None
                        
                        if chart_type == "Bar Chart":
                            data = filtered_df.groupby(x_col).size().reset_index(name='count')
                            fig = px.bar(data, x=x_col, y='count', title=f"{chart_type} - Filtered Data")
                        elif chart_type == "Line Chart":
                            fig = px.line(filtered_df, x=x_col, y=y_col, title=f"{chart_type} - Filtered Data")
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(filtered_df, x=x_col, y=y_col, title=f"{chart_type} - Filtered Data")
                        elif chart_type == "Pie Chart":
                            data = filtered_df[x_col].value_counts()
                            fig = px.pie(values=data.values, names=data.index, title=f"{chart_type} - Filtered Data")
                        elif chart_type == "Histogram":
                            fig = px.histogram(filtered_df, x=x_col, title=f"{chart_type} - Filtered Data")
                        else:
                            data = filtered_df.groupby(x_col).size().reset_index(name='count')
                            fig = px.bar(data, x=x_col, y='count', title=f"{chart_type} - Filtered Data")
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            self._add_chart_export_options(fig, f"interactive_{chart_type}")
                    except Exception as e:
                        st.error(f"Error generating {chart_type}: {str(e)}")
                
                st.success("✅ Interactive dashboard generated successfully!")
            except Exception as e:
                st.error(f"Error generating interactive dashboard: {str(e)}")

    def _render_statistical_analysis(self, df, dataset_name):
        """Render statistical analysis plots"""
        st.markdown("### Statistical Analysis Plots")
        
        st.markdown("Generate advanced statistical visualizations for deeper insights.")
        
        # Analysis type selection
        analysis_type = st.selectbox("Analysis Type", [
            "Correlation Analysis",
            "Distribution Analysis",
            "Regression Analysis",
            "Time Series Analysis",
            "Cluster Analysis"
        ])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if analysis_type == "Correlation Analysis":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select Columns for Correlation", numeric_cols, default=numeric_cols[:4])
                if len(selected_cols) >= 2:
                    if st.button("📊 Generate Correlation Analysis"):
                        try:
                            corr_df = df[selected_cols].corr()
                            fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                                          title="Correlation Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                            self._add_chart_export_options(fig, "correlation_matrix")
                            
                            # Add correlation heatmap
                            fig2 = px.density_heatmap(df, x=selected_cols[0], y=selected_cols[1],
                                                    title=f"Density Heatmap: {selected_cols[0]} vs {selected_cols[1]}")
                            st.plotly_chart(fig2, use_container_width=True)
                            self._add_chart_export_options(fig2, "density_heatmap")
                        except Exception as e:
                            st.error(f"Error generating correlation analysis: {str(e)}")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
                
        elif analysis_type == "Distribution Analysis":
            selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
            if selected_col:
                if st.button("📊 Generate Distribution Analysis"):
                    try:
                        # Histogram
                        fig1 = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig1, use_container_width=True)
                        self._add_chart_export_options(fig1, "distribution_histogram")
                        
                        # Box plot
                        fig2 = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig2, use_container_width=True)
                        self._add_chart_export_options(fig2, "distribution_boxplot")
                        
                        # Violin plot
                        fig3 = px.violin(df, y=selected_col, title=f"Violin Plot of {selected_col}")
                        st.plotly_chart(fig3, use_container_width=True)
                        self._add_chart_export_options(fig3, "distribution_violin")
                    except Exception as e:
                        st.error(f"Error generating distribution analysis: {str(e)}")
                        
        elif analysis_type == "Regression Analysis":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select X Variable", numeric_cols)
                with col2:
                    y_col = st.selectbox("Select Y Variable", [col for col in numeric_cols if col != x_col])
                
                if x_col and y_col:
                    if st.button("📊 Generate Regression Analysis"):
                        try:
                            # Scatter plot with trendline
                            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                           title=f"Regression Analysis: {y_col} vs {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            self._add_chart_export_options(fig, "regression_scatter")
                            
                            # Residual plot
                            # This would require more complex calculations in a real implementation
                            st.info("Residual plot would be generated here in a full implementation.")
                        except Exception as e:
                            st.error(f"Error generating regression analysis: {str(e)}")
            else:
                st.warning("Need at least 2 numeric columns for regression analysis.")
                
        elif analysis_type == "Time Series Analysis":
            # This would require a datetime column
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                date_col = st.selectbox("Select Date Column", date_cols)
                value_col = st.selectbox("Select Value Column", numeric_cols)
                
                if date_col and value_col:
                    if st.button("📊 Generate Time Series Analysis"):
                        try:
                            # Line chart
                            fig = px.line(df, x=date_col, y=value_col,
                                        title=f"Time Series: {value_col} over time")
                            st.plotly_chart(fig, use_container_width=True)
                            self._add_chart_export_options(fig, "time_series_line")
                            
                            # Add moving average
                            df_sorted = df.sort_values(date_col)
                            df_sorted['MA_7'] = df_sorted[value_col].rolling(window=7).mean()
                            df_sorted['MA_30'] = df_sorted[value_col].rolling(window=30).mean()
                            
                            fig2 = px.line(df_sorted, x=date_col, y=[value_col, 'MA_7', 'MA_30'],
                                         title=f"Time Series with Moving Averages: {value_col}")
                            st.plotly_chart(fig2, use_container_width=True)
                            self._add_chart_export_options(fig2, "time_series_ma")
                        except Exception as e:
                            st.error(f"Error generating time series analysis: {str(e)}")
            else:
                st.warning("No datetime columns found. Please ensure your dataset has date/time data.")
                
        elif analysis_type == "Cluster Analysis":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select Columns for Clustering", numeric_cols, 
                                             default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
                if len(selected_cols) >= 2:
                    num_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    
                    if st.button("📊 Generate Cluster Analysis"):
                        try:
                            # This would require scikit-learn for actual clustering
                            st.info("In a full implementation, this would perform K-means clustering and visualize the results.")
                            
                            # For now, just show a scatter plot
                            if len(selected_cols) >= 2:
                                fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1],
                                               title=f"Cluster Analysis Preview: {selected_cols[0]} vs {selected_cols[1]}")
                                st.plotly_chart(fig, use_container_width=True)
                                self._add_chart_export_options(fig, "cluster_preview")
                        except Exception as e:
                            st.error(f"Error generating cluster analysis: {str(e)}")
            else:
                st.warning("Need at least 2 numeric columns for cluster analysis.")