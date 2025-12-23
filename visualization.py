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

warnings.filterwarnings('ignore')

# Try to import WordCloud - handle if not installed
try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

from pipeline_history import PipelineHistory


class Visualization:
    def __init__(self):
        self.history = PipelineHistory()

    def render_visualization_ui(self):
        """Render the visualization interface"""

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Visualization", dataset_names, key="viz_dataset_selector")

        if not selected_dataset:
            st.warning("No datasets available.")
            return

        df = st.session_state.datasets[selected_dataset].copy()
        st.session_state.current_dataset = selected_dataset

        st.markdown(f"**Dataset:** {selected_dataset} ({df.shape[0]} rows × {df.shape[1]} columns)")

        # Tabs for different visualization sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview",
            "📈 Chart Builder",
            "🔍 Auto Audit",
            "🎨 Advanced Charts"
        ])

        with tab1:
            self._render_data_overview(df)

        with tab2:
            self._render_chart_builder(df, selected_dataset)

        with tab3:
            self._render_auto_audit(df, selected_dataset)

        with tab4:
            self._render_advanced_charts(df, selected_dataset)

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

    def _render_data_overview(self, df):
        """Render data overview section"""
        st.subheader("📊 Data Overview")

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

        # Column information with safe handling
        st.markdown("### 📋 Column Information")

        column_info = []
        for col in df.columns:
            try:
                col_info = {
                    'Column': str(col),  # Convert to string
                    'Data Type': str(df[col].dtype),  # Convert to string
                    'Non-Null Count': int(df[col].count()),  # Convert to int
                    'Null Count': int(df[col].isnull().sum()),  # Convert to int
                    'Unique Values': self._safe_nunique(df[col]),  # Safe unique count
                    'Sample Values': self._safe_sample_values(df[col])  # Safe sample values
                }
                column_info.append(col_info)
            except Exception as e:
                # Fallback for problematic columns
                col_info = {
                    'Column': str(col),
                    'Data Type': str(df[col].dtype),
                    'Non-Null Count': "Error",
                    'Null Count': "Error",
                    'Unique Values': "Error",
                    'Sample Values': f"Error: {str(e)[:30]}..."
                }
                column_info.append(col_info)

        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)

        # Missing values chart
        if df.isnull().sum().sum() > 0:
            st.markdown("### 🕳️ Missing Values Analysis")

            missing_data = df.isnull().sum().sort_values(ascending=False)
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
                st.plotly_chart(fig, use_container_width=True, key="overview_missing_values")

        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("### 📈 Statistical Summary (Numeric Columns)")
            try:
                summary_df = df[numeric_cols].describe()
                # Convert all values to native Python types
                for col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(float)
                st.dataframe(summary_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating statistical summary: {str(e)}")

        # Value counts for categorical columns (safe handling)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            # Filter out columns that might contain unhashable types
            safe_categorical_cols = []
            for col in categorical_cols:
                try:
                    # Test if we can get value counts
                    test_counts = df[col].value_counts().head(1)
                    safe_categorical_cols.append(col)
                except Exception:
                    # Skip columns with unhashable types
                    continue

            if safe_categorical_cols:
                st.markdown("### 🏷️ Category Distributions")

                selected_cat_col = st.selectbox("Select Categorical Column", safe_categorical_cols,
                                                key="overview_cat_selector")
                if selected_cat_col:
                    try:
                        value_counts = df[selected_cat_col].value_counts().head(10)

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
                        st.plotly_chart(fig, use_container_width=True, key="overview_category_dist")
                    except Exception as e:
                        st.error(f"Error creating category distribution: {str(e)}")
            else:
                st.info("No suitable categorical columns found for distribution analysis.")

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

    def _render_chart_builder(self, df, dataset_name):
        """Render interactive chart builder with ALL chart types implemented"""
        st.subheader("📈 Interactive Chart Builder")

        # Get safe columns for visualization
        numeric_cols = self._get_safe_columns(df, [np.number])
        categorical_cols = self._get_safe_columns(df, ['object', 'category'])

        if not numeric_cols and not categorical_cols:
            st.error("No suitable columns found for visualization. The dataset may contain complex data types.")
            return

        # Chart type categorization
        categorical_charts = {
            'Basic': ['Bar Chart', 'Column Chart', 'Pie Chart', 'Donut Chart'],
            'Intermediate': ['Stacked Bar', 'Grouped Bar', 'Lollipop Chart'],
            'Advanced': ['Treemap', 'Sunburst', 'Sankey Diagram']
        }

        numerical_charts = {
            'Basic': ['Histogram', 'Line Chart', 'Scatter Plot', 'Box Plot'],
            'Intermediate': ['Area Chart', 'Violin Plot', 'Heatmap', 'Bubble Chart'],
            'Advanced': ['3D Scatter', 'Contour Plot', 'Parallel Coordinates', 'Radar Chart']
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

            data_type = st.selectbox("Data Type", available_data_types, key="chart_data_type")

            if data_type == "Categorical":
                available_charts = categorical_charts
            elif data_type == "Numerical":
                available_charts = numerical_charts
            else:  # Mixed
                available_charts = {**categorical_charts, **numerical_charts}

        with col2:
            complexity = st.selectbox("Complexity Level", ["Basic", "Intermediate", "Advanced"], key="chart_complexity")
            chart_options = available_charts.get(complexity, [])

            if not chart_options:
                st.warning(f"No {complexity} charts available for {data_type} data type.")
                return

            selected_chart = st.selectbox("Chart Type", chart_options, key="chart_type_selector")

        # Column selection based on chart type
        self._render_chart_configuration(df, selected_chart, dataset_name, numeric_cols, categorical_cols)

    def _render_chart_configuration(self, df, chart_type, dataset_name, numeric_cols, categorical_cols):
        """Render chart configuration interface"""

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
                x_column = st.selectbox("Select Category Column", categorical_cols, key=f"{chart_type}_x_col")
            with col2:
                if chart_type in ['Bar Chart', 'Column Chart']:
                    y_column = st.selectbox("Select Value Column (optional)", [None] + numeric_cols,
                                            key=f"{chart_type}_y_col")
                    aggregation = st.selectbox("Aggregation", ["count", "sum", "mean", "median"],
                                               key=f"{chart_type}_agg")
                else:
                    y_column = None
                    aggregation = "count"

        elif chart_type in ['Stacked Bar', 'Grouped Bar']:
            if not categorical_cols or not numeric_cols:
                st.error("Both categorical and numeric columns are required for this chart type.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis Column", categorical_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"{chart_type}_y_col")
            with col3:
                color_column = st.selectbox("Stack/Group by", categorical_cols, key=f"{chart_type}_color_col")
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"], key=f"{chart_type}_agg")

        elif chart_type == 'Lollipop Chart':
            if not categorical_cols or not numeric_cols:
                st.error("Both categorical and numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Category Column", categorical_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column", numeric_cols, key=f"{chart_type}_y_col")
            aggregation = None

        elif chart_type in ['Histogram', 'Box Plot', 'Violin Plot']:
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            x_column = st.selectbox("Select Numeric Column", numeric_cols, key=f"{chart_type}_x_col")
            y_column = None
            aggregation = None

        elif chart_type in ['Line Chart', 'Scatter Plot', 'Bubble Chart']:
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"{chart_type}_y_col")
            aggregation = None

        elif chart_type == 'Area Chart':
            if not numeric_cols:
                st.error("Numeric columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column", numeric_cols, key=f"{chart_type}_y_col")
            aggregation = None

        elif chart_type == 'Heatmap':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for heatmap.")
                return

            x_column = st.selectbox("X-axis Column", all_cols, key=f"{chart_type}_x_col")
            y_column = st.selectbox("Y-axis Column", all_cols, key=f"{chart_type}_y_col")
            z_column = st.selectbox("Value Column", numeric_cols, key=f"{chart_type}_z_col")
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"], key=f"{chart_type}_agg")

        elif chart_type == '3D Scatter':
            if len(numeric_cols) < 3:
                st.error("At least 3 numeric columns are required for 3D scatter plot.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis", numeric_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis", numeric_cols, key=f"{chart_type}_y_col")
            with col3:
                z_column = st.selectbox("Z-axis", numeric_cols, key=f"{chart_type}_z_col")
            aggregation = None

        elif chart_type == 'Contour Plot':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for contour plot.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                x_column = st.selectbox("X-axis", numeric_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis", numeric_cols, key=f"{chart_type}_y_col")
            with col3:
                z_column = st.selectbox("Z-axis (optional)", [None] + numeric_cols, key=f"{chart_type}_z_col")
            aggregation = None

        elif chart_type == 'Parallel Coordinates':
            if len(numeric_cols) < 2:
                st.error("At least 2 numeric columns are required for parallel coordinates.")
                return

            selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, default=numeric_cols[:4],
                                           key=f"{chart_type}_cols")
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
                                           key=f"{chart_type}_cols")
            if not selected_cols:
                st.warning("Please select at least 2 columns.")
                return
            category_col = st.selectbox("Category Column (optional)", [None] + categorical_cols,
                                        key=f"{chart_type}_cat_col")
            x_column = selected_cols
            y_column = category_col
            aggregation = None

        elif chart_type in ['Treemap', 'Sunburst']:
            if not categorical_cols:
                st.error("Categorical columns are required for this chart type.")
                return

            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Category Column", categorical_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Value Column (optional)", [None] + numeric_cols, key=f"{chart_type}_y_col")
            aggregation = "sum" if y_column else "count"

        elif chart_type == 'Sankey Diagram':
            if len(categorical_cols) < 2:
                st.error("At least 2 categorical columns are required for Sankey diagram.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                source_col = st.selectbox("Source Column", categorical_cols, key=f"{chart_type}_source_col")
            with col2:
                target_col = st.selectbox("Target Column", categorical_cols, key=f"{chart_type}_target_col")
            with col3:
                value_col = st.selectbox("Value Column (optional)", [None] + numeric_cols,
                                         key=f"{chart_type}_value_col")
            x_column = source_col
            y_column = target_col
            z_column = value_col
            aggregation = "sum" if value_col else "count"

        elif chart_type == 'Word Cloud' and HAS_WORDCLOUD:
            if not categorical_cols:
                st.error("Text columns are required for word cloud.")
                return

            x_column = st.selectbox("Text Column", categorical_cols, key=f"{chart_type}_x_col")
            y_column = None
            aggregation = None

        else:
            # Default configuration for other chart types
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis Column", all_cols, key=f"{chart_type}_x_col")
            with col2:
                y_column = st.selectbox("Y-axis Column (optional)", [None] + all_cols, key=f"{chart_type}_y_col")
            aggregation = None

        # Additional chart options
        with st.expander("🎨 Chart Options"):
            if chart_type not in ['Parallel Coordinates', 'Radar Chart', 'Sankey Diagram']:
                color_column = st.selectbox("Color by Column (optional)", [None] + all_cols,
                                            key=f"{chart_type}_color_opt")
            else:
                color_column = None

            size_column = st.selectbox("Size by Column (optional)", [None] + numeric_cols,
                                       key=f"{chart_type}_size_opt") if chart_type in ['Scatter Plot',
                                                                                       'Bubble Chart'] else None

            # Chart styling options
            col1, col2, col3 = st.columns(3)
            with col1:
                chart_title = st.text_input("Chart Title", value=f"{chart_type} - {x_column}",
                                            key=f"{chart_type}_title")
            with col2:
                chart_height = st.number_input("Chart Height (px)", value=500, min_value=300, max_value=1000,
                                               key=f"{chart_type}_height")
            with col3:
                show_legend = st.checkbox("Show Legend", value=True, key=f"{chart_type}_legend")

        # Generate chart button
        if st.button("🎨 Generate Chart", type="primary", key=f"{chart_type}_generate"):
            if x_column:
                self._generate_chart(
                    df, chart_type, x_column, y_column,
                    color_column, size_column, aggregation,
                    chart_title, chart_height, show_legend,
                    dataset_name, locals()
                )
            else:
                st.error("Please select at least one column for the chart.")

    def _generate_chart(self, df, chart_type, x_col, y_col, color_col, size_col,
                        aggregation, title, height, show_legend, dataset_name, params={}):
        """Generate the specified chart with ALL implementations"""

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

            else:
                st.warning(f"Chart type '{chart_type}' is not yet implemented.")
                return

            # Display the chart
            if fig:
                fig.update_layout(
                    title=title,
                    height=height,
                    showlegend=show_legend
                )
                # Use unique key for each chart
                chart_key = f"{chart_type.lower().replace(' ', '_')}_{x_col}_{dataset_name}_{id(fig)}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)

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

    # === AUTO AUDIT AND ADVANCED FEATURES ===

    def _render_auto_audit(self, df, dataset_name):
        """Render auto audit and recommendations"""
        st.subheader("🔍 Auto Data Audit & Recommendations")

        if st.button("🤖 Run Automated Audit", type="primary", key="auto_audit_btn"):
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

    def _render_advanced_charts(self, df, dataset_name):
        """Render advanced chart options"""
        st.subheader("🎨 Advanced Charts & Analysis")

        # Chart type selection for advanced charts
        advanced_charts = [
            "Correlation Heatmap",
            "Distribution Matrix",
            "Parallel Coordinates",
            "3D Scatter Plot",
            "Statistical Summary"
        ]

        selected_advanced = st.selectbox("Select Advanced Chart", advanced_charts, key="advanced_chart_selector")

        if selected_advanced == "Correlation Heatmap":
            self._render_correlation_heatmap(df)

        elif selected_advanced == "Distribution Matrix":
            self._render_distribution_matrix(df)

        elif selected_advanced == "Parallel Coordinates":
            self._render_parallel_coordinates_advanced(df)

        elif selected_advanced == "3D Scatter Plot":
            self._render_3d_scatter_advanced(df)

        elif selected_advanced == "Statistical Summary":
            self._render_statistical_summary(df)

    def _render_correlation_heatmap(self, df):
        """Render correlation heatmap for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return

        selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, default=numeric_cols,
                                       key="advanced_corr_cols")

        if len(selected_cols) >= 2:
            try:
                corr_matrix = df[selected_cols].corr().astype(float)  # Convert to float

                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    text_auto=True
                )

                st.plotly_chart(fig, use_container_width=True, key="advanced_correlation_matrix")
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {str(e)}")

    def _render_distribution_matrix(self, df):
        """Render distribution matrix (simplified version)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for distribution matrix.")
            return

        selected_cols = st.multiselect("Select Numeric Columns (max 4)", numeric_cols, default=numeric_cols[:3],
                                       key="advanced_dist_cols")

        if len(selected_cols) >= 2 and len(selected_cols) <= 4:
            try:
                # Create subplots for distribution matrix
                n_cols = len(selected_cols)
                fig = make_subplots(
                    rows=n_cols, cols=n_cols,
                    subplot_titles=[f"{col1} vs {col2}" for col1 in selected_cols for col2 in selected_cols]
                )

                for i, col1 in enumerate(selected_cols):
                    for j, col2 in enumerate(selected_cols):
                        if i == j:
                            # Diagonal: histogram
                            fig.add_histogram(x=df[col1], row=i + 1, col=j + 1, name=f"{col1} hist")
                        else:
                            # Off-diagonal: scatter plot
                            fig.add_scatter(x=df[col2], y=df[col1], mode='markers',
                                            row=i + 1, col=j + 1, name=f"{col1} vs {col2}")

                fig.update_layout(title="Distribution Matrix", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="advanced_distribution_matrix")

            except Exception as e:
                st.error(f"Error creating distribution matrix: {str(e)}")

    def _render_parallel_coordinates_advanced(self, df):
        """Render parallel coordinates plot"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for parallel coordinates.")
            return

        selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, default=numeric_cols[:5],
                                       key="advanced_parallel_cols")
        color_col = st.selectbox("Color by Column (optional)", [None] + categorical_cols + numeric_cols,
                                 key="advanced_parallel_color")

        if len(selected_cols) >= 2:
            try:
                # Convert data for plotly compatibility
                plot_df = self._convert_for_plotly(df[selected_cols + ([color_col] if color_col else [])])

                # Sample data if too large
                if len(plot_df) > 1000:
                    plot_df = plot_df.sample(1000)

                fig = px.parallel_coordinates(
                    plot_df,
                    dimensions=selected_cols,
                    color=color_col,
                    title="Parallel Coordinates Plot"
                )

                st.plotly_chart(fig, use_container_width=True, key="advanced_parallel_coordinates")
            except Exception as e:
                st.error(f"Error creating parallel coordinates: {str(e)}")

    def _render_3d_scatter_advanced(self, df):
        """Render 3D scatter plot"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns for 3D scatter plot.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="advanced_3d_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="advanced_3d_y")
        with col3:
            z_col = st.selectbox("Z-axis", numeric_cols, key="advanced_3d_z")

        color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist(), key="advanced_3d_color")

        if x_col and y_col and z_col:
            try:
                plot_df = self._convert_for_plotly(df)

                fig = px.scatter_3d(
                    plot_df, x=x_col, y=y_col, z=z_col,
                    color=color_col,
                    title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}"
                )

                st.plotly_chart(fig, use_container_width=True, key="advanced_3d_scatter")
            except Exception as e:
                st.error(f"Error creating 3D scatter: {str(e)}")

    def _render_statistical_summary(self, df):
        """Render comprehensive statistical summary"""
        st.markdown("### 📊 Statistical Summary")

        # Overall dataset info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            try:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            except Exception:
                st.metric("Memory Usage", "N/A")
        with col3:
            try:
                completeness = ((df.count().sum()) / df.size) * 100
                st.metric("Completeness", f"{completeness:.1f}%")
            except Exception:
                st.metric("Completeness", "N/A")

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            st.markdown("**Numeric Columns Statistics:**")
            try:
                summary_df = df[numeric_cols].describe()
                # Convert to float to avoid dtype issues
                for col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(float)
                st.dataframe(summary_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating numeric summary: {str(e)}")

        # Categorical columns summary (safe handling)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            st.markdown("**Categorical Columns Statistics:**")

            cat_stats = []
            for col in categorical_cols:
                try:
                    unique_count = self._safe_nunique(df[col])
                    most_frequent = "N/A"
                    missing_count = int(df[col].isnull().sum())

                    try:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            most_frequent = str(mode_val.iloc[0])
                    except Exception:
                        most_frequent = "Unable to calculate"

                    cat_stats.append({
                        'Column': str(col),
                        'Unique Values': unique_count,
                        'Most Frequent': most_frequent,
                        'Missing Count': missing_count,
                        'Missing %': f"{(missing_count / len(df)) * 100:.1f}%"
                    })
                except Exception as e:
                    cat_stats.append({
                        'Column': str(col),
                        'Unique Values': "Error",
                        'Most Frequent': "Error",
                        'Missing Count': "Error",
                        'Missing %': "Error"
                    })

            if cat_stats:
                cat_df = pd.DataFrame(cat_stats)
                st.dataframe(cat_df, use_container_width=True)