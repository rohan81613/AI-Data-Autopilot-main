"""
Modern Gradio UI for AI Data Platform - Complete Implementation
A completely new implementation with all features from the 11-tab Streamlit interface
Fixed all dropdown issues and enhanced with modern UI patterns
"""
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
import json
import base64
import tempfile
from typing import Dict, List, Any, Optional, Tuple

# Import project modules
from data_connections import DataConnections
from data_preprocessing import DataPreprocessing
from visualization import Visualization
from modeling import ModelTraining
from validation import DataValidation
from sample_data import SampleDatasets
from enhanced_visualization import EnhancedVisualization
from enhanced_preprocessing import EnhancedDataPreprocessing
from enhanced_modeling import EnhancedModelTraining
from ai.ai_assistant import AIAssistant
from smart_autopilot import smart_pilot
from preprocessing_advanced import AdvancedPreprocessing
try:
    from modeling_advanced import AdvancedModeling
    ADVANCED_MODELING_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Advanced modeling not available: {e}")
    ADVANCED_MODELING_AVAILABLE = False
    AdvancedModeling = None
from report_generation_advanced import AdvancedReportGenerator

# Import new utilities
from utils.profiling import profile_dataframe
from utils.export import export_data
from utils.lineage import get_lineage_tracker
from utils.monitoring import get_system_monitor, get_performance_tracker
from utils.professional_report import generate_professional_report

warnings.filterwarnings('ignore')

# Suppress Streamlit warnings from imported modules (we use Gradio, not Streamlit)
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Initialize helper classes
data_conn = DataConnections()
data_prep = DataPreprocessing()
viz = Visualization()
model_trainer = ModelTraining()
validator = DataValidation()
sample_data = SampleDatasets()
enhanced_viz = EnhancedVisualization()
enhanced_prep = EnhancedDataPreprocessing()
enhanced_model = EnhancedModelTraining()
ai_assistant = AIAssistant()
adv_prep = AdvancedPreprocessing()
adv_model = AdvancedModeling() if ADVANCED_MODELING_AVAILABLE else None
report_gen = AdvancedReportGenerator()

# Global session state
class SessionState:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.charts = []
        self.history = []
        self.current_dataset = None
        self.ai_chat_history = []
        self.ai_insights = {}  # Initialize as empty dict
        self.preferences = {
            'theme': 'light',
            'auto_refresh': True
        }
    
    def add_dataset(self, name: str, df: pd.DataFrame):
        self.datasets[name] = df
        self.current_dataset = name
        self.log_action("Data Loading", f"Loaded dataset: {name}", {"rows": len(df), "cols": len(df.columns)})
    
    def log_action(self, operation: str, description: str, details: Optional[Dict] = None):
        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "description": description,
            "details": details or {}
        })

# Global state instance
state = SessionState()

# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

def get_dataset_choices():
    """Get updated choices for all dataset dropdowns"""
    choices = list(state.datasets.keys())
    return gr.update(choices=choices, value=choices[-1] if choices else None)

def get_empty_dataset_choices():
    """Get empty choices for dataset dropdowns"""
    return gr.update(choices=[], value=None)

def get_all_dataset_updates():
    """Get updates for ALL dataset dropdowns - returns 15 updates"""
    choices = list(state.datasets.keys())
    default_value = choices[-1] if choices else None
    update = gr.update(choices=choices, value=default_value)
    # Return 15 updates for all dataset dropdowns
    return tuple([update] * 15)

def get_all_model_updates():
    """Get updates for ALL model dropdowns - returns 2 updates"""
    choices = list(state.models.keys())
    default_value = choices[-1] if choices else None
    update = gr.update(choices=choices, value=default_value)
    return tuple([update] * 2)

def get_columns(dataset_name: str):
    """Get columns for selected dataset - returns 4 dropdown updates"""
    if not dataset_name or dataset_name not in state.datasets:
        return (
            gr.update(choices=[]),  # X-axis
            gr.update(choices=[]),  # Y-axis
            gr.update(choices=[]),  # Color
            gr.update(choices=[])   # Size
        )
    
    df = state.datasets[dataset_name]
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Return updates for all 4 column dropdowns
    return (
        gr.update(choices=all_cols, value=all_cols[0] if all_cols else None),  # X-axis
        gr.update(choices=all_cols, value=all_cols[1] if len(all_cols) > 1 else all_cols[0] if all_cols else None),  # Y-axis
        gr.update(choices=all_cols, value=None),  # Color
        gr.update(choices=all_cols, value=None)   # Size
    )

def update_target_cols(dataset_name: str):
    """Update target columns when dataset changes"""
    if dataset_name and dataset_name in state.datasets:
        cols = state.datasets[dataset_name].columns.tolist()
        return gr.update(choices=cols, value=cols[0] if cols else None)
    return gr.update(choices=[], value=None)

# ============================================================================
# SESSION MANAGEMENT FUNCTIONS
# ============================================================================

def save_session() -> Tuple[str, Optional[str]]:
    """Save current session to file"""
    try:
        session_data = {
            'datasets': {name: df.to_dict() for name, df in state.datasets.items()},
            'history': state.history,
            'models': list(state.models.keys()),
            'preferences': state.preferences
        }
        
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str, indent=2)
        
        state.log_action("Session Management", f"Saved session as {filename}")
        return f"✅ Session saved as {filename}", filename
    except Exception as e:
        return f"❌ Error saving session: {str(e)}", None

def load_session(file):
    """Load session from file"""
    if file is None:
        return "Please select a session file"
    
    try:
        with open(file.name, 'r') as f:
            session_data = json.load(f)
        
        # Load datasets
        state.datasets = {}
        for name, data in session_data.get('datasets', {}).items():
            state.datasets[name] = pd.DataFrame(data)
        
        # Load history
        state.history = session_data.get('history', [])
        
        # Load preferences
        state.preferences = session_data.get('preferences', state.preferences)
        
        state.log_action("Session Management", "Loaded session from file")
        return "✅ Session loaded successfully!"
    except Exception as e:
        return f"❌ Error loading session: {str(e)}"

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_sample_data(dataset_name: str):
    """Load sample dataset"""
    try:
        df = sample_data.load_sample(dataset_name.lower())
        
        if df is not None:
            name = f"sample_{dataset_name.lower()}"
            state.add_dataset(name, df)
            
            info = f"✅ Loaded {dataset_name} dataset\n"
            info += f"Dataset: {name}\n"
            info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            info += f"Columns: {', '.join(df.columns.tolist()[:5])}"
            if len(df.columns) > 5:
                info += f"... (+{len(df.columns)-5} more)"
            info += f"\n\n**📌 Important:** The dataset is now available in all dropdowns throughout the interface."
            
            # Return info, data preview, and list of dataset names
            return info, df.head(10), list(state.datasets.keys())
        
        return "❌ Failed to load dataset", pd.DataFrame(), []
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame(), []

def upload_file(file):
    """Upload and load file"""
    if file is None:
        return "Please select a file", pd.DataFrame(), []
    
    try:
        file_path = file.name
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return f"❌ Unsupported file type: {ext}", pd.DataFrame(), []
        
        name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state.add_dataset(name, df)
        
        info = f"✅ Uploaded successfully\n"
        info += f"Dataset: {name}\n"
        info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        info += f"**📌 Important:** After uploading, manually refresh the page or click on any dropdown to see the new dataset in the list."
        
        # Return info, data preview, and list of dataset names
        return info, df.head(10), list(state.datasets.keys())
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame(), []

def get_dataset_info() -> str:
    """Get info about all datasets"""
    if not state.datasets:
        return "No datasets loaded"
    
    info = "📊 **Current Datasets:**\n\n"
    for name, df in state.datasets.items():
        info += f"**{name}**\n"
        info += f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        info += f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
    return info

# ============================================================================
# DATA PROFILING FUNCTIONS
# ============================================================================

def profile_dataset(dataset_name: str) -> Tuple[str, Optional[go.Figure]]:
    """Generate data quality profile"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset", None
    
    try:
        df = state.datasets[dataset_name]
        profile = profile_dataframe(df, dataset_name)
        
        # Create summary
        summary = f"# Data Quality Report: {dataset_name}\n\n"
        summary += f"## Quality Score: {profile['quality_score']['overall']}/100 (Grade: {profile['quality_score']['grade']})\n\n"
        
        summary += "### Overview\n"
        summary += f"- Rows: {profile['overview']['n_rows']:,}\n"
        summary += f"- Columns: {profile['overview']['n_columns']}\n"
        summary += f"- Missing: {profile['overview']['missing_percentage']:.2f}%\n"
        summary += f"- Duplicates: {profile['duplicates']['n_duplicates']}\n\n"
        
        if profile['alerts']:
            summary += "### ⚠️ Alerts\n"
            for alert in profile['alerts'][:5]:
                summary += f"- **{alert['type']}**: {alert['message']}\n"
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=profile['quality_score']['overall'],
            title={'text': "Quality Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 60], 'color': "lightgray"},
                       {'range': [60, 80], 'color': "gray"},
                       {'range': [80, 100], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        
        state.log_action("Data Profiling", f"Profiled {dataset_name}", profile['quality_score'])
        
        return summary, fig
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ============================================================================

# Professional color schemes
COLOR_SCHEMES = {
    'Professional Blue': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'Vibrant': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22'],
    'Pastel': ['#a8dadc', '#457b9d', '#1d3557', '#f1faee', '#e63946', '#ffb4a2', '#b5838d', '#6d6875'],
    'Corporate': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600', '#7a5195', '#ef5675', '#ffa600'],
    'Earth Tones': ['#8d5524', '#c68642', '#e0ac69', '#f1c27d', '#ffdbac', '#a0522d', '#cd853f', '#deb887'],
    'Ocean': ['#03045e', '#023e8a', '#0077b6', '#0096c7', '#00b4d8', '#48cae4', '#90e0ef', '#ade8f4'],
    'Sunset': ['#ff6b6b', '#ee5a6f', '#c44569', '#a8336a', '#6c5b7b', '#4a5859', '#355c7d', '#2a4d69']
}

def recommend_chart_types(df: pd.DataFrame, x_col: str, y_col: str = None) -> List[Dict]:
    """Recommend best chart types based on data characteristics"""
    recommendations = []
    
    x_dtype = df[x_col].dtype
    x_unique = df[x_col].nunique()
    is_x_numeric = pd.api.types.is_numeric_dtype(x_dtype)
    is_x_categorical = x_unique < 20 or x_dtype == 'object'
    
    y_dtype = df[y_col].dtype if y_col and y_col in df.columns else None
    is_y_numeric = pd.api.types.is_numeric_dtype(y_dtype) if y_dtype else False
    
    # Categorical X
    if is_x_categorical:
        recommendations.append({
            'type': 'Bar Chart',
            'score': 95,
            'reason': f'Perfect for categorical data with {x_unique} categories',
            'best_for': 'Comparing categories'
        })
        recommendations.append({
            'type': 'Pie Chart',
            'score': 85,
            'reason': 'Shows proportions of categories',
            'best_for': 'Part-to-whole relationships'
        })
        if x_unique <= 10:
            recommendations.append({
                'type': 'Donut Chart',
                'score': 80,
                'reason': 'Modern alternative to pie chart',
                'best_for': 'Proportions with central metric'
            })
    
    # Numeric X and Y
    if is_x_numeric and is_y_numeric:
        recommendations.append({
            'type': 'Scatter Plot',
            'score': 95,
            'reason': 'Shows relationship between two numeric variables',
            'best_for': 'Correlation analysis'
        })
        recommendations.append({
            'type': 'Line Chart',
            'score': 90,
            'reason': 'Shows trends over continuous data',
            'best_for': 'Time series or continuous trends'
        })
    
    # Single numeric column
    if is_x_numeric and not y_col:
        recommendations.append({
            'type': 'Histogram',
            'score': 95,
            'reason': 'Shows distribution of numeric data',
            'best_for': 'Understanding data distribution'
        })
        recommendations.append({
            'type': 'Box Plot',
            'score': 90,
            'reason': 'Shows statistical summary and outliers',
            'best_for': 'Identifying outliers and quartiles'
        })
        recommendations.append({
            'type': 'Violin Plot',
            'score': 85,
            'reason': 'Combines box plot with distribution',
            'best_for': 'Detailed distribution analysis'
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:5]

def create_advanced_visualization(
    dataset_name: str, 
    chart_type: str, 
    x_col: str, 
    y_col: str, 
    color_col: str, 
    size_col: str,
    limit_data: int,
    color_scheme: str,
    show_values: bool,
    chart_title: str
) -> Tuple[Optional[go.Figure], str, str]:
    """Create advanced visualization with Power BI-like features"""
    if not dataset_name or dataset_name not in state.datasets:
        return None, "Please select a dataset", ""
    
    if not x_col:
        return None, "Please select X column", ""
    
    try:
        df = state.datasets[dataset_name].copy()
        
        # Apply data limiting - IMPROVED
        if limit_data > 0 and limit_data < len(df):
            if chart_type in ["Bar Chart", "Pie Chart", "Donut Chart", "Treemap", "Funnel Chart"]:
                # For categorical charts, get top N by frequency or value
                if y_col and y_col in df.columns:
                    # If Y column exists, group by X and sum/mean Y, then get top N
                    grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(limit_data)
                    df_plot = df[df[x_col].isin(grouped.index)].copy()
                    limit_msg = f"Showing top {limit_data} {x_col} by {y_col}"
                else:
                    # No Y column, get top N by frequency
                    top_categories = df[x_col].value_counts().head(limit_data).index
                    df_plot = df[df[x_col].isin(top_categories)].copy()
                    limit_msg = f"Showing top {limit_data} categories"
            else:
                # For other charts (scatter, line, etc.), take first N rows
                df_plot = df.head(limit_data)
                limit_msg = f"Showing first {limit_data} rows"
        else:
            df_plot = df
            limit_msg = f"Showing all {len(df)} rows"
        
        # Get color scheme
        colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['Professional Blue'])
        
        # Custom title
        title = chart_title if chart_title else f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col else "")
        
        # Create chart based on type
        fig = None
        powerbi_guide = ""
        
        if chart_type == "Bar Chart":
            if y_col and y_col in df_plot.columns:
                # Aggregate data by X column to show clean bars
                df_agg = df_plot.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(df_agg, x=x_col, y=y_col, color=color_col if color_col and color_col in df_agg.columns else None, 
                           title=title, color_discrete_sequence=colors)
                powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Bar Chart' or 'Column Chart' visual
2. Drag '{x_col}' to Axis
3. Drag '{y_col}' to Values
4. {'Drag "' + color_col + '" to Legend' if color_col else 'Optional: Add field to Legend for colors'}
5. Format → Data colors → Choose color scheme
6. Format → Data labels → On (to show values)
7. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
            else:
                value_counts = df_plot[x_col].value_counts().sort_values(ascending=False).reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.bar(value_counts, x=x_col, y='count', title=title,
                           color_discrete_sequence=colors)
                powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Bar Chart' visual
2. Drag '{x_col}' to Axis
3. Drag '{x_col}' to Values (will auto-count)
4. Format → Data colors → Choose colors
5. Format → Data labels → On
6. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
        
        elif chart_type == "Line Chart":
            fig = px.line(df_plot, x=x_col, y=y_col, color=color_col, 
                        title=title, color_discrete_sequence=colors,
                        markers=True)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Line Chart' visual
2. Drag '{x_col}' to Axis
3. Drag '{y_col}' to Values
4. {'Drag "' + color_col + '" to Legend' if color_col else ''}
5. Format → Data colors → Choose scheme
6. Format → Markers → On (for data points)
"""
        
        elif chart_type == "Scatter Plot":
            size_param = size_col if size_col and size_col in df_plot.columns else None
            fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_col, size=size_param,
                           title=title, color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Scatter Chart' visual
2. Drag '{x_col}' to X-Axis
3. Drag '{y_col}' to Y-Axis
4. {'Drag "' + color_col + '" to Legend' if color_col else ''}
5. {'Drag "' + size_param + '" to Size' if size_param else ''}
6. Format → Data colors → Choose colors
"""
        
        elif chart_type == "Pie Chart":
            if y_col and y_col in df_plot.columns:
                # Aggregate by X column
                df_agg = df_plot.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
                fig = px.pie(df_agg, values=y_col, names=x_col, title=title,
                           color_discrete_sequence=colors)
            else:
                value_counts = df_plot[x_col].value_counts().sort_values(ascending=False).reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.pie(value_counts, values='count', names=x_col, title=title,
                           color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Pie Chart' visual
2. Drag '{x_col}' to Legend
3. Drag '{y_col if y_col else x_col}' to Values
4. Format → Data colors → Choose colors
5. Format → Detail labels → On (show percentages)
6. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
        
        elif chart_type == "Donut Chart":
            if y_col and y_col in df_plot.columns:
                # Aggregate by X column
                df_agg = df_plot.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
                fig = px.pie(df_agg, values=y_col, names=x_col, title=title,
                           color_discrete_sequence=colors, hole=0.4)
            else:
                value_counts = df_plot[x_col].value_counts().sort_values(ascending=False).reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.pie(value_counts, values='count', names=x_col, title=title,
                           color_discrete_sequence=colors, hole=0.4)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Donut Chart' visual
2. Drag '{x_col}' to Legend
3. Drag '{y_col if y_col else x_col}' to Values
4. Format → Data colors → Choose colors
5. Format → Detail labels → On
6. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
        
        elif chart_type == "Histogram":
            fig = px.histogram(df_plot, x=x_col, color=color_col, title=title,
                             color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Column Chart' visual
2. Drag '{x_col}' to Axis
3. Drag '{x_col}' to Values (Count)
4. Format → X-Axis → Type → Continuous (for bins)
5. Format → Data colors → Choose colors
"""
        
        elif chart_type == "Box Plot":
            fig = px.box(df_plot, y=x_col, color=color_col, title=title,
                       color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Import 'Box and Whisker' custom visual
2. Drag '{x_col}' to Values
3. {'Drag "' + color_col + '" to Category' if color_col else ''}
4. Format → Data colors → Choose colors
"""
        
        elif chart_type == "Violin Plot":
            fig = px.violin(df_plot, y=x_col, color=color_col, title=title,
                          color_discrete_sequence=colors, box=True)
            powerbi_guide = f"""
**Power BI Steps:**
1. Import 'Violin Plot' custom visual from AppSource
2. Drag '{x_col}' to Values
3. {'Drag "' + color_col + '" to Category' if color_col else ''}
4. Format → Colors → Choose scheme
"""
        
        elif chart_type == "Heatmap":
            numeric_df = df_plot.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return None, "Need at least 2 numeric columns for heatmap", ""
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", title=title,
                          color_continuous_scale=colors[:3] if len(colors) >= 3 else 'RdBu_r')
            powerbi_guide = """
**Power BI Steps:**
1. Import 'Heatmap' custom visual
2. Drag numeric columns to Values
3. Format → Color scale → Choose gradient
4. Format → Data labels → On (show correlations)
"""
        
        elif chart_type == "Treemap":
            if y_col and y_col in df_plot.columns:
                # Aggregate by X column
                df_agg = df_plot.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
                fig = px.treemap(df_agg, path=[x_col], values=y_col, color=color_col if color_col and color_col in df_agg.columns else None,
                               title=title, color_discrete_sequence=colors)
            else:
                value_counts = df_plot[x_col].value_counts().sort_values(ascending=False).reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.treemap(value_counts, path=[x_col], values='count', title=title,
                               color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Treemap' visual
2. Drag '{x_col}' to Group
3. Drag '{y_col if y_col else x_col}' to Values
4. Format → Data colors → Choose colors
5. Format → Data labels → On
6. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
        
        elif chart_type == "Area Chart":
            fig = px.area(df_plot, x=x_col, y=y_col, color=color_col, title=title,
                        color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Area Chart' visual
2. Drag '{x_col}' to Axis
3. Drag '{y_col}' to Values
4. {'Drag "' + color_col + '" to Legend' if color_col else ''}
5. Format → Data colors → Choose colors
"""
        
        elif chart_type == "Funnel Chart":
            if y_col and y_col in df_plot.columns:
                # Aggregate by X column
                df_agg = df_plot.groupby(x_col)[y_col].sum().sort_values(ascending=False).reset_index()
                fig = px.funnel(df_agg, x=y_col, y=x_col, title=title,
                              color_discrete_sequence=colors)
            else:
                value_counts = df_plot[x_col].value_counts().sort_values(ascending=False).reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.funnel(value_counts, x='count', y=x_col, title=title,
                              color_discrete_sequence=colors)
            powerbi_guide = f"""
**Power BI Steps:**
1. Select 'Funnel' visual
2. Drag '{x_col}' to Group
3. Drag '{y_col if y_col else 'count'}' to Values
4. Format → Data colors → Choose colors
5. {'Format → Filters → Top N → ' + str(limit_data) if limit_data > 0 else ''}
"""
        
        else:
            fig = px.bar(df_plot, x=x_col, y=y_col, title=title,
                       color_discrete_sequence=colors)
            powerbi_guide = "Select appropriate visual in Power BI and drag fields accordingly."
        
        # Apply professional styling
        if fig:
            fig.update_layout(
                height=600,
                template='plotly_white',
                font=dict(family="Arial, sans-serif", size=12),
                title_font=dict(size=18, family="Arial Black"),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='closest'
            )
            
            # Show values on chart if requested
            if show_values and chart_type in ["Bar Chart", "Line Chart"]:
                fig.update_traces(texttemplate='%{y}', textposition='outside')
        
        state.log_action("Visualization", f"Created {chart_type}", {
            "x": x_col, "y": y_col, "limit": limit_data, "scheme": color_scheme
        })
        
        info = f"✅ {chart_type} created successfully\n"
        info += f"📊 {limit_msg}\n"
        info += f"🎨 Color scheme: {color_scheme}\n"
        info += f"📈 Chart optimized for professional presentation"
        
        return fig, info, powerbi_guide
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""

def generate_auto_dashboard(dataset_name: str) -> Tuple[List[go.Figure], str]:
    """Automatically generate a dashboard with best charts for the dataset"""
    if not dataset_name or dataset_name not in state.datasets:
        return [], "Please select a dataset"
    
    try:
        df = state.datasets[dataset_name]
        charts = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Chart 1: Distribution of first numeric column
        if len(numeric_cols) > 0:
            fig1 = px.histogram(df, x=numeric_cols[0], 
                              title=f"Distribution of {numeric_cols[0]}",
                              color_discrete_sequence=COLOR_SCHEMES['Professional Blue'])
            fig1.update_layout(height=400, template='plotly_white')
            charts.append(fig1)
        
        # Chart 2: Top categories if categorical data exists
        if len(categorical_cols) > 0:
            value_counts = df[categorical_cols[0]].value_counts().head(10).reset_index()
            value_counts.columns = [categorical_cols[0], 'count']
            fig2 = px.bar(value_counts, x=categorical_cols[0], y='count',
                        title=f"Top 10 {categorical_cols[0]}",
                        color_discrete_sequence=COLOR_SCHEMES['Vibrant'])
            fig2.update_layout(height=400, template='plotly_white')
            charts.append(fig2)
        
        # Chart 3: Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig3 = px.imshow(corr, text_auto='.2f', aspect="auto",
                           title="Correlation Matrix",
                           color_continuous_scale='RdBu_r')
            fig3.update_layout(height=400, template='plotly_white')
            charts.append(fig3)
        
        # Chart 4: Scatter plot if 2+ numeric columns
        if len(numeric_cols) >= 2:
            color_col = categorical_cols[0] if categorical_cols else None
            fig4 = px.scatter(df.head(1000), x=numeric_cols[0], y=numeric_cols[1],
                            color=color_col, title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                            color_discrete_sequence=COLOR_SCHEMES['Corporate'])
            fig4.update_layout(height=400, template='plotly_white')
            charts.append(fig4)
        
        summary = f"✅ Generated {len(charts)} charts automatically\n"
        summary += f"📊 Dataset: {dataset_name}\n"
        summary += f"📈 Numeric columns: {len(numeric_cols)}\n"
        summary += f"📋 Categorical columns: {len(categorical_cols)}"
        
        return charts, summary
    except Exception as e:
        return [], f"❌ Error: {str(e)}"

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def get_data_overview(dataset_name: str) -> str:
    """Get comprehensive data overview"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    df = state.datasets[dataset_name]
    
    overview = f"# Data Overview: {dataset_name}\n\n"
    overview += f"## Shape\n- Rows: {df.shape[0]:,}\n- Columns: {df.shape[1]}\n\n"
    
    overview += "## Data Types\n"
    for dtype, count in df.dtypes.value_counts().items():
        overview += f"- {dtype}: {count} columns\n"
    
    overview += "\n## Missing Values\n"
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            overview += f"- **{col}**: {count} ({pct:.1f}%)\n"
    else:
        overview += "✅ No missing values\n"
    
    overview += f"\n## Duplicates\n"
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        overview += f"⚠️ {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)\n"
    else:
        overview += "✅ No duplicates\n"
    
    return overview

def handle_missing_values(dataset_name: str, strategy: str) -> str:
    """Handle missing values"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    try:
        df = state.datasets[dataset_name].copy()
        original_shape = df.shape
        
        if strategy == "Drop rows":
            df = df.dropna()
        elif strategy == "Fill with mean":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "Fill with median":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "Fill with mode":
            for col in df.columns:
                try:
                    mode_val = df[col].mode()
                    if not mode_val.empty and len(mode_val) > 0:
                        mode_scalar = mode_val.iloc[0]
                        # Convert to Python native type to avoid pandas type issues
                        if pd.notna(mode_scalar):
                            python_value = mode_scalar.item() if hasattr(mode_scalar, 'item') else str(mode_scalar)
                            df[col] = df[col].fillna(python_value)
                except:
                    # If mode calculation fails, skip this column
                    pass
        
        new_name = f"{dataset_name}_cleaned"
        state.add_dataset(new_name, df)
        
        result = f"✅ Missing values handled using '{strategy}'\n"
        result += f"Original: {original_shape[0]} rows\n"
        result += f"New: {df.shape[0]} rows\n"
        result += f"Saved as: {new_name}"
        
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

def remove_duplicates(dataset_name: str) -> str:
    """Remove duplicate rows"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    try:
        df = state.datasets[dataset_name].copy()
        original_count = len(df)
        df = df.drop_duplicates()
        removed = original_count - len(df)
        
        new_name = f"{dataset_name}_deduped"
        state.add_dataset(new_name, df)
        
        result = f"✅ Removed {removed} duplicate rows\n"
        result += f"New dataset: {new_name}\n"
        result += f"Rows: {len(df)}"
        
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# MODELING FUNCTIONS
# ============================================================================

def train_model(dataset_name: str, target_col: str, algorithm: str, test_size: int, cross_validation: bool) -> str:
    """Train machine learning model"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    if not target_col:
        return "Please select target column"
    
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
        import xgboost as xgb
        import lightgbm as lgb
        
        df = state.datasets[dataset_name]
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        # Select model
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        if algorithm == "Random Forest":
            model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
        elif algorithm == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss') if is_classification else xgb.XGBRegressor(random_state=42)
        elif algorithm == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1) if is_classification else lgb.LGBMRegressor(random_state=42, verbose=-1)
        elif algorithm == "Logistic/Linear Regression":
            model = LogisticRegression(random_state=42, max_iter=1000) if is_classification else LinearRegression()
        elif algorithm == "SVM":
            model = SVC(random_state=42) if is_classification else SVR()
        else:
            # Default model
            model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        mse = None
        if is_classification:
            score = accuracy_score(y_test, y_pred)
            metric = "Accuracy"
        else:
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            metric = "R²"
        
        # Cross-validation if requested
        cv_score = None
        if cross_validation:
            try:
                cv_scores = cross_val_score(model, X, y, cv=5)
                cv_score = cv_scores.mean()
            except:
                pass
        
        # Save model
        model_name = f"{algorithm}_{dataset_name}_{datetime.now().strftime('%H%M%S')}"
        state.models[model_name] = {
            'model': model,
            'features': X.columns.tolist(),
            'target': target_col,
            'score': score
        }
        
        result = f"✅ Model trained successfully!\n\n"
        result += f"**Algorithm**: {algorithm}\n"
        result += f"**Target**: {target_col}\n"
        result += f"**{metric}**: {score:.4f}\n"
        if not is_classification and mse is not None:
            result += f"**MSE**: {mse:.4f}\n"
        if cv_score is not None:
            result += f"**CV Score**: {cv_score:.4f}\n"
        result += f"**Model saved as**: {model_name}\n"
        result += f"**Features**: {len(X.columns)}"
        
        state.log_action("Model Training", f"Trained {algorithm}", {"score": score, "cv_score": cv_score})
        
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_dataset(dataset_name: str, export_format: str) -> Tuple[str, Optional[str]]:
    """Export dataset in various formats"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset", None
    
    try:
        df = state.datasets[dataset_name]
        filename = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = None
        if export_format == "CSV":
            filepath = f"{filename}.csv"
            export_data(df, filepath, format='csv')
        elif export_format == "Excel":
            filepath = f"{filename}.xlsx"
            export_data(df, filepath, format='excel', with_formatting=True)
        elif export_format == "JSON":
            filepath = f"{filename}.json"
            export_data(df, filepath, format='json')
        elif export_format == "Power BI":
            filepath = f"{filename}_powerbi.csv"
            export_data(df, filepath, format='powerbi')
        elif export_format == "Parquet":
            filepath = f"{filename}.parquet"
            export_data(df, filepath, format='parquet')
        
        state.log_action("Export", f"Exported {dataset_name} as {export_format}")
        
        if filepath:
            return f"✅ Exported as {export_format}\nFile: {filepath}", filepath
        else:
            return f"❌ Error exporting file", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ============================================================================
# AI ASSISTANT FUNCTIONS
# ============================================================================

def ask_ai(dataset_name: str, question: str) -> str:
    """Ask AI assistant a question about the dataset"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    if not question:
        return "Please enter a question"
    
    try:
        df = state.datasets[dataset_name]
        response = ai_assistant.llm.analyze_dataframe(df, question)
        
        # Add to chat history
        state.ai_chat_history.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'query': question,
            'response': response
        })
        
        state.log_action("AI Query", f"Asked: {question[:50]}...", {"dataset": dataset_name})
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_ai_chat_history() -> str:
    """Get AI chat history"""
    if not state.ai_chat_history:
        return "No AI conversations yet"
    
    history_text = "# AI Chat History\n\n"
    for item in reversed(state.ai_chat_history[-10:]):  # Last 10 items
        history_text += f"**You ({item['timestamp']}):** {item['query']}\n\n"
        history_text += f"**AI Assistant:** {item['response']}\n\n"
        history_text += "---\n\n"
    
    return history_text

def generate_ai_insights(dataset_name: str) -> str:
    """Generate AI insights for dataset"""
    if not dataset_name or dataset_name not in state.datasets:
        return "Please select a dataset"
    
    try:
        df = state.datasets[dataset_name]
        insights = ai_assistant.llm.generate_insights(df)
        state.ai_insights = insights
        
        state.log_action("AI Insights", f"Generated insights for {dataset_name}")
        
        # Format insights
        result = "# 🤖 AI-Generated Insights\n\n"
        result += f"## 📋 Executive Summary\n{insights.get('summary', 'No summary available')}\n\n"
        result += "## 📈 Patterns & Trends\n"
        patterns = insights.get('patterns', [])
        for i, pattern in enumerate(patterns[:10], 1):
            if pattern.strip():
                result += f"{i}. {pattern}\n"
        result += "\n## ⚠️ Anomalies & Data Quality Issues\n"
        anomalies = insights.get('anomalies', [])
        for i, anomaly in enumerate(anomalies[:10], 1):
            if anomaly.strip():
                result += f"{i}. {anomaly}\n"
        result += "\n## 💡 Recommendations\n"
        recommendations = insights.get('recommendations', [])
        for i, rec in enumerate(recommendations[:10], 1):
            if rec.strip():
                result += f"{i}. {rec}\n"
        
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# HISTORY & MONITORING FUNCTIONS
# ============================================================================

def get_history() -> str:
    """Get pipeline history"""
    if not state.history:
        return "No operations performed yet"
    
    history_text = "# Pipeline History\n\n"
    for item in reversed(state.history[-20:]):  # Last 20 items
        history_text += f"**{item['timestamp']}** - {item['operation']}\n"
        history_text += f"_{item['description']}_\n\n"
    
    return history_text

def clear_history() -> str:
    """Clear pipeline history"""
    state.history = []
    return "History cleared"

def get_system_status() -> str:
    """Get system monitoring info"""
    try:
        monitor = get_system_monitor()
        metrics = monitor.get_current_metrics()
        
        status = "# System Status\n\n"
        status += f"## CPU\n- Usage: {metrics['cpu']['percent']:.1f}%\n"
        status += f"- Cores: {metrics['cpu']['count']}\n\n"
        status += f"## Memory\n- Usage: {metrics['memory']['percent']:.1f}%\n"
        status += f"- Available: {metrics['memory']['available_gb']:.2f} GB\n\n"
        status += f"## Disk\n- Usage: {metrics['disk']['percent']:.1f}%\n"
        status += f"- Free: {metrics['disk']['free_gb']:.2f} GB\n"
        
        return status
    except:
        return "System monitoring not available"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the main Gradio interface with modern UI patterns"""
    
    # Dictionary to store all dropdown references
    dropdown_refs = {}
    
    # Helper function to refresh all dropdowns (defined early so it can be used anywhere)
    def refresh_all_dropdowns_fn():
        """Refresh all dataset dropdowns with current data"""
        choices = list(state.datasets.keys())
        default_val = choices[-1] if choices else None
        update = gr.update(choices=choices, value=default_val)
        # Return updates for all 21 dataset dropdowns (including autopilot, automl, autofix, report, and ppt)
        return tuple([update] * 21)
    
    def refresh_model_dropdowns_fn():
        """Refresh all model dropdowns with current trained models"""
        choices = list(state.models.keys())
        default_val = choices[-1] if choices else None
        update = gr.update(choices=choices, value=default_val)
        # Return updates for both model dropdowns (performance and feature importance)
        return update, update
    
    with gr.Blocks(
        title="AI Data Platform 2025 - Modern UI",
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 100%;
        }
        .tab-nav button {
            font-size: 16px;
            font-weight: 600;
            padding: 12px 24px;
            border-radius: 8px 8px 0 0;
            margin-right: 4px;
        }
        .panel {
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            background: white;
        }
        .metric-card {
            text-align: center;
            padding: 16px;
            border-radius: 8px;
            background: #f8f9fa;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
        }
        """
    ) as demo:
        
        # Create a State variable to trigger updates
        dataset_list_state = gr.State(value=[])
        
        # Header
        with gr.Column(elem_classes=["header"]):
            gr.Markdown("""
            # 🚀 AI Data Platform 2025
            ### Enterprise-grade data analysis, visualization, and machine learning
            """)
            with gr.Row():
                status_text = gr.Markdown(f"**Datasets Loaded:** {len(state.datasets)} | **Models Trained:** {len(state.models)}")
                refresh_all_btn = gr.Button("🔄 Refresh All Dropdowns", size="sm", variant="secondary")
        
        # Main Tabs
        with gr.Tabs():
            
            # ========== HOME DASHBOARD ==========
            with gr.Tab("🏠 Home Dashboard"):
                gr.Markdown("## 🎯 Quick Actions")
                with gr.Row():
                    with gr.Column():
                        quick_sample = gr.Radio(
                            choices=["Iris", "Titanic", "Housing", "Wine Quality"],
                            label="Load Sample Dataset",
                            value="Iris"
                        )
                        quick_load_btn = gr.Button("⚡ Quick Load", variant="primary", size="sm")
                        quick_status = gr.Textbox(label="Status", lines=2, max_lines=3)
                        
                        def quick_load_sample(dataset_name):
                            try:
                                df = sample_data.load_sample(dataset_name.lower())
                                if df is not None:
                                    name = f"sample_{dataset_name.lower()}"
                                    state.add_dataset(name, df)
                                    return f"✅ Loaded {dataset_name}: {df.shape[0]}×{df.shape[1]}"
                                return "❌ Failed to load"
                            except Exception as e:
                                return f"❌ Error: {str(e)}"
                        
                        quick_load_btn.click(
                            fn=quick_load_sample,
                            inputs=quick_sample,
                            outputs=quick_status
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Platform Statistics")
                        
                        def get_platform_stats():
                            stats = f"### {len(state.datasets)}\n**Datasets Loaded**\n\n"
                            stats += f"### {len(state.models)}\n**Models Trained**\n\n"
                            stats += f"### {len(state.history)}\n**Operations Performed**\n\n"
                            
                            if state.datasets:
                                total_rows = sum(df.shape[0] for df in state.datasets.values())
                                total_cols = sum(df.shape[1] for df in state.datasets.values())
                                stats += f"### {total_rows:,}\n**Total Rows**\n\n"
                                stats += f"### {total_cols}\n**Total Columns**"
                            
                            return stats
                        
                        platform_stats = gr.Markdown(value=get_platform_stats())
                        
                        gr.Markdown("### 📁 Current Datasets")
                        dataset_info_display = gr.Markdown(value=get_dataset_info())
                        refresh_dashboard = gr.Button("🔄 Refresh Dashboard")
                        
                        def refresh_dashboard_data():
                            return get_platform_stats(), get_dataset_info()
                        
                        refresh_dashboard.click(
                            fn=refresh_dashboard_data,
                            outputs=[platform_stats, dataset_info_display]
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📚 Recent Activity")
                        history_display = gr.Markdown(value=get_history())
                        with gr.Row():
                            refresh_history = gr.Button("🔄 Refresh")
                            clear_history_btn = gr.Button("🗑️ Clear History")
                        
                        refresh_history.click(
                            fn=get_history,
                            outputs=history_display
                        )
                        clear_history_btn.click(
                            fn=clear_history,
                            outputs=history_display
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 💻 System Status")
                        system_status_display = gr.Markdown(value=get_system_status())
                        refresh_system = gr.Button("🔄 Refresh System")
                        refresh_system.click(
                            fn=get_system_status,
                            outputs=system_status_display
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Data Quality Overview")
                        
                        def get_quality_overview():
                            if not state.datasets:
                                return "No datasets loaded"
                            
                            overview = "| Dataset | Rows | Cols | Missing % | Duplicates |\n"
                            overview += "|---------|------|------|-----------|------------|\n"
                            
                            for name, df in list(state.datasets.items())[:5]:
                                rows = df.shape[0]
                                cols = df.shape[1]
                                missing_pct = (df.isnull().sum().sum() / (rows * cols) * 100) if rows * cols > 0 else 0
                                duplicates = df.duplicated().sum()
                                overview += f"| {name[:20]} | {rows:,} | {cols} | {missing_pct:.1f}% | {duplicates} |\n"
                            
                            if len(state.datasets) > 5:
                                overview += f"\n*...and {len(state.datasets)-5} more datasets*"
                            
                            return overview
                        
                        quality_overview = gr.Markdown(value=get_quality_overview())
                        refresh_quality = gr.Button("🔄 Refresh Quality")
                        refresh_quality.click(
                            fn=get_quality_overview,
                            outputs=quality_overview
                        )
            
            # ========== SMART AUTO-PILOT (FOR BEGINNERS) ==========
            with gr.Tab("� Samart Auto-Pilot"):
                gr.Markdown("""
                # 🚀 Smart Auto-Pilot - For Beginners
                ### ✨ Clean, Prepare, and Analyze Your Data with ONE CLICK!
                
                **Perfect for beginners** - No data science knowledge required!
                """)
                
                with gr.Tabs():
                    with gr.Tab("🧹 One-Click Clean"):
                        gr.Markdown("### 🧹 Automatic Data Cleaning")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dataset_select_autopilot = gr.Dropdown(choices=list(state.datasets.keys()), label="Select Dataset", interactive=True)
                                auto_clean_btn = gr.Button("🚀 AUTO-CLEAN EVERYTHING", variant="primary", size="lg")
                                gr.Markdown("#### ✅ Removes empty data\n✅ Fixes types\n✅ Fills missing (ML)\n✅ Removes duplicates\n✅ Handles outliers\n✅ Cleans text")
                            with gr.Column(scale=2):
                                auto_clean_status = gr.Markdown(value="Click button to start...")
                                auto_clean_preview = gr.Dataframe(label="Preview")
                        
                        def auto_clean_data(dataset_name):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset", pd.DataFrame()
                            try:
                                df = state.datasets[dataset_name]
                                df_clean, report, _ = smart_pilot.auto_clean_everything(df, dataset_name)
                                new_name = f"{dataset_name}_cleaned"
                                state.add_dataset(new_name, df_clean)
                                return report + f"\n\n✅ **Saved as: '{new_name}'**\n\n📌 **The cleaned dataset is now available in ALL dropdowns throughout the app!**\n\n🎯 **You can now use this cleaned data in any feature: Visualization, Modeling, Export, etc.**", df_clean.head(10)
                            except Exception as e:
                                return f"❌ Error: {str(e)}", pd.DataFrame()
                        
                        # Store button reference for later wiring
                        auto_clean_click = auto_clean_btn.click(
                            fn=auto_clean_data, 
                            inputs=dataset_select_autopilot, 
                            outputs=[auto_clean_status, auto_clean_preview]
                        )
                    
                    with gr.Tab("🤖 One-Click ML"):
                        gr.Markdown("### 🤖 Automatic ML Training")
                        with gr.Row():
                            with gr.Column(scale=1):
                                dataset_select_automl = gr.Dropdown(choices=list(state.datasets.keys()), label="Select Dataset", interactive=True)
                                target_col_automl = gr.Dropdown(choices=[], label="What to predict?", interactive=True)
                                auto_ml_btn = gr.Button("🚀 AUTO-TRAIN MODEL", variant="primary", size="lg")
                                
                                def update_target_automl(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        cols = state.datasets[dataset_name].columns.tolist()
                                        return gr.update(choices=cols, value=cols[-1] if cols else None)
                                    return gr.update(choices=[], value=None)
                                
                                dataset_select_automl.change(fn=update_target_automl, inputs=dataset_select_automl, outputs=target_col_automl)
                            
                            with gr.Column(scale=2):
                                auto_ml_status = gr.Markdown(value="Select dataset and target...")
                        
                        def auto_train_ml(dataset_name, target_col):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset"
                            if not target_col:
                                return "❌ Select target"
                            try:
                                df = state.datasets[dataset_name]
                                X, y, features, prep = smart_pilot.auto_prepare_for_ml(df, target_col)
                                model, score, train = smart_pilot.auto_train_best_model(X, y)
                                model_name = f"AutoML_{dataset_name}"
                                state.models[model_name] = {'model': model, 'features': features, 'target': target_col, 'score': score}
                                return prep + "\n\n" + train + f"\n\n✅ **Model: '{model_name}'**"
                            except Exception as e:
                                return f"❌ Error: {str(e)}"
                        
                        auto_ml_click = auto_ml_btn.click(
                            fn=auto_train_ml, 
                            inputs=[dataset_select_automl, target_col_automl], 
                            outputs=auto_ml_status
                        )
                    
                    with gr.Tab("❓ Help"):
                        gr.Markdown("""
                        # ❓ How to Use
                        
                        ## 🚀 Quick Start (3 Steps)
                        1. **Load Data** - Go to "Data Management" tab
                        2. **Clean Data** - Use "One-Click Clean" (creates "yourdata_cleaned")
                        3. **Train Model** - Use "One-Click ML" with the cleaned data
                        
                        ## 💡 Tips
                        - Always clean data first for best results
                        - Cleaned datasets end with "_cleaned" and appear in ALL dropdowns
                        - You can use cleaned data in ANY feature: Visualization, Modeling, Export, etc.
                        - No need to download and re-upload - it's instantly available!
                        
                        **That's it!** 🎉
                        """)
            
            # ========== DATA MANAGEMENT ==========
            with gr.Tab("📂 Data Management"):
                with gr.Tabs():
                    with gr.Tab("📤 Upload Data"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📤 Upload Dataset")
                                file_upload = gr.File(
                                    label="Select File (CSV, Excel, JSON, Parquet)",
                                    file_types=[".csv", ".xlsx", ".xls", ".json", ".parquet"]
                                )
                                upload_btn = gr.Button("📤 Upload Dataset", variant="primary")
                                upload_status = gr.Textbox(label="Upload Status", lines=4, max_lines=6)
                                upload_preview = gr.Dataframe(label="Data Preview")
                    
                    with gr.Tab("🎯 Sample Data"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🎯 Load Sample Datasets")
                                sample_choice = gr.Radio(
                                    choices=["Iris", "Titanic", "Housing", "Wine Quality"],
                                    label="Select Sample Dataset",
                                    value="Iris"
                                )
                                load_sample_btn = gr.Button("📥 Load Sample Dataset", variant="primary")
                                sample_status = gr.Textbox(label="Loading Status", lines=4, max_lines=6)
                                sample_preview = gr.Dataframe(label="Sample Data Preview")
                    
                    with gr.Tab("💾 Session Management"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 💾 Save & Load Sessions")
                                
                                # Save Session
                                save_session_btn = gr.Button("💾 Save Current Session", variant="primary")
                                save_session_status = gr.Textbox(label="Save Status", lines=2)
                                save_session_file = gr.File(label="Download Session File")
                                
                                save_session_btn.click(
                                    fn=save_session,
                                    outputs=[save_session_status, save_session_file]
                                )
                                
                                # Load Session
                                load_session_upload = gr.File(label="Upload Session File", file_types=[".json"])
                                load_session_btn = gr.Button("📂 Load Session", variant="secondary")
                                load_session_status = gr.Textbox(label="Load Status", lines=2)
                                
                                load_session_btn.click(
                                    fn=load_session,
                                    inputs=load_session_upload,
                                    outputs=load_session_status
                                )
            
            # ========== DATA EXPLORATION ==========
            with gr.Tab("🔍 Data Exploration"):
                with gr.Tabs():
                    with gr.Tab("📊 Data Profiling"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📊 Data Quality Profiling")
                                gr.Markdown("**📌 Tip:** Load data first from the 'Data Management' tab or use Quick Actions on the Home Dashboard.")
                                dataset_select_profile = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True,
                                    info="Load a dataset first to see options here"
                                )
                                profile_btn = gr.Button("🔍 Generate Quality Report", variant="primary")
                                
                                gr.Markdown("### About Data Profiling")
                                gr.Markdown("""
                                Automated data quality assessment:
                                - Quality scoring (0-100)
                                - Missing value analysis
                                - Data type detection
                                - Outlier detection
                                - Correlation analysis
                                """)
                            
                            with gr.Column(scale=2):
                                profile_report = gr.Markdown(label="Quality Report")
                                profile_chart = gr.Plot(label="Quality Score Visualization")
                                
                                profile_btn.click(
                                    fn=profile_dataset,
                                    inputs=dataset_select_profile,
                                    outputs=[profile_report, profile_chart]
                                )
                    
                    with gr.Tab("📋 Data Overview"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📋 Comprehensive Data Overview")
                                dataset_select_overview = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True
                                )
                                get_overview_btn = gr.Button("📋 Generate Overview", variant="primary")
                                overview_output = gr.Markdown(label="Dataset Overview", height=500)
                                
                                get_overview_btn.click(
                                    fn=get_data_overview,
                                    inputs=dataset_select_overview,
                                    outputs=overview_output
                                )
            
            # ========== ADVANCED VISUALIZATION STUDIO ==========
            with gr.Tab("🎨 Visualization Studio"):
                with gr.Tabs():
                    # Tab 1: Advanced Chart Builder
                    with gr.Tab("🎨 Advanced Chart Builder"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 🎨 Professional Chart Builder")
                                gr.Markdown("**📌 Power BI-like features:** Data limiting, color schemes, auto-recommendations")
                                
                                dataset_select_viz = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True,
                                    info="Load a dataset first, then refresh dropdowns"
                                )
                                
                                chart_type_select = gr.Dropdown(
                                    choices=[
                                        "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                                        "Box Plot", "Violin Plot", "Pie Chart", "Donut Chart",
                                        "Heatmap", "Treemap", "Area Chart", "Funnel Chart"
                                    ],
                                    label="Chart Type",
                                    value="Bar Chart",
                                    interactive=True
                                )
                                
                                x_column_select = gr.Dropdown(label="X-axis Column", choices=[], interactive=True)
                                y_column_select = gr.Dropdown(label="Y-axis Column (optional)", choices=[], interactive=True)
                                color_column_select = gr.Dropdown(label="Color Column (optional)", choices=[], interactive=True)
                                size_column_select = gr.Dropdown(label="Size Column (optional)", choices=[], interactive=True)
                                
                                gr.Markdown("### 🎯 Power BI Features")
                                
                                limit_slider = gr.Slider(
                                    minimum=0,
                                    maximum=1000,
                                    value=0,
                                    step=5,
                                    label="Limit Data (0 = All)",
                                    info="Show Top 10, Top 20, etc. (0 shows all data)"
                                )
                                
                                color_scheme_select = gr.Dropdown(
                                    choices=list(COLOR_SCHEMES.keys()),
                                    label="Color Scheme",
                                    value="Professional Blue",
                                    interactive=True
                                )
                                
                                show_values_check = gr.Checkbox(
                                    label="Show Values on Chart",
                                    value=False
                                )
                                
                                chart_title_input = gr.Textbox(
                                    label="Custom Chart Title (optional)",
                                    placeholder="Leave empty for auto-generated title"
                                )
                                
                                create_viz_btn = gr.Button("🎨 Create Professional Chart", variant="primary", size="lg")
                                
                                # Update columns when dataset changes
                                dataset_select_viz.change(
                                    fn=get_columns,
                                    inputs=dataset_select_viz,
                                    outputs=[x_column_select, y_column_select, color_column_select, size_column_select]
                                )
                            
                            with gr.Column(scale=2):
                                viz_output = gr.Plot(label="Chart Output")
                                viz_status = gr.Textbox(label="Status", lines=3)
                                
                                with gr.Accordion("📘 Power BI Recreation Guide", open=False):
                                    powerbi_guide_output = gr.Markdown(label="How to recreate this in Power BI")
                                
                                create_viz_btn.click(
                                    fn=create_advanced_visualization,
                                    inputs=[
                                        dataset_select_viz, chart_type_select, x_column_select, 
                                        y_column_select, color_column_select, size_column_select,
                                        limit_slider, color_scheme_select, show_values_check, chart_title_input
                                    ],
                                    outputs=[viz_output, viz_status, powerbi_guide_output]
                                )
                    
                    # Tab 2: Auto Dashboard Generator
                    with gr.Tab("📊 Smart Dashboard"):
                        gr.Markdown("### 🎯 Intelligent Dashboard Generator")
                        gr.Markdown("""
                        **Automatically creates the BEST visualizations for your data!**
                        - Analyzes your data intelligently
                        - Creates as many charts as needed (not limited to 4!)
                        - Shows all charts together in one beautiful dashboard
                        - Works with any data type
                        - Like Power BI - all charts in one place!
                        """)
                        
                        with gr.Row():
                            dataset_select_dashboard = gr.Dropdown(
                                choices=list(state.datasets.keys()),
                                label="Select Dataset",
                                interactive=True,
                                scale=3
                            )
                            refresh_dashboard_dropdown = gr.Button("🔄 Refresh", scale=1)
                        
                        generate_dashboard_btn = gr.Button("🚀 Generate Smart Dashboard", variant="primary", size="lg")
                        
                        dashboard_summary = gr.Markdown(label="Dataset Summary")
                        dashboard_plot = gr.Plot(label="Complete Dashboard - All Charts in One View")
                        
                        def refresh_dashboard_dropdown_fn():
                            choices = list(state.datasets.keys())
                            return gr.update(choices=choices, value=choices[-1] if choices else None)
                        
                        refresh_dashboard_dropdown.click(
                            fn=refresh_dashboard_dropdown_fn,
                            outputs=dataset_select_dashboard
                        )
                        
                        def generate_smart_dashboard(dataset_name):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset", None
                            try:
                                from utils.smart_dashboard import analyze_and_create_dashboard, create_summary_stats
                                
                                df = state.datasets[dataset_name]
                                
                                # Create smart dashboard
                                fig = analyze_and_create_dashboard(df, dataset_name)
                                
                                # Create summary
                                summary = create_summary_stats(df)
                                summary += f"\n\n✅ **Dashboard generated successfully!**\n"
                                summary += f"📊 **All visualizations are shown below in one interactive dashboard**\n"
                                summary += f"💡 **Tip:** Scroll and zoom to explore each chart!"
                                
                                return summary, fig
                            except Exception as e:
                                import traceback
                                return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
                        
                        generate_dashboard_btn.click(
                            fn=generate_smart_dashboard,
                            inputs=dataset_select_dashboard,
                            outputs=[dashboard_summary, dashboard_plot]
                        )
                    
                    # Tab 3: Chart Recommendations
                    with gr.Tab("💡 Smart Recommendations"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 💡 AI Chart Recommendations")
                                gr.Markdown("**Get suggestions** for the best chart types based on your data")
                                
                                dataset_select_recommend = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True
                                )
                                
                                x_col_recommend = gr.Dropdown(label="Select Column to Analyze", choices=[], interactive=True)
                                y_col_recommend = gr.Dropdown(label="Second Column (optional)", choices=[], interactive=True)
                                
                                get_recommendations_btn = gr.Button("💡 Get Recommendations", variant="primary")
                                
                                def update_recommend_cols(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        cols = state.datasets[dataset_name].columns.tolist()
                                        return gr.update(choices=cols, value=cols[0] if cols else None), gr.update(choices=cols, value=None)
                                    return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
                                
                                dataset_select_recommend.change(
                                    fn=update_recommend_cols,
                                    inputs=dataset_select_recommend,
                                    outputs=[x_col_recommend, y_col_recommend]
                                )
                                
                                recommendations_output = gr.Markdown(label="Recommendations")
                                
                                def get_recommendations_wrapper(dataset_name, x_col, y_col):
                                    if not dataset_name or dataset_name not in state.datasets:
                                        return "Please select a dataset"
                                    if not x_col:
                                        return "Please select a column"
                                    
                                    df = state.datasets[dataset_name]
                                    recommendations = recommend_chart_types(df, x_col, y_col)
                                    
                                    output = f"# 💡 Chart Recommendations for '{x_col}'\n\n"
                                    for i, rec in enumerate(recommendations, 1):
                                        output += f"## {i}. {rec['type']} (Score: {rec['score']}/100)\n"
                                        output += f"**Why:** {rec['reason']}\n\n"
                                        output += f"**Best for:** {rec['best_for']}\n\n"
                                        output += "---\n\n"
                                    
                                    return output
                                
                                get_recommendations_btn.click(
                                    fn=get_recommendations_wrapper,
                                    inputs=[dataset_select_recommend, x_col_recommend, y_col_recommend],
                                    outputs=recommendations_output
                                )
                    
                    # Tab 4: Color Scheme Preview
                    with gr.Tab("🎨 Color Schemes"):
                        gr.Markdown("### 🎨 Professional Color Schemes")
                        gr.Markdown("Preview all available color schemes for your charts")
                        
                        with gr.Row():
                            for scheme_name, colors in list(COLOR_SCHEMES.items())[:4]:
                                with gr.Column():
                                    gr.Markdown(f"**{scheme_name}**")
                                    # Create a simple color preview
                                    color_html = "<div style='display: flex; height: 50px;'>"
                                    for color in colors[:5]:
                                        color_html += f"<div style='flex: 1; background-color: {color};'></div>"
                                    color_html += "</div>"
                                    gr.HTML(color_html)
                        
                        with gr.Row():
                            for scheme_name, colors in list(COLOR_SCHEMES.items())[4:]:
                                with gr.Column():
                                    gr.Markdown(f"**{scheme_name}**")
                                    color_html = "<div style='display: flex; height: 50px;'>"
                                    for color in colors[:5]:
                                        color_html += f"<div style='flex: 1; background-color: {color};'></div>"
                                    color_html += "</div>"
                                    gr.HTML(color_html)
            
            # ========== DATA PREPROCESSING ==========
            with gr.Tab("🧹 Data Preprocessing"):
                with gr.Tabs():
                    with gr.Tab("🔧 Data Cleaning"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🔧 Data Cleaning Tools")
                                dataset_select_prep = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True
                                )
                                
                                gr.Markdown("### 🕳️ Missing Value Handling")
                                missing_strategy = gr.Radio(
                                    choices=["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                                    label="Missing Value Strategy",
                                    value="Drop rows"
                                )
                                handle_missing_btn = gr.Button("🔧 Apply Missing Value Treatment", variant="primary")
                                missing_status = gr.Textbox(label="Status", lines=3)
                                
                                handle_missing_btn.click(
                                    fn=handle_missing_values,
                                    inputs=[dataset_select_prep, missing_strategy],
                                    outputs=missing_status
                                )
                                
                                gr.Markdown("### 🗑️ Duplicate Removal")
                                remove_dup_btn = gr.Button("🗑️ Remove Duplicate Rows", variant="secondary")
                                dup_status = gr.Textbox(label="Status", lines=3)
                                
                                remove_dup_btn.click(
                                    fn=remove_duplicates,
                                    inputs=dataset_select_prep,
                                    outputs=dup_status
                                )
                    
                    with gr.Tab("🔬 Data Transformation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 🔬 Advanced Data Transformations")
                                dataset_select_transform = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True
                                )
                                
                                gr.Markdown("### ⚖️ Feature Scaling")
                                scaling_method = gr.Radio(
                                    choices=["StandardScaler", "MinMaxScaler", "RobustScaler"],
                                    label="Scaling Method",
                                    value="StandardScaler"
                                )
                                scale_columns = gr.CheckboxGroup(
                                    choices=[],
                                    label="Select Numeric Columns to Scale"
                                )
                                apply_scaling_btn = gr.Button("⚖️ Apply Scaling", variant="primary")
                                scaling_status = gr.Textbox(label="Scaling Status", lines=3)
                                
                                # Update columns when dataset changes
                                def update_numeric_cols(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        numeric_cols = state.datasets[dataset_name].select_dtypes(include=[np.number]).columns.tolist()
                                        return gr.update(choices=numeric_cols, value=numeric_cols)
                                    return gr.update(choices=[], value=[])
                                
                                dataset_select_transform.change(
                                    fn=update_numeric_cols,
                                    inputs=dataset_select_transform,
                                    outputs=scale_columns
                                )
                                
                                def apply_scaling(dataset_name, method, columns):
                                    if not dataset_name or dataset_name not in state.datasets:
                                        return "Please select a dataset"
                                    if not columns:
                                        return "Please select columns to scale"
                                    
                                    try:
                                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                                        
                                        df = state.datasets[dataset_name].copy()
                                        
                                        if method == "StandardScaler":
                                            scaler = StandardScaler()
                                        elif method == "MinMaxScaler":
                                            scaler = MinMaxScaler()
                                        else:
                                            scaler = RobustScaler()
                                        
                                        df[columns] = scaler.fit_transform(df[columns])
                                        
                                        new_name = f"{dataset_name}_scaled"
                                        state.add_dataset(new_name, df)
                                        
                                        result = f"✅ Applied {method} to {len(columns)} columns\n"
                                        result += f"New dataset: {new_name}"
                                        return result
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}"
                                
                                apply_scaling_btn.click(
                                    fn=apply_scaling,
                                    inputs=[dataset_select_transform, scaling_method, scale_columns],
                                    outputs=scaling_status
                                )
                                
                                gr.Markdown("### 🏷️ Encoding Categorical Variables")
                                encoding_method = gr.Radio(
                                    choices=["Label Encoding", "One-Hot Encoding"],
                                    label="Encoding Method",
                                    value="One-Hot Encoding"
                                )
                                encode_columns = gr.CheckboxGroup(
                                    choices=[],
                                    label="Select Categorical Columns to Encode"
                                )
                                apply_encoding_btn = gr.Button("🏷️ Apply Encoding", variant="secondary")
                                encoding_status = gr.Textbox(label="Encoding Status", lines=3)
                                
                                def update_categorical_cols(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        cat_cols = state.datasets[dataset_name].select_dtypes(include=['object', 'category']).columns.tolist()
                                        return gr.update(choices=cat_cols, value=cat_cols)
                                    return gr.update(choices=[], value=[])
                                
                                dataset_select_transform.change(
                                    fn=update_categorical_cols,
                                    inputs=dataset_select_transform,
                                    outputs=encode_columns
                                )
                                
                                def apply_encoding(dataset_name, method, columns):
                                    if not dataset_name or dataset_name not in state.datasets:
                                        return "Please select a dataset"
                                    if not columns:
                                        return "Please select columns to encode"
                                    
                                    try:
                                        df = state.datasets[dataset_name].copy()
                                        
                                        if method == "Label Encoding":
                                            from sklearn.preprocessing import LabelEncoder
                                            le = LabelEncoder()
                                            for col in columns:
                                                df[col] = le.fit_transform(df[col].astype(str))
                                        else:  # One-Hot Encoding
                                            df = pd.get_dummies(df, columns=columns, drop_first=True)
                                        
                                        new_name = f"{dataset_name}_encoded"
                                        state.add_dataset(new_name, df)
                                        
                                        result = f"✅ Applied {method} to {len(columns)} columns\n"
                                        result += f"New dataset: {new_name}\n"
                                        result += f"New shape: {df.shape[0]} rows × {df.shape[1]} columns"
                                        return result
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}"
                                
                                apply_encoding_btn.click(
                                    fn=apply_encoding,
                                    inputs=[dataset_select_transform, encoding_method, encode_columns],
                                    outputs=encoding_status
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### 📖 Transformation Guide")
                                gr.Markdown("""
                                **Scaling Methods:**
                                - **StandardScaler**: Mean=0, Std=1 (for normally distributed data)
                                - **MinMaxScaler**: Scale to [0,1] range
                                - **RobustScaler**: Uses median and IQR (robust to outliers)
                                
                                **Encoding Methods:**
                                - **Label Encoding**: Convert categories to numbers (0, 1, 2...)
                                - **One-Hot Encoding**: Create binary columns for each category
                                
                                **Best Practices:**
                                - Scale features before training ML models
                                - Use One-Hot for nominal categories
                                - Use Label Encoding for ordinal categories
                                """)
            
            # ========== MACHINE LEARNING ==========
            with gr.Tab("🤖 ML Studio"):
                with gr.Tabs():
                    with gr.Tab("🏋️ Model Training"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 🏋️ Train Machine Learning Models")
                                gr.Markdown("**📌 Tip:** Load and preprocess your data first. Then refresh dropdowns using the button at the top.")
                                dataset_select_model = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True,
                                    info="Load a dataset first to see options"
                                )
                                
                                target_column_select = gr.Dropdown(
                                    label="Target Column",
                                    choices=[],
                                    interactive=True
                                )
                                
                                algorithm_select = gr.Dropdown(
                                    choices=[
                                        "Random Forest", 
                                        "XGBoost",
                                        "LightGBM",
                                        "Logistic/Linear Regression", 
                                        "SVM"
                                    ],
                                    label="Algorithm",
                                    value="Random Forest",
                                    interactive=True
                                )
                                
                                test_size_slider = gr.Slider(
                                    minimum=10,
                                    maximum=50,
                                    value=20,
                                    step=5,
                                    label="Test Size (%)"
                                )
                                
                                cross_validation_checkbox = gr.Checkbox(
                                    label="Enable Cross-Validation",
                                    value=False
                                )
                                
                                train_btn = gr.Button("🚀 Train Model", variant="primary")
                                
                                # Update target columns when dataset changes
                                dataset_select_model.change(
                                    fn=update_target_cols,
                                    inputs=dataset_select_model,
                                    outputs=target_column_select
                                )
                            
                            with gr.Column(scale=2):
                                model_output = gr.Markdown(label="Training Results", height=500)
                                
                                train_model_click = train_btn.click(
                                    fn=train_model,
                                    inputs=[
                                        dataset_select_model, target_column_select, 
                                        algorithm_select, test_size_slider, cross_validation_checkbox
                                    ],
                                    outputs=model_output
                                )
                                
                                gr.Markdown("### 📚 Algorithm Guide")
                                gr.Markdown("""
                                - **Random Forest**: Ensemble method, works well for most problems
                                - **XGBoost**: ⚡ Fast gradient boosting, excellent performance
                                - **LightGBM**: ⚡ Ultra-fast, memory efficient, great for large datasets
                                - **Logistic/Linear Regression**: Simple, interpretable models
                                - **SVM**: Powerful for complex decision boundaries
                                
                                💡 **Recommended:** Try XGBoost or LightGBM for best results!
                                """)
                    
                    with gr.Tab("📊 Model Evaluation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📊 Model Performance Analysis")
                                
                                model_select = gr.Dropdown(
                                    choices=list(state.models.keys()),
                                    label="Select Trained Model",
                                    interactive=True
                                )
                                
                                evaluate_btn = gr.Button("📊 Evaluate Model", variant="primary")
                                
                                gr.Markdown("### 📈 Available Metrics")
                                gr.Markdown("""
                                **Classification:**
                                - Accuracy, Precision, Recall
                                - F1 Score, ROC-AUC
                                - Confusion Matrix
                                
                                **Regression:**
                                - R² Score, MSE, RMSE
                                - MAE, MAPE
                                - Residual Analysis
                                """)
                            
                            with gr.Column(scale=2):
                                model_eval_output = gr.Markdown(label="Evaluation Results", height=400)
                                model_eval_plot = gr.Plot(label="Performance Visualization")
                                
                                def evaluate_model(model_name):
                                    if not model_name or model_name not in state.models:
                                        return "Please select a model", None
                                    
                                    try:
                                        model_info = state.models[model_name]
                                        
                                        result = f"# Model Evaluation: {model_name}\n\n"
                                        result += f"## Model Details\n"
                                        result += f"- **Target**: {model_info['target']}\n"
                                        result += f"- **Features**: {len(model_info['features'])}\n"
                                        result += f"- **Score**: {model_info['score']:.4f}\n\n"
                                        
                                        result += "## Feature List\n"
                                        for i, feat in enumerate(model_info['features'][:10], 1):
                                            result += f"{i}. {feat}\n"
                                        if len(model_info['features']) > 10:
                                            result += f"... and {len(model_info['features'])-10} more features\n"
                                        
                                        # Create a simple performance gauge
                                        fig = go.Figure()
                                        fig.add_trace(go.Indicator(
                                            mode="gauge+number",
                                            value=model_info['score'] * 100,
                                            title={'text': "Model Score (%)"},
                                            gauge={'axis': {'range': [0, 100]},
                                                   'bar': {'color': "darkblue"},
                                                   'steps': [
                                                       {'range': [0, 60], 'color': "lightgray"},
                                                       {'range': [60, 80], 'color': "gray"},
                                                       {'range': [80, 100], 'color': "lightgreen"}]}))
                                        
                                        return result, fig
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}", None
                                
                                evaluate_btn.click(
                                    fn=evaluate_model,
                                    inputs=model_select,
                                    outputs=[model_eval_output, model_eval_plot]
                                )
                    
                    with gr.Tab("🎯 Feature Importance"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🎯 Feature Importance Analysis")
                                
                                model_select_fi = gr.Dropdown(
                                    choices=list(state.models.keys()),
                                    label="Select Trained Model",
                                    interactive=True
                                )
                                
                                get_importance_btn = gr.Button("🎯 Get Feature Importance", variant="primary")
                                importance_output = gr.Markdown(label="Feature Importance")
                                importance_plot = gr.Plot(label="Importance Visualization")
                                
                                def get_feature_importance(model_name):
                                    if not model_name or model_name not in state.models:
                                        return "Please select a model", None
                                    
                                    try:
                                        model_info = state.models[model_name]
                                        model = model_info['model']
                                        
                                        # Check if model has feature_importances_
                                        if hasattr(model, 'feature_importances_'):
                                            importances = model.feature_importances_
                                            features = model_info['features']
                                            
                                            # Create dataframe
                                            importance_df = pd.DataFrame({
                                                'Feature': features,
                                                'Importance': importances
                                            }).sort_values('Importance', ascending=False)
                                            
                                            # Create text output
                                            result = f"# Feature Importance: {model_name}\n\n"
                                            result += "## Top 15 Features\n"
                                            for i, row in importance_df.head(15).iterrows():
                                                result += f"{row.name+1}. **{row['Feature']}**: {row['Importance']:.4f}\n"
                                            
                                            # Create plot
                                            fig = px.bar(
                                                importance_df.head(15),
                                                x='Importance',
                                                y='Feature',
                                                orientation='h',
                                                title='Top 15 Feature Importances'
                                            )
                                            fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                                            
                                            return result, fig
                                        else:
                                            return "This model type doesn't support feature importance", None
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}", None
                                
                                get_importance_btn.click(
                                    fn=get_feature_importance,
                                    inputs=model_select_fi,
                                    outputs=[importance_output, importance_plot]
                                )
            
            # ========== EXPORT & REPORTING ==========
            with gr.Tab("💾 Export & Reports"):
                with gr.Tabs():
                    with gr.Tab("📄 PDF Report (Recommended)"):
                        gr.Markdown("### � Prpofessional PDF Report")
                        gr.Markdown("""
                        **🎯 RECOMMENDED: Generate comprehensive PDF report with everything!**
                        """)
                        gr.Markdown("""
                        Generate a comprehensive A4-sized PDF report with:
                        - Executive Summary
                        - Data Overview & Statistics
                        - Complete Cleaning History
                        - AI Chat History
                        - Models Trained
                        - Full-Page Visualizations
                        - Operation History
                        - Professional Recommendations
                        """)
                        
                        gr.Markdown("""
                        **📌 Note:** Upload data first in "Data Management" tab, then click refresh below.
                        """)
                        
                        with gr.Row():
                            report_dataset_select = gr.Dropdown(
                                choices=list(state.datasets.keys()),
                                label="Select Dataset for Report",
                                interactive=True,
                                info="Choose which dataset to include in the report",
                                scale=3
                            )
                            refresh_report_dropdown = gr.Button("🔄 Refresh", scale=1)
                        
                        def refresh_report_dropdown_fn():
                            choices = list(state.datasets.keys())
                            return gr.update(choices=choices, value=choices[-1] if choices else None)
                        
                        refresh_report_dropdown.click(
                            fn=refresh_report_dropdown_fn,
                            outputs=report_dataset_select
                        )
                        
                        generate_report_btn = gr.Button("📊 Generate Professional Report", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        ⏱️ **Generation time:** 30-60 seconds
                        📄 **Output:** Comprehensive A4-sized PDF with all project details
                        """)
                        
                        report_status = gr.Textbox(label="Report Status", lines=3)
                        report_file = gr.File(label="Download Report PDF")
                        
                        def generate_report(dataset_name):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset", None
                            try:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                output_path = f"reports/project_report_{dataset_name}_{timestamp}.pdf"
                                
                                # Create reports directory if it doesn't exist
                                os.makedirs("reports", exist_ok=True)
                                
                                # Generate report
                                pdf_path = generate_professional_report(state, dataset_name, output_path)
                                
                                status = f"""
✅ Professional Report Generated Successfully!

📄 Report: {os.path.basename(pdf_path)}
📊 Dataset: {dataset_name}
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The report includes:
✓ Executive Summary
✓ Data Overview ({state.datasets[dataset_name].shape[0]:,} rows × {state.datasets[dataset_name].shape[1]} columns)
✓ Cleaning History ({len([op for op in state.history if 'clean' in op.get('operation', '').lower()])} operations)
✓ AI Chat History ({len(state.ai_chat_history)} interactions)
✓ Models Trained ({len(state.models)} models)
✓ Full-Page Visualizations
✓ Complete Operation History
✓ Professional Recommendations

📥 Download your report below!
                                """
                                
                                return status, pdf_path
                            except Exception as e:
                                return f"❌ Error generating report: {str(e)}", None
                        
                        generate_report_btn.click(
                            fn=generate_report,
                            inputs=report_dataset_select,
                            outputs=[report_status, report_file]
                        )
                    
                    with gr.Tab("� DPowerPoint (PPT)"):
                        gr.Markdown("### 📊 Generate PowerPoint Presentations")
                        gr.Markdown("""
                        **Two Options:**
                        - **Quick PPT**: Fast, basic slides (5-10 seconds)
                        - **AI-Powered PPT**: Detailed, LLM-generated content (30-60 seconds)
                        """)
                        
                        with gr.Row():
                            ppt_dataset_select = gr.Dropdown(
                                choices=list(state.datasets.keys()),
                                label="Select Dataset",
                                interactive=True,
                                scale=3
                            )
                            refresh_ppt_dropdown = gr.Button("🔄 Refresh", scale=1)
                        
                        ppt_type = gr.Radio(
                            choices=["Quick PPT (Fast)", "AI-Powered PPT (Detailed)"],
                            label="Presentation Type",
                            value="Quick PPT (Fast)"
                        )
                        
                        generate_ppt_btn = gr.Button("📊 Generate PowerPoint", variant="primary", size="lg")
                        
                        ppt_status = gr.Textbox(label="Status", lines=3)
                        ppt_file = gr.File(label="Download PowerPoint")
                        
                        def refresh_ppt_dropdown_fn():
                            choices = list(state.datasets.keys())
                            return gr.update(choices=choices, value=choices[-1] if choices else None)
                        
                        refresh_ppt_dropdown.click(
                            fn=refresh_ppt_dropdown_fn,
                            outputs=ppt_dataset_select
                        )
                        
                        def generate_ppt(dataset_name, ppt_type):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset", None
                            try:
                                from utils.ppt_generator import generate_quick_ppt, generate_ai_ppt
                                
                                df = state.datasets[dataset_name]
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                os.makedirs("presentations", exist_ok=True)
                                
                                if "Quick" in ppt_type:
                                    output_path = f"presentations/{dataset_name}_quick_{timestamp}.pptx"
                                    ppt_path = generate_quick_ppt(df, dataset_name, output_path)
                                    status = f"""
✅ Quick PowerPoint Generated!

📊 Presentation: {os.path.basename(ppt_path)}
⚡ Type: Quick (Basic slides)
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Includes:
✓ Title slide
✓ Data overview
✓ Summary statistics
✓ Ready to present!

📥 Download below!
                                    """
                                else:
                                    # AI-Powered PPT
                                    ai_insights = generate_ai_insights(dataset_name) if dataset_name in state.datasets else ""
                                    output_path = f"presentations/{dataset_name}_ai_{timestamp}.pptx"
                                    ppt_path = generate_ai_ppt(df, dataset_name, ai_insights, output_path)
                                    status = f"""
✅ AI-Powered PowerPoint Generated!

📊 Presentation: {os.path.basename(ppt_path)}
🤖 Type: AI-Powered (Detailed with LLM insights)
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Includes:
✓ Professional themed slides
✓ Executive summary
✓ AI-generated insights
✓ Data quality assessment
✓ Ready for business presentation!

📥 Download below!
                                    """
                                
                                return status, ppt_path
                            except Exception as e:
                                import traceback
                                return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None
                        
                        generate_ppt_btn.click(
                            fn=generate_ppt,
                            inputs=[ppt_dataset_select, ppt_type],
                            outputs=[ppt_status, ppt_file]
                        )
                    
                    with gr.Tab("💾 Data Export"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 💾 Export Data Files")
                                dataset_select_export = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset to Export",
                                    interactive=True
                                )
                                
                                export_format_select = gr.Radio(
                                    choices=["CSV", "Excel", "JSON", "Power BI", "Parquet"],
                                    label="Export Format",
                                    value="CSV"
                                )
                                
                                export_btn = gr.Button("💾 Export Dataset", variant="primary")
                                export_status = gr.Textbox(label="Export Status", lines=3)
                                export_file = gr.File(label="Download File")
                                
                                export_btn.click(
                                    fn=export_dataset,
                                    inputs=[dataset_select_export, export_format_select],
                                    outputs=[export_status, export_file]
                                )
                                
                                gr.Markdown("### 📤 Export Formats")
                                gr.Markdown("""
                                - **CSV**: Universal format for data exchange
                                - **Excel**: With formatting for business use
                                - **JSON**: For APIs and web applications
                                - **Power BI**: Optimized for Power BI integration
                                - **Parquet**: Efficient columnar format for big data
                                """)
            
            # ========== AI ASSISTANT ==========
            with gr.Tab("🧠 AI Assistant"):
                with gr.Tabs():
                    with gr.Tab("💬 Chat"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 💬 AI Data Assistant")
                                dataset_select_ai = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset for Analysis",
                                    interactive=True
                                )
                                
                                ai_question = gr.Textbox(
                                    label="Ask a Question About Your Data",
                                    placeholder="e.g., What are the key insights from this data?",
                                    lines=3,
                                    max_lines=5
                                )
                                ask_ai_btn = gr.Button("🚀 Ask AI Assistant", variant="primary")
                                
                                gr.Markdown("### 📌 Example Questions")
                                gr.Markdown("""
                                - What are the main patterns in this dataset?
                                - Which columns have the most missing values?
                                - What's the correlation between [column A] and [column B]?
                                - Are there any outliers in [column name]?
                                - What insights can you provide about [specific aspect]?
                                """)
                            
                            with gr.Column(scale=2):
                                ai_response = gr.Markdown(label="AI Response", height=400)
                                
                                ask_ai_btn.click(
                                    fn=ask_ai,
                                    inputs=[dataset_select_ai, ai_question],
                                    outputs=ai_response
                                )
                    
                    with gr.Tab("🔮 Insights"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🔮 Automated AI Insights")
                                dataset_select_insights = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset for Insights",
                                    interactive=True
                                )
                                
                                generate_insights_btn = gr.Button("🔮 Generate AI Insights", variant="primary")
                                ai_insights_output = gr.Markdown(label="AI Insights", height=600)
                                
                                generate_insights_btn.click(
                                    fn=generate_ai_insights,
                                    inputs=dataset_select_insights,
                                    outputs=ai_insights_output
                                )
                    
                    with gr.Tab("� Ahuto-Fix Data"):
                        gr.Markdown("""
                        ### 🔧 AI-Powered Auto-Fix
                        Let AI analyze your data and automatically fix all issues!
                        """)
                        with gr.Row():
                            with gr.Column(scale=1):
                                dataset_select_autofix = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset to Fix",
                                    interactive=True
                                )
                                autofix_btn = gr.Button("🔧 Analyze & Auto-Fix", variant="primary", size="lg")
                                
                                gr.Markdown("""
                                ### What AI Will Do:
                                1. 🔍 Analyze data quality issues
                                2. 🧠 Generate recommendations
                                3. ✨ Automatically fix all issues
                                4. 💾 Save cleaned data
                                
                                **Perfect for beginners!**
                                """)
                            
                            with gr.Column(scale=2):
                                autofix_status = gr.Markdown(label="AI Analysis & Fixes")
                                autofix_preview = gr.Dataframe(label="Fixed Data Preview")
                        
                        def ai_autofix_data(dataset_name):
                            if not dataset_name or dataset_name not in state.datasets:
                                return "❌ Select dataset", pd.DataFrame()
                            try:
                                df = state.datasets[dataset_name].copy()
                                original_shape = df.shape
                                
                                # Step 1: Get AI analysis using LLM
                                report = f"### 🔍 AI Analysis for '{dataset_name}'\n\n"
                                report += "[*] Asking LLM to analyze data quality...\n\n"
                                
                                # Get AI insights about data quality
                                try:
                                    ai_analysis = generate_ai_insights(dataset_name)
                                    report += "### 🤖 LLM Identified Issues:\n\n"
                                    report += ai_analysis[:800] + "...\n\n"
                                except:
                                    report += "### 🤖 LLM Analysis: Using standard cleaning\n\n"
                                
                                # Step 2: Apply intelligent fixes based on common issues
                                report += "### ✨ Applying Intelligent Fixes:\n\n"
                                fixes_applied = []
                                
                                # Fix 1: Remove negative values in quantity/price columns
                                for col in df.columns:
                                    if any(keyword in col.lower() for keyword in ['quantity', 'price', 'amount', 'cost', 'sales']):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            neg_count = (df[col] < 0).sum()
                                            if neg_count > 0:
                                                df[col] = df[col].abs()
                                                fixes_applied.append(f"✓ Fixed {neg_count} negative values in '{col}'")
                                
                                # Fix 2: Remove zero values where they don't make sense
                                for col in df.columns:
                                    if any(keyword in col.lower() for keyword in ['quantity', 'price']):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            zero_count = (df[col] == 0).sum()
                                            if zero_count > 0 and zero_count < len(df) * 0.5:  # Only if < 50% are zeros
                                                df = df[df[col] != 0]
                                                fixes_applied.append(f"✓ Removed {zero_count} zero values in '{col}'")
                                
                                # Fix 3: Handle outliers using IQR method (only for numeric, not datetime)
                                numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                                for col in numeric_cols:
                                    try:
                                        Q1 = df[col].quantile(0.25)
                                        Q3 = df[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        if IQR > 0:
                                            lower_bound = Q1 - 3 * IQR
                                            upper_bound = Q3 + 3 * IQR
                                            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                                            if outliers > 0:
                                                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                                                fixes_applied.append(f"✓ Capped {outliers} outliers in '{col}' using IQR method")
                                    except Exception as e:
                                        # Skip columns that can't be processed
                                        continue
                                
                                # Fix 4: Handle missing values intelligently
                                for col in df.columns:
                                    missing = df[col].isnull().sum()
                                    if missing > 0:
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            df[col].fillna(df[col].median(), inplace=True)
                                            fixes_applied.append(f"✓ Filled {missing} missing values in '{col}' with median")
                                        else:
                                            mode_val = df[col].mode()
                                            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                                            df[col].fillna(fill_val, inplace=True)
                                            fixes_applied.append(f"✓ Filled {missing} missing values in '{col}' with mode")
                                
                                # Fix 5: Remove duplicates
                                dup_count = df.duplicated().sum()
                                if dup_count > 0:
                                    df = df.drop_duplicates()
                                    fixes_applied.append(f"✓ Removed {dup_count} duplicate rows")
                                
                                # Fix 6: Standardize categorical data
                                for col in df.select_dtypes(include=['object']).columns:
                                    if df[col].nunique() < 50:  # Only for columns with reasonable categories
                                        try:
                                            df[col] = df[col].str.strip().str.title()
                                            fixes_applied.append(f"✓ Standardized categorical values in '{col}'")
                                        except:
                                            pass
                                
                                # Fix 7: Convert date columns
                                for col in df.columns:
                                    if 'date' in col.lower():
                                        try:
                                            df[col] = pd.to_datetime(df[col], errors='coerce')
                                            fixes_applied.append(f"✓ Converted '{col}' to datetime format")
                                        except:
                                            pass
                                
                                # Step 3: Save fixed data
                                new_name = f"{dataset_name}_AI_fixed"
                                state.add_dataset(new_name, df)
                                
                                # Step 4: Generate comprehensive report
                                report += "\n".join(fixes_applied)
                                report += f"\n\n### ✅ AI Auto-Fix Complete!\n\n"
                                report += f"**Original:** {original_shape[0]:,} rows × {original_shape[1]} columns\n\n"
                                report += f"**Cleaned:** {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
                                report += f"**Rows Removed:** {original_shape[0] - df.shape[0]:,}\n\n"
                                report += f"**Total Fixes Applied:** {len(fixes_applied)}\n\n"
                                report += f"### 💾 Saved As: `{new_name}`\n\n"
                                report += "### ✨ Your data is now clean and ready to use!\n"
                                report += f"**The fixed dataset '{new_name}' is now available in all dropdowns.**\n\n"
                                report += "### 🎯 Issues Addressed:\n"
                                report += "✓ Negative values in quantity/price columns\n"
                                report += "✓ Zero values where inappropriate\n"
                                report += "✓ Outliers using IQR method\n"
                                report += "✓ Missing values with smart imputation\n"
                                report += "✓ Duplicate rows\n"
                                report += "✓ Categorical data standardization\n"
                                report += "✓ Date format conversion\n"
                                
                                return report, df.head(20)
                            except Exception as e:
                                import traceback
                                return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", pd.DataFrame()
                        
                        autofix_click = autofix_btn.click(
                            fn=ai_autofix_data,
                            inputs=dataset_select_autofix,
                            outputs=[autofix_status, autofix_preview]
                        )
                    
                    with gr.Tab("💭 Chat History"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 💭 AI Chat History")
                                ai_chat_history_display = gr.Markdown(value=get_ai_chat_history())
                                refresh_ai_chat = gr.Button("🔄 Refresh Chat History")
                                
                                refresh_ai_chat.click(
                                    fn=get_ai_chat_history,
                                    outputs=ai_chat_history_display
                                )
            
            # ========== ADVANCED ANALYTICS ==========
            with gr.Tab("📈 Advanced Analytics"):
                with gr.Tabs():
                    with gr.Tab("📊 Statistical Analysis"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📊 Statistical Analysis")
                                dataset_select_stats = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Select Dataset",
                                    interactive=True
                                )
                                
                                analysis_type = gr.Radio(
                                    choices=["Descriptive Statistics", "Correlation Analysis", "Distribution Analysis"],
                                    label="Analysis Type",
                                    value="Descriptive Statistics"
                                )
                                
                                run_analysis_btn = gr.Button("📊 Run Analysis", variant="primary")
                            
                            with gr.Column(scale=2):
                                stats_output = gr.Markdown(label="Analysis Results", height=500)
                                stats_plot = gr.Plot(label="Visualization")
                                
                                def run_statistical_analysis(dataset_name, analysis_type):
                                    if not dataset_name or dataset_name not in state.datasets:
                                        return "Please select a dataset", None
                                    
                                    try:
                                        df = state.datasets[dataset_name]
                                        
                                        if analysis_type == "Descriptive Statistics":
                                            desc = df.describe(include='all').T
                                            result = f"# Descriptive Statistics: {dataset_name}\n\n"
                                            result += desc.to_markdown()
                                            
                                            # Create box plot for numeric columns
                                            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
                                            if len(numeric_cols) > 0:
                                                fig = go.Figure()
                                                for col in numeric_cols:
                                                    fig.add_trace(go.Box(y=df[col], name=col))
                                                fig.update_layout(title="Distribution of Numeric Features", height=500)
                                                return result, fig
                                            return result, None
                                        
                                        elif analysis_type == "Correlation Analysis":
                                            numeric_df = df.select_dtypes(include=[np.number])
                                            if len(numeric_df.columns) < 2:
                                                return "Need at least 2 numeric columns for correlation analysis", None
                                            
                                            corr = numeric_df.corr()
                                            result = f"# Correlation Analysis: {dataset_name}\n\n"
                                            result += "## Correlation Matrix\n"
                                            result += corr.to_markdown()
                                            
                                            # Create heatmap
                                            fig = px.imshow(corr, text_auto='.2f', aspect="auto", 
                                                          title="Correlation Heatmap",
                                                          color_continuous_scale='RdBu_r')
                                            fig.update_layout(height=600)
                                            return result, fig
                                        
                                        else:  # Distribution Analysis
                                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                                            result = f"# Distribution Analysis: {dataset_name}\n\n"
                                            
                                            for col in numeric_cols[:5]:
                                                result += f"## {col}\n"
                                                result += f"- Mean: {df[col].mean():.2f}\n"
                                                result += f"- Median: {df[col].median():.2f}\n"
                                                result += f"- Std Dev: {df[col].std():.2f}\n"
                                                result += f"- Skewness: {df[col].skew():.2f}\n\n"
                                            
                                            # Create histogram
                                            if len(numeric_cols) > 0:
                                                fig = px.histogram(df, x=numeric_cols[0], 
                                                                 title=f"Distribution of {numeric_cols[0]}",
                                                                 marginal="box")
                                                fig.update_layout(height=500)
                                                return result, fig
                                            return result, None
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}", None
                                
                                run_analysis_btn.click(
                                    fn=run_statistical_analysis,
                                    inputs=[dataset_select_stats, analysis_type],
                                    outputs=[stats_output, stats_plot]
                                )
                    
                    with gr.Tab("� Datta Merge/Join"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### � Merge/eJoin Datasets")
                                
                                merge_dataset_a = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Left Dataset",
                                    interactive=True
                                )
                                merge_dataset_b = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Right Dataset",
                                    interactive=True
                                )
                                
                                merge_type = gr.Radio(
                                    choices=["inner", "left", "right", "outer"],
                                    label="Join Type",
                                    value="inner"
                                )
                                
                                merge_key_a = gr.Dropdown(
                                    choices=[],
                                    label="Key Column (Left)",
                                    interactive=True
                                )
                                merge_key_b = gr.Dropdown(
                                    choices=[],
                                    label="Key Column (Right)",
                                    interactive=True
                                )
                                
                                merge_btn = gr.Button("🔗 Merge Datasets", variant="primary")
                                
                                # Update key columns when datasets change
                                def update_merge_keys_a(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        cols = state.datasets[dataset_name].columns.tolist()
                                        return gr.update(choices=cols, value=cols[0] if cols else None)
                                    return gr.update(choices=[], value=None)
                                
                                def update_merge_keys_b(dataset_name):
                                    if dataset_name and dataset_name in state.datasets:
                                        cols = state.datasets[dataset_name].columns.tolist()
                                        return gr.update(choices=cols, value=cols[0] if cols else None)
                                    return gr.update(choices=[], value=None)
                                
                                merge_dataset_a.change(
                                    fn=update_merge_keys_a,
                                    inputs=merge_dataset_a,
                                    outputs=merge_key_a
                                )
                                merge_dataset_b.change(
                                    fn=update_merge_keys_b,
                                    inputs=merge_dataset_b,
                                    outputs=merge_key_b
                                )
                                
                                gr.Markdown("""
                                **Join Types:**
                                - **inner**: Only matching rows
                                - **left**: All from left + matching from right
                                - **right**: All from right + matching from left
                                - **outer**: All rows from both
                                """)
                            
                            with gr.Column(scale=2):
                                merge_output = gr.Markdown(label="Merge Results", height=400)
                                merge_preview = gr.Dataframe(label="Merged Data Preview")
                                
                                def merge_datasets(dataset_a_name, dataset_b_name, how, key_a, key_b):
                                    if not dataset_a_name or not dataset_b_name:
                                        return "Please select both datasets", pd.DataFrame()
                                    if dataset_a_name not in state.datasets or dataset_b_name not in state.datasets:
                                        return "Invalid dataset selection", pd.DataFrame()
                                    if not key_a or not key_b:
                                        return "Please select key columns", pd.DataFrame()
                                    
                                    try:
                                        df_a = state.datasets[dataset_a_name]
                                        df_b = state.datasets[dataset_b_name]
                                        
                                        merged = pd.merge(df_a, df_b, left_on=key_a, right_on=key_b, how=how)
                                        
                                        new_name = f"merged_{dataset_a_name}_{dataset_b_name}"
                                        state.add_dataset(new_name, merged)
                                        
                                        result = f"✅ Datasets merged successfully!\n\n"
                                        result += f"**Left Dataset**: {dataset_a_name} ({df_a.shape[0]} rows)\n"
                                        result += f"**Right Dataset**: {dataset_b_name} ({df_b.shape[0]} rows)\n"
                                        result += f"**Join Type**: {how}\n"
                                        result += f"**Result**: {merged.shape[0]} rows × {merged.shape[1]} columns\n"
                                        result += f"**Saved as**: {new_name}"
                                        
                                        state.log_action("Data Merge", f"Merged {dataset_a_name} and {dataset_b_name}", 
                                                       {"how": how, "result_shape": merged.shape})
                                        
                                        return result, merged.head(10)
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}", pd.DataFrame()
                                
                                merge_btn.click(
                                    fn=merge_datasets,
                                    inputs=[merge_dataset_a, merge_dataset_b, merge_type, merge_key_a, merge_key_b],
                                    outputs=[merge_output, merge_preview]
                                )
                    
                    with gr.Tab("🔄 Data Comparison"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🔄 Compare Two Datasets")
                                
                                dataset_a = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Dataset A",
                                    interactive=True
                                )
                                dataset_b = gr.Dropdown(
                                    choices=list(state.datasets.keys()),
                                    label="Dataset B",
                                    interactive=True
                                )
                                
                                compare_btn = gr.Button("🔄 Compare Datasets", variant="primary")
                                comparison_output = gr.Markdown(label="Comparison Results", height=600)
                                
                                def compare_datasets(dataset_a_name, dataset_b_name):
                                    if not dataset_a_name or not dataset_b_name:
                                        return "Please select both datasets"
                                    if dataset_a_name not in state.datasets or dataset_b_name not in state.datasets:
                                        return "Invalid dataset selection"
                                    
                                    try:
                                        df_a = state.datasets[dataset_a_name]
                                        df_b = state.datasets[dataset_b_name]
                                        
                                        result = f"# Dataset Comparison\n\n"
                                        result += f"## {dataset_a_name} vs {dataset_b_name}\n\n"
                                        
                                        result += "### Shape Comparison\n"
                                        result += f"- **{dataset_a_name}**: {df_a.shape[0]} rows × {df_a.shape[1]} columns\n"
                                        result += f"- **{dataset_b_name}**: {df_b.shape[0]} rows × {df_b.shape[1]} columns\n\n"
                                        
                                        result += "### Column Comparison\n"
                                        cols_a = set(df_a.columns)
                                        cols_b = set(df_b.columns)
                                        common = cols_a & cols_b
                                        only_a = cols_a - cols_b
                                        only_b = cols_b - cols_a
                                        
                                        result += f"- **Common columns**: {len(common)}\n"
                                        if common:
                                            result += f"  - {', '.join(list(common)[:10])}\n"
                                        result += f"- **Only in {dataset_a_name}**: {len(only_a)}\n"
                                        if only_a:
                                            result += f"  - {', '.join(list(only_a)[:10])}\n"
                                        result += f"- **Only in {dataset_b_name}**: {len(only_b)}\n"
                                        if only_b:
                                            result += f"  - {', '.join(list(only_b)[:10])}\n"
                                        
                                        result += "\n### Data Type Comparison\n"
                                        result += f"**{dataset_a_name}:**\n"
                                        for dtype, count in df_a.dtypes.value_counts().items():
                                            result += f"- {dtype}: {count}\n"
                                        result += f"\n**{dataset_b_name}:**\n"
                                        for dtype, count in df_b.dtypes.value_counts().items():
                                            result += f"- {dtype}: {count}\n"
                                        
                                        return result
                                    except Exception as e:
                                        return f"❌ Error: {str(e)}"
                                
                                compare_btn.click(
                                    fn=compare_datasets,
                                    inputs=[dataset_a, dataset_b],
                                    outputs=comparison_output
                                )
            
            # ========== HISTORY & SETTINGS ==========
            with gr.Tab("⚙️ History & Settings"):
                with gr.Tabs():
                    with gr.Tab("📚 Pipeline History"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 📚 Complete Pipeline History")
                                history_full = gr.Markdown(value=get_history())
                                
                                with gr.Row():
                                    refresh_history_btn = gr.Button("🔄 Refresh History")
                                    clear_history_btn_full = gr.Button("🗑️ Clear History")
                                
                                refresh_history_btn.click(
                                    fn=get_history,
                                    outputs=history_full
                                )
                                
                                clear_history_btn_full.click(
                                    fn=clear_history,
                                    outputs=history_full
                                )
                    
                    with gr.Tab("🔧 Preferences"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🔧 Platform Preferences")
                                gr.Markdown("""
                                Coming soon:
                                - Theme customization (Light/Dark mode)
                                - Auto-refresh settings
                                - Default chart preferences
                                - Notification settings
                                - Keyboard shortcuts
                                """)
                                gr.Markdown("🎯 *Preferences panel will be enhanced in the next update*")
        
        # ========== WIRE UP ALL BUTTONS AND DROPDOWNS ==========
        
        # Wire upload button to update all dropdowns
        upload_btn.click(
            fn=upload_file,
            inputs=file_upload,
            outputs=[
                upload_status, upload_preview, dataset_list_state
            ]
        ).then(
            fn=refresh_all_dropdowns_fn,
            outputs=[
                dataset_select_profile, dataset_select_overview, dataset_select_viz,
                dataset_select_prep, dataset_select_transform, dataset_select_model,
                dataset_select_export, dataset_select_ai, dataset_select_insights,
                dataset_select_stats, merge_dataset_a, merge_dataset_b,
                dataset_a, dataset_b, dataset_select_dashboard, dataset_select_recommend,
                dataset_select_autopilot, dataset_select_automl, dataset_select_autofix,
                report_dataset_select, ppt_dataset_select
            ]
        )
        
        # Wire sample load button to update all dropdowns
        load_sample_btn.click(
            fn=load_sample_data,
            inputs=sample_choice,
            outputs=[
                sample_status, sample_preview, dataset_list_state
            ]
        ).then(
            fn=refresh_all_dropdowns_fn,
            outputs=[
                dataset_select_profile, dataset_select_overview, dataset_select_viz,
                dataset_select_prep, dataset_select_transform, dataset_select_model,
                dataset_select_export, dataset_select_ai, dataset_select_insights,
                dataset_select_stats, merge_dataset_a, merge_dataset_b,
                dataset_a, dataset_b, dataset_select_dashboard, dataset_select_recommend,
                dataset_select_autopilot, dataset_select_automl, dataset_select_autofix,
                report_dataset_select, ppt_dataset_select
            ]
        )
        
        # Wire refresh all button
        refresh_all_btn.click(
            fn=refresh_all_dropdowns_fn,
            outputs=[
                dataset_select_profile, dataset_select_overview, dataset_select_viz,
                dataset_select_prep, dataset_select_transform, dataset_select_model,
                dataset_select_export, dataset_select_ai, dataset_select_insights,
                dataset_select_stats, merge_dataset_a, merge_dataset_b,
                dataset_a, dataset_b, dataset_select_dashboard, dataset_select_recommend,
                dataset_select_autopilot, dataset_select_automl, dataset_select_autofix,
                report_dataset_select, ppt_dataset_select
            ]
        )
        
        # Wire auto-clean button to refresh all dropdowns after cleaning
        auto_clean_click.then(
            fn=refresh_all_dropdowns_fn,
            outputs=[
                dataset_select_profile, dataset_select_overview, dataset_select_viz,
                dataset_select_prep, dataset_select_transform, dataset_select_model,
                dataset_select_export, dataset_select_ai, dataset_select_insights,
                dataset_select_stats, merge_dataset_a, merge_dataset_b,
                dataset_a, dataset_b, dataset_select_dashboard, dataset_select_recommend,
                dataset_select_autopilot, dataset_select_automl, dataset_select_autofix,
                report_dataset_select, ppt_dataset_select
            ]
        )
        
        # Wire model training buttons to refresh model dropdowns
        train_model_click.then(
            fn=refresh_model_dropdowns_fn,
            outputs=[model_select, model_select_fi]
        )
        
        auto_ml_click.then(
            fn=refresh_model_dropdowns_fn,
            outputs=[model_select, model_select_fi]
        )
        
        # Wire AI auto-fix button to refresh all dropdowns after fixing
        autofix_click.then(
            fn=refresh_all_dropdowns_fn,
            outputs=[
                dataset_select_profile, dataset_select_overview, dataset_select_viz,
                dataset_select_prep, dataset_select_transform, dataset_select_model,
                dataset_select_export, dataset_select_ai, dataset_select_insights,
                dataset_select_stats, merge_dataset_a, merge_dataset_b,
                dataset_a, dataset_b, dataset_select_dashboard, dataset_select_recommend,
                dataset_select_autopilot, dataset_select_automl, dataset_select_autofix,
                report_dataset_select, ppt_dataset_select
            ]
        )
        
        return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Print startup banner
    print("\n" + "="*70)
    print("""
    +------------------------------------------------------------------+
    |                                                                  |
    |            AI DATA PLATFORM 2025 - MODERN UI                     |
    |                                                                  |
    |        Smart Autopilot | One-Click ML | Advanced Analytics      |
    |                                                                  |
    +------------------------------------------------------------------+
    """)
    print("="*70)
    print("\n[*] Initializing platform components...")
    
    demo = create_interface()
    
    print("\n" + "="*70)
    print("[+] Platform ready! Starting web server...")
    print("="*70 + "\n")
    
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7864,  # Using the standard port for this UI
            share=False,
            show_error=True,
            favicon_path=None,
            quiet=False
        )
        print("\n" + "="*70)
        print("[+] Server started successfully!")
        print("[*] Open your browser and navigate to: http://127.0.0.1:7864")
        print("[*] Press Ctrl+C to stop the server")
        print("="*70 + "\n")
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("[!] Shutting down gracefully...")
        print("[!] Thank you for using AI Data Platform 2025!")
        print("="*70 + "\n")