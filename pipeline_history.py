
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class PipelineHistory:
    def __init__(self):
        self.history_file = "pipeline_history.json"

    def log_step(self, operation_name, description, parameters=None, status="success"):
        """Log a pipeline step to history"""
        step = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "description": description,
            "parameters": parameters or {},
            "status": status
        }

        if 'pipeline_history' not in st.session_state:
            st.session_state.pipeline_history = []

        st.session_state.pipeline_history.append(step)

        # Also save to file
        self._save_to_file()

    def _save_to_file(self):
        """Save history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(st.session_state.pipeline_history, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save history to file: {str(e)}")

    def load_from_file(self):
        """Load history from JSON file"""
        try:
            with open(self.history_file, 'r') as f:
                st.session_state.pipeline_history = json.load(f)
                return True
        except FileNotFoundError:
            return False
        except Exception as e:
            st.error(f"Failed to load history from file: {str(e)}")
            return False

    def display_history(self):
        """Display pipeline history in UI"""
        if not st.session_state.pipeline_history:
            st.info("No pipeline history available.")
            return

        # Convert to DataFrame for display
        history_df = pd.DataFrame(st.session_state.pipeline_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        # Display options
        display_mode = st.radio(
            "Display Mode",
            ["Table View", "Timeline View", "Summary Stats"],
            horizontal=True
        )

        if display_mode == "Table View":
            # Use st.table instead of st.dataframe to avoid React error #185 in sidebar
            st.table(
                history_df[['timestamp', 'operation', 'description', 'status']].head(100)  # Limit rows to prevent performance issues
            )

        elif display_mode == "Timeline View":
            # Create timeline chart
            fig = go.Figure()

            # Color mapping for status
            color_map = {'success': 'green', 'error': 'red', 'warning': 'orange', 'info': 'blue'}

            for i, step in enumerate(st.session_state.pipeline_history):
                color = color_map.get(step['status'], 'gray')

                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(step['timestamp'])],
                    y=[i],
                    mode='markers+text',
                    marker=dict(size=12, color=color),
                    text=step['operation'],
                    textposition='middle right',
                    name=step['operation'],
                    hovertemplate=(
                        f"<b>{step['operation']}</b><br>"
                        f"{step['description']}<br>"
                        f"Status: {step['status']}<br>"
                        f"Time: {step['timestamp']}<extra></extra>"
                    )
                ))

            fig.update_layout(
                title="Pipeline Timeline",
                xaxis_title="Timestamp",
                yaxis_title="Step Number",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        else:  # Summary Stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Steps", len(st.session_state.pipeline_history))

            with col2:
                success_count = sum(1 for step in st.session_state.pipeline_history if step['status'] == 'success')
                st.metric("Successful", success_count)

            with col3:
                error_count = sum(1 for step in st.session_state.pipeline_history if step['status'] == 'error')
                st.metric("Errors", error_count)

            with col4:
                if st.session_state.pipeline_history:
                    start_time = pd.to_datetime(st.session_state.pipeline_history[0]['timestamp'])
                    end_time = pd.to_datetime(st.session_state.pipeline_history[-1]['timestamp'])
                    duration = (end_time - start_time).total_seconds()
                    st.metric("Duration", f"{duration:.1f}s")

            # Operation frequency chart
            operations = [step['operation'] for step in st.session_state.pipeline_history]
            operation_counts = pd.Series(operations).value_counts()

            fig = px.bar(
                x=operation_counts.values,
                y=operation_counts.index,
                orientation='h',
                title="Operations Frequency"
            )
            fig.update_layout(xaxis_title="Count", yaxis_title="Operation")
            st.plotly_chart(fig, use_container_width=True)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download History (JSON)"):
                history_json = json.dumps(st.session_state.pipeline_history, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"pipeline_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📊 Download History (CSV)"):
                history_csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=history_csv,
                    file_name=f"pipeline_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Clear history option
        if st.button("🧹 Clear History", key="clear_history_pipeline"):
            st.session_state.pipeline_history = []
            # Remove st.rerun() to prevent infinite loop
            # st.rerun()

    def get_history_summary(self):
        """Get summary of pipeline history for reporting"""
        if not st.session_state.pipeline_history:
            return {}

        history_df = pd.DataFrame(st.session_state.pipeline_history)

        summary = {
            "total_steps": len(st.session_state.pipeline_history),
            "successful_steps": len(history_df[history_df['status'] == 'success']),
            "failed_steps": len(history_df[history_df['status'] == 'error']),
            "operations": list(history_df['operation'].unique()),
            "start_time": st.session_state.pipeline_history[0]['timestamp'],
            "end_time": st.session_state.pipeline_history[-1]['timestamp'],
            "steps": st.session_state.pipeline_history
        }

        return summary

    def clear_history(self):
        """Clear pipeline history"""
        st.session_state.pipeline_history = []
        try:
            import os
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
        except Exception as e:
            st.error(f"Failed to remove history file: {str(e)}")

    def filter_history(self, operation=None, status=None, start_date=None, end_date=None):
        """Filter history based on criteria"""
        filtered_history = st.session_state.pipeline_history.copy()

        if operation:
            filtered_history = [step for step in filtered_history if step['operation'] == operation]

        if status:
            filtered_history = [step for step in filtered_history if step['status'] == status]

        if start_date:
            filtered_history = [step for step in filtered_history
                              if pd.to_datetime(step['timestamp']) >= start_date]

        if end_date:
            filtered_history = [step for step in filtered_history
                              if pd.to_datetime(step['timestamp']) <= end_date]

        return filtered_history
