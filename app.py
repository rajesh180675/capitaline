# /mount/src/capitaline/app.py - FINAL HIGH-PERFORMANCE VERSION

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
Features a high-performance "Select and Chart" workflow using native components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Optional
import io

# Configure Streamlit page
st.set_page_config(
    page_title="High-Performance Financial Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    /* Style the primary button to be more prominent */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)


def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Parses an uploaded Capitaline file (HTML with a MultiIndex header).
    """
    if uploaded_file is None:
        return None

    try:
        html_content = uploaded_file.getvalue()
        df = pd.read_html(io.BytesIO(html_content), header=[0, 1])[0]
        
        company_info_tuple = df.columns[0][0]
        try:
            company_name = company_info_tuple.split(">>")[2].split("(")[0]
        except IndexError:
            company_name = "Company"

        new_cols = [str(col_level1) for _, col_level1 in df.columns]
        df.columns = new_cols
        
        metric_col_name = df.columns[0]
        df = df.rename(columns={metric_col_name: "Metric"}).dropna(subset=['Metric']).set_index('Metric')

        renamed_cols = {}
        for col in df.columns:
            year_str = str(col)
            if year_str.isdigit() and len(year_str) == 6:
                renamed_cols[col] = year_str[:4]
            elif year_str.isdigit() and len(year_str) == 4:
                renamed_cols[col] = year_str
        df = df.rename(columns=renamed_cols)

        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4], reverse=True)
        if not year_columns:
            st.error("Could not find valid 4-digit year columns after cleaning.")
            return None
        
        df_final = df[year_columns].apply(pd.to_numeric, errors='coerce').dropna(how='all')
        
        return {"statement": df_final, "company_name": company_name}

    except Exception as e:
        st.error(f"An unexpected error occurred during parsing: {e}")
        return None


class DashboardUI:
    """The main class for the Streamlit user interface."""

    def __init__(self):
        """Initializes the UI class and sets up the session state."""
        if "analysis_data" not in st.session_state:
            st.session_state.analysis_data = None
        if "_uploaded_file_memo" not in st.session_state:
            st.session_state._uploaded_file_memo = None
        # NEW: State to hold the generated chart figure
        if "chart_figure" not in st.session_state:
            st.session_state.chart_figure = None

    def render_header(self):
        """Renders the main title header for the application."""
        st.markdown('<div class="main-header">⚡ High-Performance Financial Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

    def render_sidebar(self):
        """Renders the sidebar controls for file uploading."""
        with st.sidebar:
            st.header("🎛️ Controls")
            st.info("Upload a Capitaline .xls file to begin your analysis.")
            uploaded_file = st.file_uploader("Upload financial data file", type=['xls'])
            return {"file": uploaded_file}

    def generate_chart(self, df: pd.DataFrame, selected_metrics: List[str], chart_type: str):
        """Generates and returns a Plotly figure object."""
        if not selected_metrics:
            return None

        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        title = f"Analysis for: {', '.join(selected_metrics)}"
        
        if chart_type == 'Bar Chart':
            fig = px.bar(plot_df, x=plot_df.index, y=plot_df.columns, title=title, barmode='group')
        else:
            fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=title, markers=True)

        fig.update_layout(xaxis_title="Year", yaxis_title="Amount (in Rs. Cr.)", legend_title="Metrics")
        return fig

    # --- REWRITTEN: This function now uses the high-performance st.data_editor ---
    def display_capitaline_data(self, analysis_data: Dict[str, Any]):
        """Renders the UI for the parsed Capitaline data using st.data_editor."""
        company_name = analysis_data.get("company_name", "Uploaded Data")
        statement_df = analysis_data.get("statement")
        
        st.header(f"Analysis for: {company_name}")
        st.info("Use the checkboxes to select rows, then click the 'Generate Chart' button.")
        
        if statement_df is not None and not statement_df.empty:
            # Prepare DataFrame for the data editor
            df_to_edit = statement_df.reset_index()
            
            # --- Display the Data Editor ---
            edited_df = st.data_editor(
                df_to_edit,
                key="data_editor",
                # Add a column configuration to make the 'Metric' column wider
                column_config={"Metric": st.column_config.TextColumn(width="large")},
                # Disable editing for all columns except the selection
                disabled=df_to_edit.columns, 
                hide_index=True
            )
            
            # Find which rows were selected by the user
            selected_rows = edited_df[edited_df.select == True]["Metric"].tolist()

            st.markdown("---")
            
            # --- Charting Controls ---
            if selected_rows:
                col1, col2 = st.columns([1, 3])
                with col1:
                    chart_type = st.radio("Select Chart Type", ["Line Chart", "Bar Chart"], horizontal=True)
                with col2:
                    # The button that triggers the chart generation
                    if st.button("📊 Generate Chart", type="primary", use_container_width=True):
                        # Generate the figure and store it in session state
                        fig = self.generate_chart(statement_df, selected_rows, chart_type)
                        st.session_state.chart_figure = fig
            else:
                # Clear the previous chart if no rows are selected
                st.session_state.chart_figure = None

        # --- Display the Chart ---
        # The chart is displayed here, outside the button's "if" block.
        # This ensures it persists across reruns until a new chart is generated.
        if st.session_state.chart_figure:
            st.plotly_chart(st.session_state.chart_figure, use_container_width=True)
        elif selected_rows:
            st.info("Click the 'Generate Chart' button to visualize your selection.")

    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        if controls["file"]:
            if controls["file"] != st.session_state._uploaded_file_memo:
                with st.spinner("Processing file..."):
                    st.session_state._uploaded_file_memo = controls["file"]
                    st.session_state.analysis_data = parse_capitaline_file(controls["file"])
                    # Clear old chart when a new file is uploaded
                    st.session_state.chart_figure = None
        
        if st.session_state.analysis_data:
            self.display_capitaline_data(st.session_state.analysis_data)
        else:
            st.info("Welcome! Please upload a Capitaline .xls file using the sidebar to begin your analysis.")

if __name__ == "__main__":
    ui = DashboardUI()
    ui.run()
