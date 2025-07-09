# /mount/src/capitaline/app.py - FINAL VERSION with AG-Grid Interactivity

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
Features interactive row selection for charting using AG-Grid.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Optional
import io # Used to read the file content

# NEW: Import the AG-Grid component
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# Configure Streamlit page
st.set_page_config(
    page_title="Interactive Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""<style> .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; } </style>""", unsafe_allow_html=True)


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

    def render_header(self):
        """Renders the main title header for the application."""
        st.markdown('<div class="main-header">📊 Interactive Financial Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

    def render_sidebar(self):
        """Renders the sidebar controls for file uploading."""
        with st.sidebar:
            st.header("🎛️ Controls")
            st.info("Upload a Capitaline .xls file to begin your analysis.")
            uploaded_file = st.file_uploader("Upload financial data file", type=['xls'])
            return {"file": uploaded_file}

    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[Dict], chart_type: str):
        """Plots multiple selected rows from a DataFrame on a single chart."""
        if not selected_rows:
            return

        # Extract the metric names from the list of selected row dictionaries
        selected_metrics = [row['Metric'] for row in selected_rows]
        
        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        st.markdown("---")
        st.subheader(f"Visual Analysis for: {', '.join(selected_metrics)}")

        if chart_type == 'Bar Chart':
            fig = px.bar(plot_df, x=plot_df.index, y=plot_df.columns, title="Comparison by Year", barmode='group')
        else: # Default to Line Chart
            fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title="Trend Analysis", markers=True)

        fig.update_layout(xaxis_title="Year", yaxis_title="Amount (in Rs. Cr.)", legend_title="Metrics")
        st.plotly_chart(fig, use_container_width=True)

    # --- REWRITTEN: This function now uses AG-Grid instead of st.dataframe ---
    def display_capitaline_data(self, analysis_data: Dict[str, Any]):
        """Renders the UI for the parsed Capitaline data using an interactive AG-Grid."""
        company_name = analysis_data.get("company_name", "Uploaded Data")
        statement_df = analysis_data.get("statement")
        
        st.header(f"Analysis for: {company_name}")
        st.info("Click on rows in the table below to select them for charting. Use Ctrl/Cmd to select multiple rows.")
        
        if statement_df is not None and not statement_df.empty:
            
            # Reset index to make the 'Metric' column available for selection
            grid_df = statement_df.reset_index()

            # --- AG-Grid Configuration ---
            gb = GridOptionsBuilder.from_dataframe(grid_df)
            
            # Configure the 'Metric' column to be wider and pinned
            gb.configure_column("Metric", headerName="Financial Metric", width=250, pinned='left')
            
            # Configure selection
            gb.configure_selection(
                'multiple',
                use_checkbox=True,
                groupSelectsChildren=True,
                header_checkbox=True
            )
            
            # Make other columns editable if needed, or set other properties
            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()

            # --- Display the Grid ---
            grid_response = AgGrid(
                grid_df,
                gridOptions=gridOptions,
                height=400,
                width='100%',
                data_return_mode=DataReturnMode.AS_INPUT,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True, # Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=False # Disable enterprise features
            )

            # --- Capture Selected Data and Controls ---
            selected_rows = grid_response['selected_rows']
            
            chart_type = "Line Chart" # Default
            if selected_rows:
                 chart_type = st.radio(
                    "Select Chart Type:",
                    ["Line Chart", "Bar Chart"],
                    key="chart_type_selector",
                    horizontal=True
                )
            
            # Pass the selected data to the plotting function
            self.plot_selected_rows(statement_df, selected_rows, chart_type)

    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        if controls["file"]:
            if controls["file"] != st.session_state._uploaded_file_memo:
                with st.spinner("Processing file..."):
                    st.session_state._uploaded_file_memo = controls["file"]
                    st.session_state.analysis_data = parse_capitaline_file(controls["file"])
        
        if st.session_state.analysis_data:
            self.display_capitaline_data(st.session_state.analysis_data)
        else:
            st.info("Welcome! Please upload a Capitaline .xls file using the sidebar to begin your analysis.")

if __name__ == "__main__":
    ui = DashboardUI()
    ui.run()
