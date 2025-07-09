# /mount/src/capitaline/app.py - FINAL VERSION

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
Supports both live data via yfinance and local file analysis from Capitaline exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Optional
import io # Used to read the file content

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""<style> .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; } </style>""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    """Helper function to apply background style to selected DataFrame rows."""
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


# --- FINAL PARSER: Tailored to the diagnosed MultiIndex structure ---
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Parses an uploaded Capitaline file (HTML with a MultiIndex header).
    """
    if uploaded_file is None:
        return None

    try:
        st.info(f"Reading HTML data from: `{uploaded_file.name}`")
        html_content = uploaded_file.getvalue()
        
        # Use read_html with header=[0,1] to correctly parse the MultiIndex
        df = pd.read_html(io.BytesIO(html_content), header=[0, 1])[0]

        # --- Data Cleaning for MultiIndex ---
        
        # 1. Extract Metadata from the header
        company_info_tuple = df.columns[0][0]
        try:
            company_name = company_info_tuple.split(">>")[2].split("(")[0]
        except IndexError:
            company_name = "Company"

        # 2. Flatten the MultiIndex header into a single level
        new_cols = []
        for col_level0, col_level1 in df.columns:
            # The year is in the second level of the header
            new_cols.append(str(col_level1))
        
        df.columns = new_cols
        
        # 3. The first column now contains the metric names. Rename it and set as index.
        metric_col_name = df.columns[0]
        df = df.rename(columns={metric_col_name: "Metric"})
        df = df.dropna(subset=['Metric']).set_index('Metric')

        # 4. Clean up the year column names (e.g., '201103' -> '2011')
        renamed_cols = {}
        for col in df.columns:
            year_str = str(col)
            if year_str.isdigit() and len(year_str) == 6:
                renamed_cols[col] = year_str[:4]
            elif year_str.isdigit() and len(year_str) == 4:
                renamed_cols[col] = year_str

        df = df.rename(columns=renamed_cols)

        # 5. Filter to keep only valid 4-digit year columns and sort them
        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4], reverse=True)
        if not year_columns:
            st.error("Could not find valid 4-digit year columns after cleaning.")
            return None
        
        df_final = df[year_columns].apply(pd.to_numeric, errors='coerce').dropna(how='all')
        
        st.success("File parsed successfully!")
        return {"statement": df_final, "company_name": company_name}

    except Exception as e:
        st.error(f"An unexpected error occurred during parsing: {e}")
        st.warning("If the issue persists, the file's HTML structure may be different from the one diagnosed.")
        return None

# The DashboardUI class and its methods are now stable.
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
        st.markdown('<div class="main-header">📈 Financial Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

    def render_sidebar(self):
        """Renders the sidebar controls and returns the user's selections."""
        with st.sidebar:
            st.header("🎛️ Controls")
            st.info("Upload a Capitaline .xls file to begin analysis.")
            uploaded_file = st.file_uploader("Upload financial data file", type=['xls'])
            return {"file": uploaded_file}

    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str]):
        """Plots multiple selected rows from a DataFrame on a single chart."""
        if not selected_rows:
            return

        plot_df = df.loc[selected_rows].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        st.markdown("---")
        st.subheader(f"📊 Chart for: {', '.join(selected_rows)}")
        fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title="Trend Analysis", markers=True)
        fig.update_layout(xaxis_title="Year", yaxis_title="Amount (in Rs. Cr.)", legend_title="Metrics")
        st.plotly_chart(fig, use_container_width=True)

    def display_capitaline_data(self, analysis_data: Dict[str, Any]):
        """Renders the UI for the parsed Capitaline data."""
        company_name = analysis_data.get("company_name", "Uploaded Data")
        statement_df = analysis_data.get("statement")
        
        st.header(f"Analysis for: {company_name}")
        st.info("Select one or more rows from the table below to visualize their trends.")
        
        if statement_df is not None and not statement_df.empty:
            selected_rows = st.multiselect(
                "Select metrics to chart:",
                options=statement_df.index.tolist(),
                key="capitaline_multiselect"
            )
            styled_df = statement_df.style.format("{:,.2f}", na_rep="-")
            if selected_rows:
                styled_df = styled_df.apply(_style_selected_rows, selected_rows=selected_rows, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            self.plot_selected_rows(statement_df, selected_rows)

    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        if controls["file"]:
            if controls["file"] != st.session_state._uploaded_file_memo:
                st.session_state._uploaded_file_memo = controls["file"]
                st.session_state.analysis_data = parse_capitaline_file(controls["file"])
        
        if st.session_state.analysis_data:
            self.display_capitaline_data(st.session_state.analysis_data)
        else:
            st.info("Welcome! Please upload a Capitaline .xls file using the sidebar to begin your analysis.")

if __name__ == "__main__":
    ui = DashboardUI()
    ui.run()
