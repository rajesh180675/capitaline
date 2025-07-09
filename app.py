# /mount/src/capitaline/app.py

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
Supports both live data via yfinance and local file analysis from Capitaline exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Optional

# Try to import backend; handle error gracefully if it's missing
try:
    from financial_engine import get_stock_analysis, validate_symbol
except ImportError:
    st.error("Warning: `financial_engine.py` not found. Live Data mode will not be available.")
    get_stock_analysis = None
    validate_symbol = None

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center;
        margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    """Helper function to apply background style to selected DataFrame rows."""
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Parses an uploaded Capitaline .xls file and transforms it into the
    app's expected data structure.
    """
    if uploaded_file is None:
        return None

    try:
        st.info(f"Reading data from: `{uploaded_file.name}`")
        df = pd.read_excel(uploaded_file, engine='xlrd')

        metric_col = df.columns[0]
        df = df.dropna(subset=[metric_col]).set_index(metric_col)

        renamed_cols = {}
        for col in df.columns:
            try:
                # Extracts the last 4-digit number from a string like 'Mar 2023'
                year_str = ''.join(filter(str.isdigit, str(col)))[-4:]
                if year_str:
                    renamed_cols[col] = year_str
            except IndexError:
                continue
        
        df = df.rename(columns=renamed_cols)
        
        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4], reverse=True)
        if not year_columns:
            st.error("Could not find year columns (e.g., 2023, 2022) in the file's header.")
            return None
        
        df = df[year_columns].apply(pd.to_numeric, errors='coerce')
        
        st.success("File processed successfully!")
        return {"capitaline_statement": df}

    except Exception as e:
        st.error(f"Error parsing Excel file: {e}")
        return None


# IMPORTANT: All methods below must be correctly indented to be part of the class.
class DashboardUI:
    """The main class for the Streamlit user interface."""

    def __init__(self):
        """Initializes the UI class and sets up the session state."""
        # Initialize session state keys to avoid errors on the first run
        if "analysis_data" not in st.session_state:
            st.session_state.analysis_data = None
        if "active_source" not in st.session_state:
            st.session_state.active_source = "capitaline" # Default to file upload
        if "user_symbol_input" not in st.session_state:
            st.session_state.user_symbol_input = "RELIANCE.NS"

    def render_header(self):
        """Renders the main title header for the application."""
        st.markdown('<div class="main-header">📈 Financial Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

    def render_sidebar(self):
        """Renders the sidebar controls and returns the user's selections."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            data_source = st.radio(
                "Select Data Source",
                ["Local File (Capitaline)", "Live Data (Yahoo Finance)"],
                key="data_source_selector",
                on_change=lambda: st.session_state.update(analysis_data=None) # Clear data when switching source
            )
            
            # --- Return a dictionary of controls based on the selected source ---
            if data_source == "Local File (Capitaline)":
                st.subheader("Capitaline File Upload")
                uploaded_file = st.file_uploader("Upload .xls financial data file", type=['xls'])
                return {"source": "capitaline", "file": uploaded_file}
            
            else: # Yahoo Finance is selected
                if not get_stock_analysis: # Check if import failed
                    st.error("Live Data mode is disabled because `financial_engine.py` was not found.")
                    return {"source": "yfinance", "fetch": False, "symbol": ""}
                
                st.subheader("Stock Selection")
                symbol = st.text_input("Enter Stock Symbol (e.g., INFY.NS)", value=st.session_state.user_symbol_input)
                st.session_state.user_symbol_input = symbol # Remember the last typed symbol
                
                fetch = st.button("🔍 Fetch Live Data", use_container_width=True, type="primary")
                return {"source": "yfinance", "fetch": fetch, "symbol": symbol}

    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str], title: str):
        """Plots multiple selected rows from a DataFrame on a single chart."""
        if not selected_rows:
            return

        plot_df = df.loc[selected_rows].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        st.markdown("---")
        st.subheader(f"📊 Chart for: {', '.join(selected_rows)}")
        fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Trend Analysis", markers=True)
        fig.update_layout(xaxis_title="Period", yaxis_title="Amount", legend_title="Metrics")
        st.plotly_chart(fig, use_container_width=True)

    def display_capitaline_data(self, analysis_data: Dict[str, pd.DataFrame]):
        """Renders the UI for the parsed Capitaline data."""
        st.header("Capitaline Data Analysis")
        st.info("Displaying data from the uploaded Excel file. Select rows to visualize.")
        
        statement_df = analysis_data.get("capitaline_statement")
        
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
            self.plot_selected_rows(statement_df, selected_rows, "Uploaded Data")

    def display_yfinance_data(self, analysis_data: Dict[str, Any]):
        """Renders the UI for data fetched from Yahoo Finance."""
        st.header(f"Live Data Analysis: {st.session_state.user_symbol_input}")
        # This is where you would build out the more complex tabbed view for yfinance data
        # For now, we'll keep it simple and show the raw data.
        st.json(analysis_data, expanded=False)
        st.warning("Display for live data is simplified. You can build out the tabbed interface here.")

    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        # --- Data Loading Logic ---
        if controls["source"] == "capitaline" and controls["file"]:
            # If a new file is uploaded, parse it.
            if controls["file"] != st.session_state.get("_uploaded_file_memo"):
                st.session_state._uploaded_file_memo = controls["file"]
                st.session_state.analysis_data = parse_capitaline_file(controls["file"])
                st.session_state.active_source = "capitaline"
        
        elif controls["source"] == "yfinance" and controls["fetch"]:
            # If fetch button is clicked for a valid symbol
            if controls["symbol"] and validate_symbol:
                is_valid, formatted_symbol = validate_symbol(controls["symbol"])
                if is_valid:
                    with st.spinner(f"Fetching data for {formatted_symbol}..."):
                        st.session_state.analysis_data = get_stock_analysis(formatted_symbol)
                        st.session_state.active_source = "yfinance"
                else:
                    st.error(f"Invalid symbol format: {controls['symbol']}")

        # --- Data Display Logic ---
        if st.session_state.analysis_data:
            if st.session_state.active_source == "capitaline":
                self.display_capitaline_data(st.session_state.analysis_data)
            elif st.session_state.active_source == "yfinance":
                self.display_yfinance_data(st.session_state.analysis_data)
        else:
            st.info("Welcome! Please select a data source from the sidebar to begin.")


if __name__ == "__main__":
    ui = DashboardUI()
    ui.run()
