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
# NEW: Import BeautifulSoup for HTML parsing
from bs4 import BeautifulSoup

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
st.markdown("""<style>/* ... your CSS ... */</style>""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    """Helper function to apply background style to selected DataFrame rows."""
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


# --- REWRITTEN: Function to parse the uploaded HTML file ---
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Parses an uploaded Capitaline file, which is an HTML table saved with a .xls extension.
    """
    if uploaded_file is None:
        return None

    try:
        st.info(f"Reading HTML data from: `{uploaded_file.name}`")
        # Read the file's content directly as a string (HTML)
        html_content = uploaded_file.getvalue().decode("utf-8")

        # Use Pandas' powerful read_html which finds all tables in an HTML string.
        # It returns a list of DataFrames. The main data table is usually the first or largest one.
        tables = pd.read_html(html_content)
        
        if not tables:
            st.error("No data tables were found in the uploaded file.")
            return None

        # Assume the largest table is the one we want
        df = max(tables, key=len)

        # --- Data Cleaning and Structuring (same logic as before) ---
        # 1. The first column is usually the metric, and it becomes the header after read_html.
        #    We need to find it and set it as the index.
        df = df.rename(columns={0: 'Metric'}).dropna(subset=['Metric']).set_index('Metric')

        # 2. The column headers are now the first row of data. Promote them.
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        df = df.reset_index() # Reset index to work with the 'Metric' column again
        
        # 3. Clean up the now-promoted header (years)
        df = df.rename(columns={'index': 'Metric'}).set_index('Metric')
        
        # 4. Clean column names (e.g., 'Mar. 23' -> '2023')
        renamed_cols = {}
        for col in df.columns:
            year_str = ''.join(filter(str.isdigit, str(col)))
            if len(year_str) >= 2: # Look for '23' or '2023'
                renamed_cols[col] = "20" + year_str[-2:] if len(year_str) < 4 else year_str

        df = df.rename(columns=renamed_cols)
        
        # 5. Filter to keep only columns that look like years and sort them
        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4], reverse=True)
        if not year_columns:
            st.error("Could not find year columns in the file's header after parsing.")
            return None
        
        df_final = df[year_columns].apply(pd.to_numeric, errors='coerce')
        
        st.success("File parsed successfully!")
        return {"capitaline_statement": df_final}

    except Exception as e:
        st.error(f"Error parsing the file: {e}")
        st.warning("Please ensure the uploaded file is the unmodified export from Capitaline.")
        return None


# The DashboardUI class and its methods remain the same as the previous version.
# The only change needed was in the parsing function above.
class DashboardUI:
    """The main class for the Streamlit user interface."""

    def __init__(self):
        """Initializes the UI class and sets up the session state."""
        if "analysis_data" not in st.session_state:
            st.session_state.analysis_data = None
        if "active_source" not in st.session_state:
            st.session_state.active_source = "capitaline"
        if "user_symbol_input" not in st.session_state:
            st.session_state.user_symbol_input = "RELIANCE.NS"
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
            
            data_source = st.radio(
                "Select Data Source",
                ["Local File (Capitaline)", "Live Data (Yahoo Finance)"],
                key="data_source_selector",
                on_change=lambda: st.session_state.update(analysis_data=None)
            )
            
            if data_source == "Local File (Capitaline)":
                uploaded_file = st.file_uploader("Upload .xls financial data file", type=['xls'])
                return {"source": "capitaline", "file": uploaded_file}
            
            else:
                if not get_stock_analysis:
                    st.error("Live Data mode is disabled.")
                    return {"source": "yfinance", "fetch": False, "symbol": ""}
                
                st.subheader("Stock Selection")
                symbol = st.text_input("Enter Stock Symbol (e.g., INFY.NS)", value=st.session_state.user_symbol_input)
                st.session_state.user_symbol_input = symbol
                
                fetch = st.button("🔍 Fetch Live Data", use_container_width=True, type="primary")
                return {"source": "yfinance", "fetch": fetch, "symbol": symbol}

    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str]):
        """Plots multiple selected rows from a DataFrame on a single chart."""
        if not selected_rows:
            return

        plot_df = df.loc[selected_rows].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        st.markdown("---")
        st.subheader(f"📊 Chart for: {', '.join(selected_rows)}")
        fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Trend Analysis", markers=True)
        fig.update_layout(xaxis_title="Period", yaxis_title="Amount (in Rs. Cr.)", legend_title="Metrics")
        st.plotly_chart(fig, use_container_width=True)

    def display_capitaline_data(self, analysis_data: Dict[str, pd.DataFrame]):
        """Renders the UI for the parsed Capitaline data."""
        st.header("Capitaline Data Analysis")
        st.info("Displaying data from the uploaded file. Select rows to visualize.")
        
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
            self.plot_selected_rows(statement_df, selected_rows)

    def display_yfinance_data(self, analysis_data: Dict[str, Any]):
        """Renders the UI for data fetched from Yahoo Finance."""
        st.header(f"Live Data Analysis: {st.session_state.user_symbol_input}")
        st.json(analysis_data, expanded=False)
        st.warning("Display for live data is simplified.")

    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        if controls["source"] == "capitaline" and controls["file"]:
            if controls["file"] != st.session_state._uploaded_file_memo:
                st.session_state._uploaded_file_memo = controls["file"]
                st.session_state.analysis_data = parse_capitaline_file(controls["file"])
                st.session_state.active_source = "capitaline"
        
        elif controls["source"] == "yfinance" and controls["fetch"]:
            if controls["symbol"] and validate_symbol:
                is_valid, formatted_symbol = validate_symbol(controls["symbol"])
                if is_valid:
                    with st.spinner(f"Fetching data for {formatted_symbol}..."):
                        st.session_state.analysis_data = get_stock_analysis(formatted_symbol)
                        st.session_state.active_source = "yfinance"
                else:
                    st.error(f"Invalid symbol format: {controls['symbol']}")

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
