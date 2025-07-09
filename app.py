# /mount/src/data/app.py

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
Supports both live data via yfinance and local file analysis from Capitaline exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Optional

# Import the robust backend engine for the 'Live Data' mode
from financial_engine import (
    get_stock_analysis,
    validate_symbol,
    clear_cache,
    get_cache_stats
)

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


# --- NEW: Function to parse the uploaded Capitaline Excel file ---
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Parses an uploaded Capitaline .xls file and transforms it into the
    app's expected data structure.
    
    This function assumes the Excel file has:
    - Financial metrics in the first column (e.g., 'Net Sales', 'Net Profit').
    - Years as subsequent column headers (e.g., 'Mar 2023', 'Mar 2022').
    """
    if uploaded_file is None:
        return None

    try:
        # Use the 'xlrd' engine for old .xls files
        st.info(f"Reading data from uploaded file: `{uploaded_file.name}`")
        # We assume the data is on the first sheet. If not, add sheet_name='SheetName'
        df = pd.read_excel(uploaded_file, engine='xlrd')

        # --- Data Cleaning and Structuring ---
        # 1. Identify the metric column (usually the first one) and set it as the index.
        #    We drop any rows where the metric name is missing.
        metric_col = df.columns[0]
        df = df.dropna(subset=[metric_col])
        df = df.set_index(metric_col)

        # 2. Clean up year columns: Convert 'Mar 2023' to '2023'
        #    This makes them numeric and easier to sort.
        renamed_cols = {}
        for col in df.columns:
            # Try to extract a 4-digit year from the column name
            match = pd.to_numeric(str(col), errors='coerce')
            if pd.notna(match) and 1900 < match < 2100:
                 renamed_cols[col] = str(int(match))
            elif isinstance(col, str) and " " in col:
                # Fallback for 'Mar 2023' format
                try:
                    year_str = col.split(" ")[-1]
                    if len(year_str) == 4:
                        renamed_cols[col] = year_str
                except:
                    pass # Ignore columns that don't fit the format

        df = df.rename(columns=renamed_cols)

        # 3. Filter to keep only columns that look like years
        year_columns = [col for col in df.columns if col.isdigit() and len(col) == 4]
        if not year_columns:
            st.error("Could not find year columns in the uploaded file. Please ensure years (e.g., 2023, 2022) are in the header.")
            return None
        
        df = df[sorted(year_columns, reverse=True)] # Sort from most recent to oldest
        
        # 4. Convert all data to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        st.success("File processed successfully!")
        
        # We'll return the data structured as a single statement for simplicity
        return {"capitaline_statement": df}

    except Exception as e:
        st.error(f"Error parsing the Excel file: {e}")
        st.warning("Please ensure the file is a standard financial report with metrics in the first column and years in the header.")
        return None


class DashboardUI:
    # ... (__init__, initialize_session_state, render_header are the same) ...
    
    def render_sidebar(self):
        """Render the sidebar with a data source selector."""
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            
            data_source = st.radio(
                "Select Data Source",
                ["Live Data (Yahoo Finance)", "Local File (Capitaline)"],
                key="data_source_selector",
                help="Choose a data source."
            )
            
            # --- Conditional UI based on data source ---
            if st.session_state.data_source_selector == "Local File (Capitaline)":
                st.subheader("Capitaline File Upload")
                uploaded_file = st.file_uploader(
                    "Upload .xls financial data file",
                    type=['xls']
                )
                return {"source": "capitaline", "file": uploaded_file}

            else: # Yahoo Finance is selected
                st.subheader("Stock Selection (Yahoo Finance)")
                symbol_input_from_user = st.text_input(
                    "Enter Stock Symbol:",
                    value=st.session_state.user_symbol_input,
                )
                st.session_state.user_symbol_input = symbol_input_from_user
                
                hist_period = st.selectbox("Historical Period", ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"], index=4)
                fetch_data = st.button("🔍 Fetch Live Data", use_container_width=True, type="primary")
                return {"source": "yfinance", "symbol": symbol_input_from_user, "period": hist_period, "fetch": fetch_data}

    # This method is now generalized to plot any DataFrame with the right structure
    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str], title: str):
        """Plots multiple selected rows from a DataFrame on a single chart."""
        if not selected_rows:
            return

        plot_df = df.loc[selected_rows].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)

        chart_prefs = st.session_state.get('chart_preferences', {'chart_type': 'line', 'theme': 'plotly_white'})
        chart_type = chart_prefs.get('chart_type', 'line')
        theme = chart_prefs.get('theme', 'plotly_white')

        st.markdown("---")
        st.subheader(f"📊 Chart for Selected Rows in {title}")

        if chart_type == 'bar':
            fig = px.bar(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Comparison: {', '.join(selected_rows)}", barmode='group')
        else:
            fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Trend Analysis: {', '.join(selected_rows)}", markers=True)

        fig.update_layout(xaxis_title="Period", yaxis_title="Amount", legend_title="Metrics", template=theme)
        st.plotly_chart(fig, use_container_width=True)

    def display_data_from_file(self, analysis_data):
        """Displays the parsed data from the Capitaline file."""
        st.header("Capitaline Data Analysis")
        st.info("Displaying data from the uploaded Excel file.")
        
        # The parsed data is in the 'capitaline_statement' key
        statement_df = analysis_data.get("capitaline_statement")
        
        if statement_df is not None and not statement_df.empty:
            st.markdown(f"**Select rows from the table below to visualize them.**")
            selected_rows = st.multiselect(
                f"Select rows to chart:",
                options=statement_df.index.tolist(),
                key="capitaline_multiselect"
            )

            styled_df = statement_df.style.format("{:,.2f}", na_rep="N/A")
            if selected_rows:
                styled_df = styled_df.apply(_style_selected_rows, selected_rows=selected_rows, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            self.plot_selected_rows(statement_df, selected_rows, "Uploaded Data")
        else:
            st.error("The processed data frame is empty.")

    def display_data_from_yfinance(self, analysis_data):
        """The existing display logic for Yahoo Finance data."""
        # This function contains your previous tabbed layout for yfinance
        tab1, tab2 = st.tabs(["📊 Overview", "📈 Statements & Ratios"])
        # ... (This logic is the same as your last working yfinance version) ...

    def run(self):
        """Main method to run the dashboard application."""
        self.render_header()
        
        # Get controls from the sidebar
        controls = self.render_sidebar()

        if controls["source"] == "capitaline":
            # If a file is uploaded, parse it and store it in session state
            if controls["file"]:
                analysis_data = parse_capitaline_file(controls["file"])
                if analysis_data:
                    st.session_state.analysis_data = analysis_data
                    st.session_state.active_source = "capitaline"

        elif controls["source"] == "yfinance":
            # Existing yfinance fetch logic
            symbol_input = controls["symbol"]
            needs_fetch = (controls["fetch"] or (symbol_input and symbol_input != st.session_state.current_symbol))
            if needs_fetch and symbol_input:
                is_valid, formatted_symbol = validate_symbol(symbol_input)
                if is_valid:
                    with st.spinner(f"Fetching data for {formatted_symbol}..."):
                        st.session_state.analysis_data = get_stock_analysis(formatted_symbol, period=controls["period"])
                        st.session_state.current_symbol = formatted_symbol
                        st.session_state.active_source = "yfinance"

        # --- Main Display Logic ---
        # Render content based on which data source is active
        if st.session_state.get("analysis_data"):
            if st.session_state.get("active_source") == "capitaline":
                self.display_data_from_file(st.session_state.analysis_data)
            elif st.session_state.get("active_source") == "yfinance":
                # For simplicity, we can reuse the file display logic if it's general enough,
                # or call a dedicated yfinance display function.
                # Let's assume you have a function `display_data_from_yfinance`
                 st.write("Displaying Yahoo Finance Data (placeholder)")
                 # self.display_data_from_yfinance(st.session_state.analysis_data)
            else:
                 st.info("Please select a data source and fetch or upload data.")
        else:
            st.info("Welcome! Please select a data source from the sidebar to begin.")


if __name__ == "__main__":
    ui = DashboardUI()
    # Initialize session state keys to avoid errors on first run
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "active_source" not in st.session_state:
        st.session_state.active_source = None
    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = None
    if "user_symbol_input" not in st.session_state:
        st.session_state.user_symbol_input = "AAPL"
        
    ui.run()
