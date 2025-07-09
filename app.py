# /mount/src/capitaline/app.py - COMPLETED ENHANCED VERSION

"""
Enhanced Financial Dashboard - Improved Version
A robust Streamlit application for financial data analysis with enhanced error handling,
performance optimization, and additional features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import io
import logging
from datetime import datetime
import re
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .data-quality-indicator {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
    }
    .quality-high { background-color: #28a745; }
    .quality-medium { background-color: #ffc107; }
    .quality-low { background-color: #dc3545; }
    .welcome-container {
        text-align: center; padding: 3rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px; margin: 2rem 0;
    }
    .feature-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class DataQualityMetrics:
    """Data class to store data quality metrics."""
    total_rows: int
    missing_values: int
    missing_percentage: float
    duplicate_rows: int
    quality_score: str = field(init=False)
    
    def __post_init__(self):
        if self.missing_percentage < 5:
            self.quality_score = "High"
        elif self.missing_percentage < 20:
            self.quality_score = "Medium"
        else:
            self.quality_score = "Low"

class FileValidator:
    """Validates uploaded files and their content."""
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        if uploaded_file is None:
            return False, "No file uploaded"
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File size exceeds 10MB limit"
        allowed_types = ['xls', 'xlsx', 'html', 'htm']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"Unsupported file type: {file_extension}"
        return True, file_extension

class DataProcessor:
    """Handles data processing and cleaning operations."""
    
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[str]]:
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        return outliers
    
    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total_cells = df.size
        if total_cells == 0:
            return DataQualityMetrics(0, 0, 0, 0)
        missing_values = df.isnull().sum().sum()
        return DataQualityMetrics(
            total_rows=len(df),
            missing_values=missing_values,
            missing_percentage=(missing_values / total_cells) * 100,
            duplicate_rows=df.duplicated().sum()
        )

@st.cache_data(show_spinner=False)
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Enhanced parser for Capitaline files with better error handling and validation."""
    if uploaded_file is None: return None
    is_valid, file_extension = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(f"File validation failed: {file_extension}")
        return None

    try:
        file_content = uploaded_file.getvalue()
        df = pd.read_html(io.BytesIO(file_content), header=[0, 1])[0]

        company_name = "Unknown Company"
        try:
            company_info_tuple = str(df.columns[0][0])
            if ">>" in company_info_tuple:
                company_name = company_info_tuple.split(">>")[2].split("(")[0].strip()
        except (IndexError, AttributeError): pass

        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=['Metric']).set_index('Metric')
        
        renamed_cols = {}
        for col in df.columns:
            year = ''.join(filter(str.isdigit, str(col)))
            if len(year) >= 4:
                renamed_cols[col] = year[:4]
        df = df.rename(columns=renamed_cols)

        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4 and '1990' < col < '2050'], reverse=True)
        if not year_columns:
            st.error("Could not find valid year columns in the data.")
            return None
        
        df_final = df[year_columns].copy()
        df_final = DataProcessor.clean_numeric_data(df_final).dropna(how='all')
        
        return {
            "statement": df_final, "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(df_final),
            "outliers": DataProcessor.detect_outliers(df_final),
            "year_columns": year_columns,
            "file_info": {"name": uploaded_file.name, "size": uploaded_file.size}
        }
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"An error occurred while parsing: {e}")
        return None

class ChartGenerator:
    """Enhanced chart generation with multiple chart types and customization options."""
    @staticmethod
    def _create_base_figure(title: str, chart_theme: str, show_grid: bool) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=20), x=0.5),
            xaxis_title="Year", yaxis_title="Amount (in Rs. Cr.)",
            template=chart_theme, height=500, legend_title_text='Metrics',
            xaxis=dict(showgrid=show_grid), yaxis=dict(showgrid=show_grid)
        )
        return fig

    @staticmethod
    def create_line_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        for metric in selected_metrics:
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines+markers', name=metric))
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        fig.update_layout(barmode='group')
        for metric in selected_metrics:
            fig.add_trace(go.Bar(x=df.columns, y=df.loc[metric], name=metric))
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        for metric in selected_metrics:
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines', name=metric, stackgroup='one'))
        return fig

    @staticmethod
    def create_heatmap(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        corr_matrix = df.loc[selected_metrics].T.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1,1])
        fig.update_layout(title=f"Correlation Matrix: {title}", template=theme, height=500)
        return fig

class DashboardUI:
    """Enhanced UI class with improved functionality and user experience."""
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        for var, default in {"analysis_data": None, "_uploaded_file_memo": None, "chart_figure": None}.items():
            if var not in st.session_state: st.session_state[var] = default

    def render_header(self):
        st.markdown('<div class="main-header">📊 Advanced Financial Dashboard</div>', unsafe_allow_html=True)
        if st.session_state.analysis_data:
            col1, col2 = st.columns([2,1])
            col1.markdown(f"**Company:** {st.session_state.analysis_data.get('company_name', 'N/A')}")
            col2.markdown(f"**Source File:** `{st.session_state.analysis_data['file_info']['name']}`")
        st.markdown("---")

    def render_sidebar(self) -> Dict[str, Any]:
        with st.sidebar:
            st.header("🎛️ Controls")
            st.subheader("📁 File Upload")
            uploaded_file = st.file_uploader("Upload a Capitaline file", type=['xls', 'xlsx', 'html', 'htm'])
            
            controls = {"file": uploaded_file}
            if st.session_state.analysis_data:
                st.subheader("📊 Display Options")
                controls["show_data_quality"] = st.checkbox("Show Data Quality Metrics")
                controls["show_outliers"] = st.checkbox("Show Outlier Detection")
                st.subheader("🎨 Chart Settings")
                controls["chart_theme"] = st.selectbox("Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])
                controls["show_grid"] = st.checkbox("Show Grid Lines", True)
            return controls

    def display_data_quality_metrics(self, data_quality: DataQualityMetrics):
        st.subheader("🔍 Data Quality Assessment")
        cols = st.columns(4)
        cols[0].metric("Total Rows", data_quality.total_rows)
        cols[1].metric("Missing Values", data_quality.missing_values)
        cols[2].metric("Missing %", f"{data_quality.missing_percentage:.1f}%")
        color = {"High": "green", "Medium": "orange", "Low": "red"}[data_quality.quality_score]
        cols[3].markdown(f"**Quality Score:** :{color}[{data_quality.quality_score}]")

    def display_outliers(self, outliers: Dict[str, List[str]]):
        st.subheader("⚠️ Outlier Detection")
        if not outliers:
            st.success("No significant outliers detected based on the IQR method.")
            return
        for year, outlier_metrics in outliers.items():
            with st.expander(f"Potential Outliers in **{year}**"):
                for metric in outlier_metrics: st.write(f"• {metric}")
    
    def generate_chart(self, df: pd.DataFrame, selected_metrics: List[str], chart_type: str, theme: str, grid: bool) -> Optional[go.Figure]:
        if not selected_metrics: return None
        title = f"Analysis for {st.session_state.analysis_data.get('company_name')}"
        chart_generators = {'Line Chart': ChartGenerator.create_line_chart, 'Bar Chart': ChartGenerator.create_bar_chart, 'Area Chart': ChartGenerator.create_area_chart, 'Heatmap': ChartGenerator.create_heatmap}
        return chart_generators.get(chart_type, ChartGenerator.create_line_chart)(df.loc[selected_metrics], selected_metrics, title, theme, grid)

    def display_summary_statistics(self, df: pd.DataFrame, selected_metrics: List[str]):
        if not selected_metrics: return
        st.subheader("📈 Summary Statistics")
        st.dataframe(df.loc[selected_metrics].T.describe().round(2))

    def display_welcome_message(self):
        st.markdown('<div class="welcome-container"><h2>🎯 Welcome to the Advanced Financial Dashboard</h2><p>Upload your Capitaline financial data to begin.</p></div>', unsafe_allow_html=True)
        st.subheader("📋 Getting Started")
        st.markdown("1. **Upload File**: Use the sidebar to upload your Capitaline .xls, .xlsx, or .html file.\n2. **Select Metrics**: Choose metrics from the dropdown that appears.\n3. **Choose Chart Type**: Select from Line, Bar, Area, or Heatmap.\n4. **Generate**: Click the 'Generate Chart' button to visualize.")
        st.info("💡 **Tip**: Enable 'Show Data Quality Metrics' in the sidebar for advanced insights!")

    # --- THIS IS THE FINAL PIECE OF THE PUZZLE ---
    def display_capitaline_data(self, analysis_data: Dict[str, Any], controls: Dict[str, Any]):
        """The main display function for the dashboard once data is loaded."""
        data_quality = analysis_data.get("data_quality")
        if controls.get("show_data_quality") and data_quality:
            self.display_data_quality_metrics(data_quality)
            st.markdown("---")
        
        outliers = analysis_data.get("outliers", {})
        if controls.get("show_outliers") and outliers:
            self.display_outliers(outliers)
            st.markdown("---")

        statement_df = analysis_data.get("statement")
        if statement_df is not None and not statement_df.empty:
            st.subheader("Chart Generation Controls")
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                selected_rows = st.multiselect("Select metrics to chart:", options=statement_df.index.tolist())
            with col2:
                chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Heatmap"])
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                generate = st.button("📊 Generate Chart", type="primary", use_container_width=True, disabled=(not selected_rows))

            if generate:
                with st.spinner("Generating chart..."):
                    fig = self.generate_chart(statement_df, selected_rows, chart_type, controls.get("chart_theme", "plotly_white"), controls.get("show_grid", True))
                    st.session_state.chart_figure = fig
            
            if st.session_state.chart_figure:
                st.markdown("---")
                st.plotly_chart(st.session_state.chart_figure, use_container_width=True)
                self.display_summary_statistics(statement_df, selected_rows)
            
            # Data Table with Search
            st.markdown("---")
            st.subheader("📄 Raw Data Table")
            search_term = st.text_input("Search for a metric in the table below:", placeholder="e.g., Net Sales, EPS...")
            
            if search_term:
                filtered_df = statement_df[statement_df.index.str.contains(search_term, case=False)]
            else:
                filtered_df = statement_df
            
            st.dataframe(filtered_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
            
    # --- The main execution block of the UI class ---
    def run(self):
        """The main execution loop for the Streamlit app."""
        self.render_header()
        controls = self.render_sidebar()

        if controls["file"]:
            if controls["file"] != st.session_state._uploaded_file_memo:
                with st.spinner("Processing file..."):
                    st.session_state._uploaded_file_memo = controls["file"]
                    analysis_data = parse_capitaline_file(controls["file"])
                    st.session_state.analysis_data = analysis_data
                    # Clear old chart when a new file is uploaded
                    st.session_state.chart_figure = None
        
        if st.session_state.analysis_data:
            self.display_capitaline_data(st.session_state.analysis_data, controls)
        else:
            self.display_welcome_message()

# --- Main entry point of the application ---
if __name__ == "__main__":
    ui = DashboardUI()
    ui.run()
