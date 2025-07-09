# /mount/src/capitaline/app.py - DIAGNOSTIC VERSION

"""
Frontend Streamlit Dashboard - Diagnostic Tool for Capitaline File Parsing
"""

import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="File Parser Diagnostic Tool",
    layout="wide"
)

def run_diagnostic_tool():
    """
    This function creates a UI to upload a file and display its raw structure for debugging.
    """
    st.title("Capitaline File Structure - Diagnostic Tool")
    
    st.info(
        "**Instructions:**\n\n"
        "1. Upload the `.xls` file you downloaded from Capitaline.\n"
        "2. The application will display the raw structure of the data table(s) found inside.\n"
        "3. **Please copy all the information and tables that appear below and paste them in your reply.**\n\n"
        "This will allow me to write a parser that is perfectly tailored to your file format."
    )
    
    uploaded_file = st.file_uploader(
        "Upload your .xls file here",
        type=['xls']
    )
    
    if uploaded_file:
        try:
            st.markdown("---")
            st.header("Diagnostic Information")

            # Read the file's content as raw HTML text
            html_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")

            # Use Pandas to find all tables in the HTML
            tables = pd.read_html(io.StringIO(html_content))
            
            st.success(f"Successfully found **{len(tables)}** table(s) in the file.")
            
            # Loop through each table found and display its raw structure
            for i, df in enumerate(tables):
                st.subheader(f"Table #{i+1} (Raw Structure)")
                st.write(f"Shape (rows, columns): {df.shape}")
                
                st.text("First 10 rows of this table (including headers):")
                st.dataframe(df.head(10))
                
                # Capture the df.info() output to show column types
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_string = buffer.getvalue()
                
                st.text("Table Info (Column Data Types):")
                st.code(info_string)
                
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred while trying to read the file: {e}")
            st.warning("This might happen if the file is not an HTML-based table. Please still provide any output you see.")

# --- Main Execution Block ---
if __name__ == "__main__":
    run_diagnostic_tool()
