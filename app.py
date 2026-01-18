import streamlit as st
import pandas as pd
import random
from datetime import datetime
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide")

# --- 1. HELPER FUNCTIONS ---

def clean_currency(x):
    """Converts string currency to float."""
    if isinstance(x, (int, float)):
        return x
    if pd.isna(x) or x == '':
        return 0.0
    clean_str = str(x).replace(',', '').replace(' ', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def generate_pdf(df, params):
    """Generates a PDF Audit Working Paper."""
    pdf = FPDF(orientation='L', unit='mm', format='A4') # Landscape for better table fit
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Audit Working Paper: Monetary Unit Sampling", ln=True, align='C')
    pdf.ln(5)
    
    # Audit Parameters Block
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Sampling Parameters", ln=True)
    pdf.set_font("Arial", size=10)
    
    # Parameter Grid
    col_width = 60
    line_height = 8
    
    # Row 1
    pdf.cell(col_width, line_height, f"Execution Time: {params['timestamp']}", border=1)
    pdf.cell(col_width, line_height, f"Random Seed: {params['random_seed']}", border=1)
    pdf.cell(col_width, line_height, f"Population Total: ${params['total_value']:,.2f}", border=1)
    pdf.ln(line_height)
    
    # Row 2
    pdf.cell(col_width, line_height, f"Sampling Interval: ${params['interval']:,.2f}", border=1)
    pdf.cell(col_width, line_height, f"Random Start: {params['random_start']:,}", border=1)
    pdf.cell(col_width, line_height, f"Items Selected: {params['count']}", border=1)
    pdf.ln(15)
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Selected Sample Items", ln=True)
    
    # Table Columns (Adjusted for PDF width)
    # We will pick specific columns to print to ensure it fits
    pdf.set_font("Arial", 'B', 9)
    pdf.set_fill_color(200, 220, 255) # Light blue header
    
    # Define widths
    w_idx = 20
    w_amount = 40
    w_hit = 40
    w_cum = 40
    w_note = 40
    # Remaining width for description (approx 275 total width in A4 landscape)
    w_desc = 90 
    
    pdf.cell(w_idx, 8, "Row #", 1, 0, 'C', True)
    pdf.cell(w_amount, 8, "Recorded Amount", 1, 0, 'C', True)
    pdf.cell(w_hit, 8, "Audit Hit Point", 1, 0, 'C', True)
    pdf.cell(w_cum, 8, "Cumulative Bal", 1, 0, 'C', True)
    pdf.cell(w_note, 8, "Note", 1, 0, 'C', True)
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=9)
    for i, row in df.iterrows():
        # Clean formatting for PDF
        amt = f"{row.get('Recorded_Amount', 0):,.2f}"
        hit = f"{row.get('Audit_Hit', 0):,.0f}"
        cum = f"{row.get('Cumulative_Balance', 0):,.2f}"
        note = str(row.get('Audit_Note', ''))
        idx = str(row.get('Original_Index', ''))
        
        pdf.cell(w_idx, 8, idx, 1)
        pdf.cell(w_amount, 8, amt, 1, 0, 'R')
        pdf.cell(w_hit, 8, hit, 1, 0, 'R')
        pdf.cell(w_cum, 8, cum, 1, 0, 'R')
        pdf.cell(w_note, 8, note, 1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

def perform_mus_audit(df, amount_col, interval, random_seed):
    # Setup
    population = df.copy()
    population[amount_col] = population[amount_col].apply(clean_currency)
    population['Abs_Amount'] = population[amount_col].abs()
    
    # Cumulative Calculation
    population['Cumulative_Balance'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Balance'] - population['Abs_Amount']
    
    total_value = population['Abs_Amount'].sum()
    
    if total_value == 0:
        return None, "Error: Total value is 0.", {}
    if interval <= 0:
        return None, "Error: Interval must be > 0.", {}

    # Timestamp
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Random Start
    random.seed(random_seed)
    random_start = random.randint(1, int(interval))
    
    # Hit Points
    hit_points = []
    current_hit = random_start
    while current_hit <= total_value:
        hit_points.append(current_hit)
        current_hit += interval
        
    # Selection Logic
    selection_results = []
    hit_idx = 0
    num_hits = len(hit_points)

    for index, row in population.iterrows():
        low = row['Previous_Cumulative']
        high = row['Cumulative_Balance']
        
        while hit_idx < num_hits and hit_points[hit_idx] <= high:
            current_target = hit_points[hit_idx]
            if current_target > low:
                selection_results.append({
                    'Original_Index': index,
                    'Audit_Hit': current_target,
                    'Cumulative_Balance': high,
                    'Audit_Note': 'High Value' if row['Abs_Amount'] >= interval else 'Sampled'
                })
            hit_idx += 1

    if selection_results:
        matches_df = pd.DataFrame(selection_results)
        original_rows = df.loc[matches_df['Original_Index']].copy()
        
        # Add Audit Columns
        original_rows['Cumulative_Balance'] = matches_df['Cumulative_Balance'].values
        original_rows['Audit_Hit'] = matches_df['Audit_Hit'].values
        original_rows['Audit_Note'] = matches_df['Audit_Note'].values
        original_rows['Original_Index'] = matches_df['Original_Index'].values # Needed for PDF
        
        # Metadata
        audit_params = {
            'timestamp': run_timestamp,
            'total_value': total_value,
            'random_seed': random_seed,
            'random_start': random_start,
            'interval': interval,
            'count': len(original_rows)
        }
        
        return original_rows, "Success", audit_params
    else:
        return pd.DataFrame(), "No items selected.", {}

# --- 2. THE APP INTERFACE ---

st.title("🛡️ Audit Assistant: Monetary Unit Sampling")
st.markdown("Statutory audit tool for detailed substantive testing.")

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    st.header("2. Sampling Settings")
    
    method = st.radio("Selection Method:", ["Target Sample Size", "Manual Interval", "Confidence Calculator"])
    
    final_interval = 0.0
    target_sample_size = 0
    
    if method == "Target Sample Size":
        target_sample_size = st.number_input("Items to Select", value=25, min_value=1)
    elif method == "Manual Interval":
        final_interval = st.number_input("Interval Amount ($)", value=100000.0, step=1000.0)
    else:
        st.info("Zero Expected Errors Model")
        # UPDATED CONFIDENCE LEVELS based on the image
        confidence_options = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99]
        # Defaulting to 95%, which is at index 9
        conf_level = st.selectbox("Confidence Level (%)", confidence_options, index=9)
        
        tol_misstatement = st.number_input("Tolerable Misstatement ($)", value=50000.0)
        if tol_misstatement > 0:
            # UPDATED CONFIDENCE FACTORS based on the image
            lookup = {
                50: 0.7, 55: 0.8, 60: 0.9, 65: 1.1, 70: 1.2,
                75: 1.4, 80: 1.6, 85: 1.9, 90: 2.3, 95: 3.0,
                98: 3.7, 99: 4.6
            }
            factor = lookup.get(conf_level, 3.0)
            final_interval = tol_misstatement / factor
            st.write(f"**Confidence Factor:** {factor}")
            st.write(f"**Calculated Interval:** ${final_interval:,.2f}")

    st.markdown("---")
    st.header("3. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    run_btn = st.button("Run Sampling")

# Main Area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.write("### Data Preview")
        st.dataframe(df.head(5))
        
        all_cols = df.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
             amount_col = st.selectbox("Select Amount Column", all_cols)
        
        temp_clean = df[amount_col].apply(clean_currency)
        total_val = temp_clean.abs().sum()
        
        with col2:
            st.metric("Total Population Value", f"${total_val:,.2f}")

        if method == "Target Sample Size" and total_val > 0:
            final_interval = total_val / target_sample_size
            st.info(f"**Interval:** ${final_interval:,.2f} (Target: {target_sample_size} items)")

        if run_btn:
            if final_interval <= 0:
                st.error("Interval must be greater than 0.")
            else:
                with st.spinner('Calculating...'):
                    result_df, msg, params = perform_mus_audit(df, amount_col, final_interval, random_seed)
                
                if not result_df.empty:
                    st.success(f"Audit Complete: {params['count']} items selected.")
                    
                    # --- PARAMETERS REPORT ---
                    st.subheader("📋 Audit Parameters Report")
                    st.caption(f"Run Date/Time: {params['timestamp']}")
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Random Seed", params['random_seed'])
                    p_col2.metric("Random Start", f"{params['random_start']:,}")
                    p_col3.metric("Interval", f"${params['interval']:,.2f}")
                    p_col4.metric("Population Total", f"${params['total_value']:,.2f}")
                    
                    st.divider()
                    
                    # --- RESULTS TABLE ---
                    st.subheader("✅ Selected Sample Items")
                    st.dataframe(result_df)
                    
                    # --- EXPORT SECTION ---
                    st.subheader("💾 Export Workpapers")
                    e_col1, e_col2 = st.columns(2)
                    
                    # 1. CSV Download
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    with e_col1:
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name='audit_sample.csv',
                            mime='text/csv',
                        )
                    
                    # 2. PDF Download
                    try:
                        pdf_bytes = generate_pdf(result_df, params)
                        with e_col2:
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_bytes,
                                file_name='audit_working_paper.pdf',
                                mime='application/pdf'
                            )
                    except Exception as e:
                        st.error(f"PDF Generation Error: {e}")

                else:
                    st.warning(msg)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Upload your client's CSV file in the sidebar to start.")
