import streamlit as st
import pandas as pd
import random
from datetime import datetime
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide")

# --- 1. HELPER FUNCTIONS ---

def clean_currency(x):
    """
    Robust cleaning: Converts string currency/text to float.
    Returns 0.0 if conversion fails.
    """
    if isinstance(x, (int, float)):
        return float(x)
    if pd.isna(x) or str(x).strip() == '':
        return 0.0
    # Remove commas, spaces, and currency symbols if present
    clean_str = str(x).replace(',', '').replace(' ', '').replace('$', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def generate_pdf(df, params, amount_col, desc_col):
    """Generates a PDF Audit Working Paper."""
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Audit Working Paper: Monetary Unit Sampling", ln=True, align='C')
    pdf.ln(5)
    
    # 1. Audit Parameters (Single Column List)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Sampling Parameters", ln=True)
    pdf.set_font("Arial", size=10)
    
    line_height = 7
    
    # Safely handle potentially missing keys
    conf_level = params.get('conf_level', 'N/A')
    conf_factor = params.get('conf_factor', 'N/A')
    
    params_list = [
        f"Execution Time:   {params['timestamp']}",
        f"Random Seed:      {params['random_seed']}",
        f"Population Total: ${params['total_value']:,.2f}",
        f"Confidence Level: {conf_level}", 
        f"Confidence Factor:{conf_factor}",
        f"Sampling Interval:${params['interval']:,.2f}",
        f"Random Start:     {params['random_start']:,}",
        f"Items Selected:   {params['count']}"
    ]

    for p in params_list:
        pdf.cell(0, line_height, p, ln=True)
        
    pdf.ln(5)
    
    # 2. Results Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Selected Sample Items", ln=True)
    
    # Table Header
    pdf.set_font("Arial", 'B', 8)
    pdf.set_fill_color(220, 230, 240) # Light blue
    
    w_row = 12
    w_desc = 75   
    w_amt = 35    
    w_hit = 40    
    w_cum = 40    
    w_note = 30   
    
    pdf.cell(w_row, 8, "Row #", 1, 0, 'C', True)
    pdf.cell(w_desc, 8, "Customer / Description", 1, 0, 'L', True)
    pdf.cell(w_amt, 8, "Balance", 1, 0, 'R', True)
    pdf.cell(w_hit, 8, "Audit Hit Point", 1, 0, 'R', True)
    pdf.cell(w_cum, 8, "Cumulative Bal", 1, 0, 'R', True)
    pdf.cell(w_note, 8, "Note", 1, 1, 'C', True)
    
    # Table Rows
    pdf.set_font("Arial", size=8)
    for i, row in df.iterrows():
        # Retrieve data
        row_num = str(row.get('Row_Index_1_Based', ''))
        
        # Description
        desc_text = str(row.get(desc_col, ''))
        if len(desc_text) > 45: desc_text = desc_text[:42] + "..."
        
        # FIX: Force float conversion before formatting
        # This fixes "Unknown format code 'f' for object of type 'str'"
        try:
            amt_val = clean_currency(row.get(amount_col, 0))
            hit_val = clean_currency(row.get('Audit_Hit', 0))
            cum_val = clean_currency(row.get('Cumulative_Balance', 0))
        except:
            amt_val, hit_val, cum_val = 0.0, 0.0, 0.0
            
        note_val = str(row.get('Audit_Note', ''))

        # Print Cells
        pdf.cell(w_row, 8, row_num, 1, 0, 'C')
        pdf.cell(w_desc, 8, desc_text, 1, 0, 'L')
        pdf.cell(w_amt, 8, f"{amt_val:,.2f}", 1, 0, 'R')
        pdf.cell(w_hit, 8, f"{hit_val:,.2f}", 1, 0, 'R')
        pdf.cell(w_cum, 8, f"{cum_val:,.2f}", 1, 0, 'R')
        pdf.cell(w_note, 8, note_val, 1, 1, 'C')
        
    return pdf.output(dest='S').encode('latin-1')

def perform_mus_audit(df, amount_col, interval, random_seed):
    # Setup
    population = df.copy()
    
    # Clean Data
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
        
        # Merge back
        original_rows = df.loc[matches_df['Original_Index']].copy()
        
        # Add Audit Columns
        original_rows['Cumulative_Balance'] = matches_df['Cumulative_Balance'].values
        original_rows['Audit_Hit'] = matches_df['Audit_Hit'].values
        original_rows['Audit_Note'] = matches_df['Audit_Note'].values
        
        # 1-based Index creation (Original Index + 1)
        original_rows['Row_Index_1_Based'] = matches_df['Original_Index'].values + 1
        
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
    audit_params_display = {}
    
    if method == "Target Sample Size":
        target_sample_size = st.number_input("Items to Select", value=25, min_value=1)
    elif method == "Manual Interval":
        final_interval = st.number_input("Interval Amount ($)", value=100000.0, step=1000.0)
    else:
        st.info("Zero Expected Errors Model")
        confidence_options = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99]
        conf_level = st.selectbox("Confidence Level (%)", confidence_options, index=9)
        
        tol_misstatement = st.number_input("Tolerable Misstatement ($)", value=50000.0)
        
        if tol_misstatement > 0:
            lookup = {
                50: 0.7, 55: 0.8, 60: 0.9, 65: 1.1, 70: 1.2,
                75: 1.4, 80: 1.6, 85: 1.9, 90: 2.3, 95: 3.0,
                98: 3.7, 99: 4.6
            }
            factor = lookup.get(conf_level, 3.0)
            final_interval = tol_misstatement / factor
            st.write(f"**Confidence Factor:** {factor}")
            st.write(f"**Calculated Interval:** ${final_interval:,.2f}")
            
            audit_params_display['conf_level'] = f"{conf_level}%"
            audit_params_display['conf_factor'] = factor

    st.markdown("---")
    st.header("3. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    run_btn = st.button("Run Sampling")

# Main Area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.write("### Data Preview")
        
        # --- FIX: ADJUST PREVIEW INDEX TO START AT 1 ---
        preview_df = df.head(5).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(preview_df)
        # -----------------------------------------------
        
        # Column Selectors
        all_cols = df.columns.tolist()
        col1, col2, col3 = st.columns(3)
        
        with col1:
             amount_col = st.selectbox("Select Amount Column", all_cols)
        
        with col2:
             # Guess Description
             default_idx = 0
             for i, col in enumerate(all_cols):
                 if any(x in col.lower() for x in ['customer', 'desc', 'name', 'details']):
                     default_idx = i
                     break
             desc_col = st.selectbox("Select Description/Customer Column", all_cols, index=default_idx)
             
        # Dynamic Total Calculation
        temp_clean = df[amount_col].apply(clean_currency)
        total_val = temp_clean.abs().sum()
        
        with col3:
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
                    params.update(audit_params_display)
                
                if not result_df.empty:
                    st.success(f"Audit Complete: {params['count']} items selected.")
                    
                    # --- PARAMETERS REPORT ---
                    st.subheader("📋 Audit Parameters Report")
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Random Seed", params['random_seed'])
                    p_col2.metric("Random Start", f"{params['random_start']:,}")
                    p_col3.metric("Interval", f"${params['interval']:,.2f}")
                    p_col4.metric("Population Total", f"${params['total_value']:,.2f}")
                    
                    st.divider()
                    
                    # --- RESULTS TABLE ---
                    st.subheader("✅ Selected Sample Items")
                    
                    # Display logic: Reorder so "Row Index" is first
                    display_cols = ['Row_Index_1_Based'] + [c for c in result_df.columns if c != 'Row_Index_1_Based']
                    display_df = result_df[display_cols].copy()
                    
                    # Rounding for Display/CSV
                    display_df['Audit_Hit'] = display_df['Audit_Hit'].round(2)
                    display_df['Cumulative_Balance'] = display_df['Cumulative_Balance'].round(2)
                    
                    # Make Index clean in display
                    display_df.set_index('Row_Index_1_Based', inplace=True)
                    
                    st.dataframe(display_df)
                    
                    # --- EXPORT SECTION ---
                    st.subheader("💾 Export Workpapers")
                    e_col1, e_col2 = st.columns(2)
                    
                    # 1. CSV Download
                    csv = display_df.to_csv().encode('utf-8')
                    with e_col1:
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name='audit_sample.csv',
                            mime='text/csv',
                        )
                    
                    # 2. PDF Download
                    try:
                        pdf_bytes = generate_pdf(result_df, params, amount_col, desc_col)
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
