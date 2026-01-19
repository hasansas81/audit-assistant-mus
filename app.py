import streamlit as st
import pandas as pd
import random
from datetime import datetime
import pytz
from fpdf import FPDF
import re # Added for Regex cleaning

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide")

# --- 1. HELPER FUNCTIONS ---

def clean_currency(x):
    """
    Advanced Cleaner: Handles (100), -100, 100-, R 100.00, etc.
    Returns float.
    """
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return 0.0
    
    # 1. Check if it looks like a negative in parentheses: (123.45)
    is_bracket_negative = False
    if s.startswith('(') and s.endswith(')'):
        is_bracket_negative = True
        s = s[1:-1] # Remove brackets

    # 2. Remove standard currency symbols and separators
    # Keep only digits, dots, and the minus sign
    # We remove R, $, spaces, commas
    s_clean = re.sub(r'[R$£€¥,\s]', '', s)
    
    # 3. Handle Trailing Negatives (e.g., "500-")
    if s_clean.endswith('-'):
        s_clean = '-' + s_clean[:-1]
        
    try:
        val = float(s_clean)
        # Apply the bracket negative logic
        if is_bracket_negative:
            val = -abs(val)
        return val
    except ValueError:
        return 0.0

# --- 2. CUSTOM PDF CLASS ---
class AuditPDF(FPDF):
    def __init__(self, orientation='L', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.is_table_active = False 
        self.col_widths = []
        self.col_names = []

    def header(self):
        # 1. WATERMARK
        self.set_font('Arial', 'B', 30)
        self.set_text_color(240, 240, 240) # Very Light Grey
        self.text(60, 25, "Prepared by Audit Assistant") 
        
        # 2. Main Title (Page 1 only)
        if self.page_no() == 1:
            self.set_y(10)
            self.set_font("Arial", 'B', 16)
            self.set_text_color(0, 0, 0)
            self.cell(0, 10, "Audit Working Paper: Monetary Unit Sampling", ln=True, align='C')
            self.ln(5)
        else:
            self.set_y(20)

        # 3. REPEATING TABLE HEADER
        if self.page_no() > 1 and self.is_table_active:
            self.print_table_header()

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        
        self.cell(0, 10, 'Terms & Conditions: Generated for audit purposes. Auditor to verify all selections.', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()} of {{nb}}', 0, 0, 'R')

    def set_table_cols(self, names, widths):
        self.col_names = names
        self.col_widths = widths

    def print_table_header(self):
        self.set_font('Arial', 'B', 8)
        self.set_fill_color(220, 230, 240) # Light blue
        self.set_text_color(0, 0, 0)
        
        for name, width in zip(self.col_names, self.col_widths):
            self.cell(width, 8, name, 1, 0, 'C', True)
        self.ln()

def generate_pdf(df, params, amount_col, desc_col, currency_symbol):
    """Generates a professional Audit PDF."""
    
    pdf = AuditPDF(orientation='L', unit='mm', format='A4')
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # --- 1. AUDIT PARAMETERS ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Sampling Parameters", ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(220, 230, 240)
    
    w_param = 60
    w_result = 80
    h_line = 7
    
    pdf.cell(w_param, h_line, "Parameter", 1, 0, 'L', True)
    pdf.cell(w_result, h_line, "Result", 1, 1, 'L', True)
    
    conf_level = params.get('conf_level', 'N/A')
    conf_factor = params.get('conf_factor', 'N/A')
    
    # Use ABSOLUTE Total for sampling info, but maybe show Net for control?
    # Standard MUS reports usually reference the Absolute Population tested.
    
    param_rows = [
        ("Execution Time", str(params['timestamp'])),
        ("Time Zone", str(params['timezone'])),
        ("Random Seed", str(params['random_seed'])),
        ("Net Control Total", f"{currency_symbol}{float(params['net_total']):,.2f}"),
        ("Abs. Sampling Pop.", f"{currency_symbol}{float(params['total_value']):,.2f}"),
        ("Confidence Level", str(conf_level)),
        ("Confidence Factor", str(conf_factor)),
        ("Sampling Interval", f"{currency_symbol}{float(params['interval']):,.2f}"),
        ("Random Start", f"{params['random_start']:,}"),
        ("Items Selected", str(params['count']))
    ]

    pdf.set_font("Arial", size=10)
    for name, value in param_rows:
        pdf.cell(w_param, h_line, name, 1, 0, 'L')
        pdf.cell(w_result, h_line, str(value), 1, 1, 'R')
        
    pdf.ln(10)
    
    # --- 2. RESULTS TABLE ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Selected Sample Items", ln=True)
    
    header_names = ["Row #", "Customer / Description", "Balance", "Audit Hit Point", "Cumulative Bal", "Note"]
    col_widths = [12, 75, 35, 40, 40, 30]
    
    pdf.set_table_cols(header_names, col_widths)
    pdf.print_table_header()
    pdf.is_table_active = True
    
    pdf.set_font("Arial", size=8)
    
    for i, row in df.iterrows():
        row_num = str(row.get('Row_Index_1_Based', ''))
        
        desc_text = str(row.get(desc_col, ''))
        if len(desc_text) > 45: desc_text = desc_text[:42] + "..."
        
        amt_val = clean_currency(row.get(amount_col, 0))
        hit_val = clean_currency(row.get('Audit_Hit', 0))
        cum_val = clean_currency(row.get('Cumulative_Balance', 0))
        note_val = str(row.get('Audit_Note', ''))

        pdf.cell(col_widths[0], 8, row_num, 1, 0, 'C')
        pdf.cell(col_widths[1], 8, desc_text, 1, 0, 'L')
        pdf.cell(col_widths[2], 8, f"{amt_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[3], 8, f"{hit_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 8, f"{cum_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[5], 8, note_val, 1, 1, 'C')
        
    return pdf.output(dest='S').encode('latin-1')

def perform_mus_audit(df, amount_col, interval, random_seed, tz_name):
    # Setup
    population = df.copy()
    
    # Clean Data using the NEW Robust Cleaner
    population[amount_col] = population[amount_col].apply(clean_currency)
    
    # IMPORTANT: MUS uses Absolute Value for sampling logic
    # But we preserve the original sign for the "Balance" display
    population['Abs_Amount'] = population[amount_col].abs()
    
    # Calculate NET total for Control purposes (e.g. -279M)
    net_total = population[amount_col].sum()
    
    # Calculate ABSOLUTE total for Sampling purposes (e.g. 375M)
    abs_total_value = population['Abs_Amount'].sum()
    
    # Cumulative Calculation on ABSOLUTE amounts
    population['Cumulative_Balance'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Balance'] - population['Abs_Amount']
    
    if abs_total_value == 0:
        return None, "Error: Total absolute value is 0.", {}
    if interval <= 0:
        return None, "Error: Interval must be > 0.", {}

    # Timezone
    try:
        local_tz = pytz.timezone(tz_name)
        run_timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    except:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S (UTC)")

    # Random Start
    random.seed(random_seed)
    random_start = random.randint(1, int(interval))
    
    # Hit Points
    hit_points = []
    current_hit = random_start
    while current_hit <= abs_total_value:
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
        
        # We need to make sure the "Balance" column in the result is the CLEANED number (with negatives)
        # otherwise it might show the original text string like "(500)"
        original_rows[amount_col] = population.loc[matches_df['Original_Index'], amount_col]
        
        original_rows['Cumulative_Balance'] = matches_df['Cumulative_Balance'].values
        original_rows['Audit_Hit'] = matches_df['Audit_Hit'].values
        original_rows['Audit_Note'] = matches_df['Audit_Note'].values
        original_rows['Row_Index_1_Based'] = matches_df['Original_Index'].values + 1
        
        audit_params = {
            'timestamp': run_timestamp,
            'timezone': tz_name,
            'total_value': abs_total_value, # The sampling population
            'net_total': net_total,         # The control total (-279M)
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

if 'audit_result_df' not in st.session_state:
    st.session_state.audit_result_df = None
if 'audit_params' not in st.session_state:
    st.session_state.audit_params = {}
if 'audit_msg' not in st.session_state:
    st.session_state.audit_msg = ""
if 'trigger_run' not in st.session_state:
    st.session_state.trigger_run = False

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    st.header("2. Regional Settings")
    
    currency_symbol = st.selectbox("Currency Symbol", ["R", "$", "€", "£", "¥"], index=0)
    common_timezones = ['Africa/Johannesburg', 'UTC', 'Europe/London', 'America/New_York', 'Australia/Sydney']
    selected_timezone = st.selectbox("Time Zone", common_timezones, index=0)

    st.markdown("---")
    st.header("3. Sampling Settings")
    
    method = st.radio("Selection Method:", ["Target Sample Size", "Manual Interval", "Confidence Calculator"])
    
    final_interval = 0.0
    target_sample_size = 0
    audit_params_display = {}
    
    # We must access the data to show total value, but we defer calculation until data loads
    
    if method == "Target Sample Size":
        target_sample_size = st.number_input("Items to Select", value=25, min_value=1)
    elif method == "Manual Interval":
        final_interval = st.number_input(f"Interval Amount ({currency_symbol})", value=100000.0, step=1000.0)
    else:
        st.info("Zero Expected Errors Model")
        confidence_options = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99]
        conf_level = st.selectbox("Confidence Level (%)", confidence_options, index=9)
        
        tol_misstatement = st.number_input(f"Tolerable Misstatement ({currency_symbol})", value=50000.0)
        
        if tol_misstatement > 0:
            lookup = {
                50: 0.7, 55: 0.8, 60: 0.9, 65: 1.1, 70: 1.2,
                75: 1.4, 80: 1.6, 85: 1.9, 90: 2.3, 95: 3.0,
                98: 3.7, 99: 4.6
            }
            factor = lookup.get(conf_level, 3.0)
            final_interval = tol_misstatement / factor
            st.write(f"**Confidence Factor:** {factor}")
            st.write(f"**Calculated Interval:** {currency_symbol}{final_interval:,.2f}")
            
            audit_params_display['conf_level'] = f"{conf_level}%"
            audit_params_display['conf_factor'] = factor

    st.markdown("---")
    st.header("4. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    
    if st.button("Run Sampling"):
        if uploaded_file is not None:
             try:
                uploaded_file.seek(0)
                # Just reading to ensure valid CSV
                df_check = pd.read_csv(uploaded_file)
                st.session_state.trigger_run = True
             except Exception as e:
                st.error(f"Error: {e}")

# Main Area
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        st.write("### Data Preview")
        preview_df = df.head(5).copy()
        preview_df.index = preview_df.index + 1
        st.dataframe(preview_df)
        
        all_cols = df.columns.tolist()
        col1, col2, col3 = st.columns(3)
        
        with col1:
             amount_col = st.selectbox("Select Amount Column", all_cols)
        
        with col2:
             default_idx = 0
             for i, col in enumerate(all_cols):
                 if any(x in col.lower() for x in ['customer', 'desc', 'name', 'details']):
                     default_idx = i
                     break
             desc_col = st.selectbox("Select Description/Customer Column", all_cols, index=default_idx)
             
        # DYNAMIC TOTAL CALCULATION (Updated for Dual Totals)
        # Apply cleaner to get the real numeric series
        clean_series = df[amount_col].apply(clean_currency)
        net_total_val = clean_series.sum()
        abs_total_val = clean_series.abs().sum()
        
        with col3:
            # Show Net Total (Control)
            st.metric("Net Control Total (Net)", f"{currency_symbol}{net_total_val:,.2f}")
            # Show Absolute Total (For Sampling)
            st.caption(f"Abs. Sampling Pop: {currency_symbol}{abs_total_val:,.2f}")

        if method == "Target Sample Size" and abs_total_val > 0:
            final_interval = abs_total_val / target_sample_size
            st.info(f"**Interval:** {currency_symbol}{final_interval:,.2f} (Target: {target_sample_size} items)")

        # RUN LOGIC
        if st.session_state.trigger_run:
            if final_interval <= 0:
                st.error("Interval must be greater than 0.")
            else:
                with st.spinner('Calculating...'):
                    # Pass Timezone here
                    result_df, msg, params = perform_mus_audit(df, amount_col, final_interval, random_seed, selected_timezone)
                    params.update(audit_params_display)
                    
                    st.session_state.audit_result_df = result_df
                    st.session_state.audit_params = params
                    st.session_state.audit_msg = msg
            
            st.session_state.trigger_run = False

        # DISPLAY RESULTS
        if st.session_state.audit_result_df is not None:
            res_df = st.session_state.audit_result_df
            p = st.session_state.audit_params
            
            if not res_df.empty:
                st.success(f"Audit Complete: {p['count']} items selected.")
                
                st.subheader("📋 Audit Parameters Report")
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                p_col1.metric("Random Seed", p['random_seed'])
                p_col2.metric("Random Start", f"{p['random_start']:,}")
                p_col3.metric("Interval", f"{currency_symbol}{p['interval']:,.2f}")
                # Show both totals in the final report
                p_col4.metric("Net Control Total", f"{currency_symbol}{p['net_total']:,.2f}")
                
                st.caption(f"Absolute Sampling Population: {currency_symbol}{p['total_value']:,.2f} | Time Zone: {p.get('timezone', 'UTC')}")
                st.divider()
                
                st.subheader("✅ Selected Sample Items")
                display_cols = ['Row_Index_1_Based'] + [c for c in res_df.columns if c != 'Row_Index_1_Based']
                display_df = res_df[display_cols].copy()
                
                display_df['Audit_Hit'] = display_df['Audit_Hit'].round(2)
                display_df['Cumulative_Balance'] = display_df['Cumulative_Balance'].round(2)
                display_df.set_index('Row_Index_1_Based', inplace=True)
                
                st.dataframe(display_df)
                
                st.subheader("💾 Export Workpapers")
                e_col1, e_col2 = st.columns(2)
                
                csv = display_df.to_csv().encode('utf-8')
                with e_col1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name='audit_sample.csv',
                        mime='text/csv',
                    )
                
                try:
                    pdf_bytes = generate_pdf(res_df, p, amount_col, desc_col, currency_symbol)
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
                st.warning(st.session_state.audit_msg)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👈 Upload your client's CSV file in the sidebar to start.")
