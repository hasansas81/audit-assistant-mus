import streamlit as st
import pandas as pd
import random
from datetime import datetime
import pytz
from fpdf import FPDF
import re
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide", initial_sidebar_state="expanded")

# --- 0. PERSISTENT STORAGE (THE MINI-DATABASE) ---
DB_FILE = "usage_db.json"

def load_usage_data():
    """Loads usage counts from a local JSON file."""
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_usage_data(data):
    """Saves usage counts to a local JSON file."""
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

def get_user_usage(username):
    data = load_usage_data()
    # If user doesn't exist, start them at 0
    return data.get(username, 0)

def increment_user_usage(username):
    data = load_usage_data()
    current = data.get(username, 0)
    data[username] = current + 1
    save_usage_data(data)
    return data[username]

# --- AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# CREDENTIALS
VALID_USER = "audit_user"
VALID_PASS = "secure_audit_2026"

def login_screen():
    st.title("🔐 Audit Assistant Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Welcome to the Audit Assistant MVP. Please log in to access the MUS Tool.")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In"):
            if user == VALID_USER and password == VALID_PASS:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("Invalid Username or Password")
        
        st.markdown("---")
        st.caption("Lost password? Contact support@auditassistant.com")

# --- 1. HELPER FUNCTIONS ---

def clean_currency(x):
    """Advanced Cleaner: Handles accounting formats like (10,000.00) or 10000-"""
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return 0.0
    
    is_bracket_negative = False
    if s.startswith('(') and s.endswith(')'):
        is_bracket_negative = True
        s = s[1:-1]

    s_clean = re.sub(r'[R$£€¥,\s]', '', s)
    
    if s_clean.endswith('-'):
        s_clean = '-' + s_clean[:-1]
        
    try:
        val = float(s_clean)
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
        if self.page_no() == 1:
            self.set_y(10)
            self.set_font("Arial", 'B', 16)
            self.set_text_color(0, 0, 0)
            self.cell(0, 10, "Audit Working Paper: Monetary Unit Sampling", ln=True, align='C')
            self.ln(5)
        else:
            self.set_y(20)

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
        self.set_fill_color(220, 230, 240)
        self.set_text_color(0, 0, 0)
        for name, width in zip(self.col_names, self.col_widths):
            self.cell(width, 8, name, 1, 0, 'C', True)
        self.ln()

def generate_pdf(df, params, amount_col, desc_col, currency_symbol):
    pdf = AuditPDF(orientation='L', unit='mm', format='A4')
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # PARAMETERS
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
    net_total = params.get('net_total', 0.0)
    total_val = params.get('total_value', 0.0)
    interval = params.get('interval', 0.0)
    sample_total = params.get('sample_net_total', 0.0)
    
    param_rows = [
        ("Execution Time", str(params.get('timestamp', 'N/A'))),
        ("Time Zone", str(params.get('timezone', 'UTC'))),
        ("Random Seed", str(params.get('random_seed', 'N/A'))),
        ("Net Control Total", f"{currency_symbol}{float(net_total):,.2f}"),
        ("Net Sample Total", f"{currency_symbol}{float(sample_total):,.2f}"),
        ("Confidence Level", str(conf_level)),
        ("Confidence Factor", str(conf_factor)),
        ("Sampling Interval", f"{currency_symbol}{float(interval):,.2f}"),
        ("Random Start", f"{params.get('random_start', 0):,}"),
        ("Items Selected", str(params.get('count', 0)))
    ]

    pdf.set_font("Arial", size=10)
    for name, value in param_rows:
        pdf.cell(w_param, h_line, name, 1, 0, 'L')
        pdf.cell(w_result, h_line, str(value), 1, 1, 'R')
        
    pdf.ln(10)
    
    # RESULTS TABLE
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Selected Sample Items", ln=True)
    
    header_names = ["No.", "Row", "Customer / Description", "Balance", "Run. Bal (Net)", "Samp. Index", "Audit Hit", "Note"]
    col_widths = [10, 12, 60, 30, 35, 35, 35, 60]
    
    pdf.set_table_cols(header_names, col_widths)
    pdf.print_table_header()
    pdf.is_table_active = True
    
    pdf.set_font("Arial", size=8)
    
    for i, row in df.iterrows():
        item_num = str(row.get('Item_No', '')) 
        row_num = str(row.get('Row_Index_1_Based', ''))
        
        desc_text = str(row.get(desc_col, ''))
        if len(desc_text) > 35: desc_text = desc_text[:32] + "..."
        
        amt_val = clean_currency(row.get(amount_col, 0))
        run_bal_val = clean_currency(row.get('Running_Net_Balance', 0))
        cum_val = clean_currency(row.get('Cumulative_Balance', 0))
        hit_val = clean_currency(row.get('Audit_Hit', 0))
        note_val = str(row.get('Audit_Note', ''))

        pdf.cell(col_widths[0], 8, item_num, 1, 0, 'C')
        pdf.cell(col_widths[1], 8, row_num, 1, 0, 'C')
        pdf.cell(col_widths[2], 8, desc_text, 1, 0, 'L')
        pdf.cell(col_widths[3], 8, f"{amt_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 8, f"{run_bal_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[5], 8, f"{cum_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[6], 8, f"{hit_val:,.2f}", 1, 0, 'R')
        pdf.cell(col_widths[7], 8, note_val, 1, 1, 'C')
        
    return pdf.output(dest='S').encode('latin-1')

def perform_mus_audit(df, amount_col, interval, random_seed, tz_name):
    population = df.copy()
    population[amount_col] = population[amount_col].apply(clean_currency)
    population['Running_Net_Balance'] = population[amount_col].cumsum()
    population['Abs_Amount'] = population[amount_col].abs()
    population['Cumulative_Balance'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Balance'] - population['Abs_Amount']
    
    net_total = population[amount_col].sum()
    abs_total_value = population['Abs_Amount'].sum()
    
    if abs_total_value == 0:
        return None, "Error: Total absolute value is 0.", {}
    if interval <= 0:
        return None, "Error: Interval must be > 0.", {}

    try:
        local_tz = pytz.timezone(tz_name)
        run_timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    except:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S (UTC)")

    random.seed(random_seed)
    random_start = random.randint(1, int(interval))
    
    hit_points = []
    current_hit = random_start
    while current_hit <= abs_total_value:
        hit_points.append(current_hit)
        current_hit += interval
        
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
                    'Running_Net_Balance': row['Running_Net_Balance'], 
                    'Audit_Note': 'High Value' if row['Abs_Amount'] >= interval else 'Sampled'
                })
            hit_idx += 1

    if selection_results:
        matches_df = pd.DataFrame(selection_results)
        original_rows = df.loc[matches_df['Original_Index']].copy()
        original_rows[amount_col] = population.loc[matches_df['Original_Index'], amount_col]
        original_rows['Running_Net_Balance'] = matches_df['Running_Net_Balance'].values
        original_rows['Cumulative_Balance'] = matches_df['Cumulative_Balance'].values
        original_rows['Audit_Hit'] = matches_df['Audit_Hit'].values
        original_rows['Audit_Note'] = matches_df['Audit_Note'].values
        original_rows['Row_Index_1_Based'] = matches_df['Original_Index'].values + 1
        original_rows['Item_No'] = range(1, len(original_rows) + 1)
        
        sample_net_total = original_rows[amount_col].sum()
        
        audit_params = {
            'timestamp': run_timestamp,
            'timezone': tz_name,
            'total_value': abs_total_value,
            'net_total': net_total, 
            'sample_net_total': sample_net_total, 
            'random_seed': random_seed,
            'random_start': random_start,
            'interval': interval,
            'count': len(original_rows)
        }
        return original_rows, "Success", audit_params
    else:
        return pd.DataFrame(), "No items selected.", {}

# --- 3. MAIN APP FLOW ---

if not st.session_state.logged_in:
    login_screen()
else:
    # --- LOGGED IN VIEW ---
    st.title("🛡️ Audit Assistant: Monetary Unit Sampling")

    if 'audit_result_df' not in st.session_state:
        st.session_state.audit_result_df = None
    if 'audit_params' not in st.session_state:
        st.session_state.audit_params = {}
    if 'audit_msg' not in st.session_state:
        st.session_state.audit_msg = ""
    if 'trigger_run' not in st.session_state:
        st.session_state.trigger_run = False

    # SIDEBAR
    with st.sidebar:
        user = st.session_state.username
        
        # Load Usage from "Mini-DB"
        current_usage = get_user_usage(user)
        
        st.write(f"Logged in as: **{user}**")
        st.info(f"Usage: {current_usage} / 5 Free Runs")
        
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
                lookup = {50: 0.7, 55: 0.8, 60: 0.9, 65: 1.1, 70: 1.2, 75: 1.4, 80: 1.6, 85: 1.9, 90: 2.3, 95: 3.0, 98: 3.7, 99: 4.6}
                factor = lookup.get(conf_level, 3.0)
                final_interval = tol_misstatement / factor
                st.write(f"**Confidence Factor:** {factor}")
                st.write(f"**Calculated Interval:** {currency_symbol}{final_interval:,.2f}")
                audit_params_display['conf_level'] = f"{conf_level}%"
                audit_params_display['conf_factor'] = factor

        st.markdown("---")
        st.header("4. Execution")
        random_seed = st.number_input("Random Seed", value=12345, step=1)
        
        # RUN BUTTON
        if st.button("Run Sampling"):
            # Update Persistent Usage
            new_count = increment_user_usage(user)
            st.session_state.usage_count = new_count # Update session to match DB
            
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    df_check = pd.read_csv(uploaded_file)
                    st.session_state.trigger_run = True
                except Exception as e:
                    st.error(f"Error: {e}")
                    
        # RESET BUTTON
        if st.button("🔄 Start New / Clear"):
            st.session_state.audit_result_df = None
            st.session_state.audit_params = {}
            st.session_state.audit_msg = ""
            st.session_state.trigger_run = False
            st.rerun()

        # LOGOUT BUTTON
        st.markdown("---")
        if st.button("🚪 Log Out"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.audit_result_df = None
            st.rerun()

        st.markdown("---")
        with st.expander("🔒 Pro Features (Locked)"):
            st.caption("• Unlimited Sampling Runs")
            st.caption("• Large File Support (>50k rows)")
            st.caption("• Cloud Storage")
            st.caption("• Team Collaboration")
            st.button("Upgrade to Pro", disabled=True)

    # MAIN AREA
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
                
            clean_series = df[amount_col].apply(clean_currency)
            net_total_val = clean_series.sum()
            abs_total_val = clean_series.abs().sum()
            
            with col3:
                st.metric("Net Control Total", f"{currency_symbol}{net_total_val:,.2f}")
                st.caption(f"Abs. Sampling Pop: {currency_symbol}{abs_total_val:,.2f}")

            if method == "Target Sample Size" and abs_total_val > 0:
                final_interval = abs_total_val / target_sample_size
                st.info(f"**Interval:** {currency_symbol}{final_interval:,.2f} (Target: {target_sample_size} items)")

            if st.session_state.trigger_run:
                if final_interval <= 0:
                    st.error("Interval must be greater than 0.")
                else:
                    with st.spinner('Calculating...'):
                        result_df, msg, params = perform_mus_audit(df, amount_col, final_interval, random_seed, selected_timezone)
                        params.update(audit_params_display)
                        st.session_state.audit_result_df = result_df
                        st.session_state.audit_params = params
                        st.session_state.audit_msg = msg
                st.session_state.trigger_run = False

            if st.session_state.audit_result_df is not None:
                res_df = st.session_state.audit_result_df
                p = st.session_state.audit_params
                
                if not res_df.empty:
                    st.success(f"Audit Complete: {p.get('count',0)} items selected.")
                    
                    st.subheader("📋 Audit Parameters Report")
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Items Selected", p.get('count', 0)) 
                    p_col2.metric("Interval", f"{currency_symbol}{p.get('interval', 0):,.2f}")
                    p_col3.metric("Net Sample Total", f"{currency_symbol}{p.get('sample_net_total', 0):,.2f}")
                    p_col4.metric("Net Control Total", f"{currency_symbol}{p.get('net_total', 0):,.2f}")
                    
                    st.caption(f"Absolute Sampling Pop: {currency_symbol}{p.get('total_value', 0):,.2f} | Seed: {p.get('random_seed')} | Start: {p.get('random_start'):,}")
                    st.divider()
                    
                    st.subheader("✅ Selected Sample Items")
                    display_cols = ['Item_No', 'Row_Index_1_Based'] + [c for c in res_df.columns if c not in ['Item_No', 'Row_Index_1_Based']]
                    display_df = res_df[display_cols].copy()
                    display_df['Audit_Hit'] = display_df['Audit_Hit'].round(2)
                    display_df['Running_Net_Balance'] = display_df['Running_Net_Balance'].round(2)
                    display_df['Cumulative_Balance'] = display_df['Cumulative_Balance'].round(2)
                    display_df.set_index('Item_No', inplace=True)
                    st.dataframe(display_df)
                    
                    st.subheader("💾 Export Workpapers")
                    e_col1, e_col2 = st.columns(2)
                    csv = display_df.to_csv().encode('utf-8')
                    with e_col1:
                        st.download_button("📥 Download as CSV", data=csv, file_name='audit_sample.csv', mime='text/csv')
                    
                    try:
                        pdf_bytes = generate_pdf(res_df, p, amount_col, desc_col, currency_symbol)
                        with e_col2:
                            st.download_button("📄 Download as PDF", data=pdf_bytes, file_name='audit_working_paper.pdf', mime='application/pdf')
                    except Exception as e:
                        st.error(f"PDF Generation Error: {e}")
                else:
                    st.warning(st.session_state.audit_msg)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👈 Upload your client's CSV file in the sidebar to start.")
        st.warning("⚠️ **Pre-Sampling Caution:** Ensure that individually significant (material) and unusual items have been removed for specific testing. This tool is designed for the *remaining* sampling population.")
