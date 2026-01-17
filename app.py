import streamlit as st
import pandas as pd
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide")

# --- 1. HELPER FUNCTIONS ---

def get_reliability_factor(confidence_level, errors=0):
    """
    Returns the Reliability Factor (R-Factor) based on the AICPA Audit Guide.
    For Zero Expected Errors (simplest scenario).
    """
    # Table for 0 errors found (Poisson distribution)
    lookup = {
        90: 2.31,
        95: 3.00,
        99: 4.61
    }
    return lookup.get(confidence_level, 3.00)

def calculate_interval(book_value, tolerable_misstatement, confidence_level):
    """
    Calculates the Sampling Interval assuming 0 expected errors.
    Formula: Interval = Tolerable Misstatement / Reliability Factor
    """
    r_factor = get_reliability_factor(confidence_level)
    if r_factor == 0: return 0
    
    # Basic Formula for MUS with 0 expected misstatements
    return tolerable_misstatement / r_factor

def perform_mus_audit(df, amount_col, interval, random_seed):
    # (Same logic as previous version)
    try:
        df[amount_col] = pd.to_numeric(df[amount_col])
    except:
        return None, "Error: The selected column contains non-numeric data."
    
    population = df.copy()
    population['Abs_Amount'] = population[amount_col].abs()
    population['Cumulative_Amount'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Amount'] - population['Abs_Amount']
    
    total_value = population['Abs_Amount'].sum()
    
    random.seed(random_seed)
    # Protection against interval being larger than total value
    if interval > total_value:
        return pd.DataFrame(), f"Error: Interval ({interval}) is larger than the total population value ({total_value})."
        
    random_start = random.randint(1, int(interval))
    
    hit_points = []
    current_hit = random_start
    while current_hit <= total_value:
        hit_points.append(current_hit)
        current_hit += interval
        
    selection_results = []
    hit_idx = 0
    num_hits = len(hit_points)

    for index, row in population.iterrows():
        low = row['Previous_Cumulative']
        high = row['Cumulative_Amount']
        
        while hit_idx < num_hits and hit_points[hit_idx] <= high:
            current_target = hit_points[hit_idx]
            if current_target > low:
                selection_results.append({
                    'Original_Index': index,
                    'Recorded_Amount': row[amount_col],
                    'Selected_By_Hit_Point': current_target,
                    'Item_Type': 'High Value' if row['Abs_Amount'] >= interval else 'Sampled'
                })
            hit_idx += 1

    results_df = pd.DataFrame(selection_results)
    
    if not results_df.empty:
        final_df = df.loc[results_df['Original_Index']].copy()
        final_df['Audit_Hit_Point'] = results_df['Selected_By_Hit_Point'].values
        final_df['Audit_Note'] = results_df['Item_Type'].values
        return final_df, f"Success: Selected {len(final_df)} items."
    else:
        return pd.DataFrame(), "No items selected."

# --- 2. THE APP INTERFACE ---

st.title("🛡️ Audit Assistant: Monetary Unit Sampling")
st.markdown("Perform statutory audit sampling with confidence.")

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    st.header("2. Determine Interval")
    
    calc_method = st.radio("Interval Method:", ["Manual Entry", "Calculate for Me"])
    
    final_interval = 0.0
    
    if calc_method == "Manual Entry":
        final_interval = st.number_input("Enter Sampling Interval ($)", value=100000.0, step=1000.0)
    else:
        st.info("Based on Zero Expected Errors")
        conf_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        tol_misstatement = st.number_input("Tolerable Misstatement ($)", value=50000.0)
        
        if tol_misstatement > 0:
            calc_interval = calculate_interval(0, tol_misstatement, conf_level) # Book value irrelevant for Interval calc in simple method
            st.write(f"**Calculated Interval:** ${calc_interval:,.2f}")
            final_interval = calc_interval
            
            # Show the math
            r_factor = get_reliability_factor(conf_level)
            st.caption(f"Math: {tol_misstatement:,.0f} / {r_factor} (R-Factor)")
        else:
            st.warning("Enter Tolerable Misstatement > 0")

    st.markdown("---")
    st.header("3. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    run_btn = st.button("Run Sampling")

# Main Area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Select Column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Data Preview")
                st.dataframe(df.head(3))
            
            with col2:
                amount_col = st.selectbox("Select Amount Column", numeric_cols)
                total_val = df[amount_col].abs().sum()
                st.metric("Total Population Value", f"${total_val:,.2f}")

            if run_btn:
                if final_interval <= 0:
                    st.error("Interval must be greater than 0.")
                else:
                    with st.spinner('Running MUS Algorithm...'):
                        result_df, message = perform_mus_audit(df, amount_col, final_interval, random_seed)
                    
                    if not result_df.empty:
                        st.success(message)
                        st.dataframe(result_df)
                        
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Workpaper",
                            data=csv,
                            file_name='mus_audit_sample.csv',
                            mime='text/csv',
                        )
                    else:
                        st.warning(message)

    except Exception as e:
        st.error(f"Error: {e}")
else:

    st.info("👈 Upload your client's CSV file in the sidebar to start.")
