import streamlit as st
import pandas as pd
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audit Assistant - MUS Tool", layout="wide")

# --- 1. HELPER FUNCTIONS ---

def clean_currency(x):
    """
    Converts string currency (e.g., "1,500.00") to float (1500.0).
    Handles commas, spaces, and empty values.
    """
    if isinstance(x, (int, float)):
        return x
    if pd.isna(x) or x == '':
        return 0.0
    # Remove commas and spaces
    clean_str = str(x).replace(',', '').replace(' ', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def perform_mus_audit(df, amount_col, interval, random_seed):
    # 1. SETUP & CLEANING
    population = df.copy()
    population[amount_col] = population[amount_col].apply(clean_currency)
    population['Abs_Amount'] = population[amount_col].abs()
    
    # Calculate Cumulative (Running Total)
    population['Cumulative_Balance'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Balance'] - population['Abs_Amount']
    
    total_value = population['Abs_Amount'].sum()
    
    # Validation
    if total_value == 0:
        return None, "Error: Total population value is 0.", {}
    if interval <= 0:
        return None, "Error: Interval must be greater than 0.", {}

    # 2. RANDOM START
    random.seed(random_seed)
    random_start = random.randint(1, int(interval))
    
    # 3. GENERATE HIT POINTS (The "Dollars" we want to select)
    hit_points = []
    current_hit = random_start
    while current_hit <= total_value:
        hit_points.append(current_hit)
        current_hit += interval
        
    # 4. FIND ITEMS CONTAINING THE HIT POINTS
    selection_results = []
    hit_idx = 0
    num_hits = len(hit_points)

    for index, row in population.iterrows():
        low = row['Previous_Cumulative']
        high = row['Cumulative_Balance']
        
        # While the current "hit point" is less than the top of this item's range...
        while hit_idx < num_hits and hit_points[hit_idx] <= high:
            current_target = hit_points[hit_idx]
            
            # If the hit point is also greater than the bottom of this item's range...
            # Then this item "contains" the hit point.
            if current_target > low:
                selection_results.append({
                    'Original_Index': index,
                    'Point_Selected': current_target, # The specific dollar unit
                    'Cumulative_Balance': high
                })
            hit_idx += 1

    # 5. BUILD FINAL RESULT
    if selection_results:
        # Create a dataframe from the findings
        matches_df = pd.DataFrame(selection_results)
        
        # Merge back to get ALL original columns
        # We perform a join to keep the original data structure
        original_rows = df.loc[matches_df['Original_Index']].copy()
        
        # Add the specific Audit Columns requested
        original_rows['Cumulative_Balance'] = matches_df['Cumulative_Balance'].values
        original_rows['Point_Selected'] = matches_df['Point_Selected'].values
        
        # Add metadata for the return
        audit_params = {
            'total_value': total_value,
            'random_seed': random_seed,
            'random_start': random_start,
            'interval': interval,
            'count': len(original_rows)
        }
        
        return original_rows, "Success", audit_params
    else:
        return pd.DataFrame(), "No items selected. Interval too high.", {}

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
        conf_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        tol_misstatement = st.number_input("Tolerable Misstatement ($)", value=50000.0)
        if tol_misstatement > 0:
            lookup = {90: 2.31, 95: 3.00, 99: 4.61}
            final_interval = tol_misstatement / lookup.get(conf_level, 3.00)
            st.write(f"**Calculated Interval:** ${final_interval:,.2f}")

    st.markdown("---")
    st.header("3. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    run_btn = st.button("Run Sampling")

# Main Area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data Preview
        st.write("### Data Preview")
        st.dataframe(df.head(5))
        
        # Column Selection
        all_cols = df.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
             amount_col = st.selectbox("Select Amount Column", all_cols)
        
        # Dynamic Total Calculation
        temp_clean = df[amount_col].apply(clean_currency)
        total_val = temp_clean.abs().sum()
        
        with col2:
            st.metric("Total Population Value", f"${total_val:,.2f}")

        # Recalculate Interval if "Target Sample Size" is chosen
        if method == "Target Sample Size" and total_val > 0:
            final_interval = total_val / target_sample_size
            st.info(f"**Interval:** ${final_interval:,.2f} (Target: {target_sample_size} items)")

        if run_btn:
            if final_interval <= 0:
                st.error("Interval must be greater than 0.")
            else:
                with st.spinner('Calculating MUS...'):
                    result_df, msg, params = perform_mus_audit(df, amount_col, final_interval, random_seed)
                
                if not result_df.empty:
                    st.success(f"Audit Complete: {params['count']} items selected.")
                    
                    # --- NEW: DISPLAY AUDIT PARAMETERS CLEARLY ---
                    st.subheader("📋 Audit Parameters Report")
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("Random Seed", params['random_seed'])
                    p_col2.metric("Random Start", f"{params['random_start']:,}")
                    p_col3.metric("Interval", f"${params['interval']:,.2f}")
                    p_col4.metric("Population Total", f"${params['total_value']:,.2f}")
                    
                    st.divider()
                    
                    st.subheader("✅ Selected Sample Items")
                    st.markdown("The table below includes the **Cumulative Balance** and the specific **Point Selected**.")
                    st.dataframe(result_df)
                    
                    # CSV Download
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Audit Workpaper (CSV)",
                        data=csv,
                        file_name='mus_audit_workpaper.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning(msg)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Upload your client's CSV file in the sidebar to start.")
