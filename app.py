import streamlit as st
import pandas as pd
import random
import numpy as np

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
    # Create a copy to avoid messing up the display dataframe
    population = df.copy()
    
    # 1. CLEAN THE DATA (Crucial Fix)
    # Apply the cleaning function to the selected column
    population[amount_col] = population[amount_col].apply(clean_currency)
    
    # MUS operates on absolute values (in case of credit notes)
    population['Abs_Amount'] = population[amount_col].abs()
    
    # Calculate Cumulative Amount
    population['Cumulative_Amount'] = population['Abs_Amount'].cumsum()
    population['Previous_Cumulative'] = population['Cumulative_Amount'] - population['Abs_Amount']
    
    total_value = population['Abs_Amount'].sum()
    
    # Safety Check
    if total_value == 0:
        return pd.DataFrame(), "Error: Total population value is 0. Check your data format."
    if interval <= 0:
        return pd.DataFrame(), "Error: Interval must be greater than 0."

    # 2. RANDOM START
    random.seed(random_seed)
    # Ensure start is within the first interval
    random_start = random.randint(1, int(interval))
    
    print(f"Total: {total_value}, Interval: {interval}, Start: {random_start}") # Debugging

    # 3. GENERATE HIT POINTS
    hit_points = []
    current_hit = random_start
    while current_hit <= total_value:
        hit_points.append(current_hit)
        current_hit += interval
        
    # 4. SELECT ITEMS
    selection_results = []
    hit_idx = 0
    num_hits = len(hit_points)

    for index, row in population.iterrows():
        low = row['Previous_Cumulative']
        high = row['Cumulative_Amount']
        
        # Check if any hit points fall within this item's range
        while hit_idx < num_hits and hit_points[hit_idx] <= high:
            current_target = hit_points[hit_idx]
            
            if current_target > low:
                selection_results.append({
                    'Original_Index': index,
                    'Recorded_Amount': row[amount_col], # Show original formatted amount if possible
                    'Selected_By_Hit_Point': current_target,
                    'Item_Type': 'High Value' if row['Abs_Amount'] >= interval else 'Sampled'
                })
            hit_idx += 1

    results_df = pd.DataFrame(selection_results)
    
    if not results_df.empty:
        # Fetch original rows
        final_df = df.loc[results_df['Original_Index']].copy()
        final_df['Audit_Hit_Point'] = results_df['Selected_By_Hit_Point'].values
        final_df['Audit_Note'] = results_df['Item_Type'].values
        return final_df, f"Success: Selected {len(final_df)} items."
    else:
        return pd.DataFrame(), "No items selected. The interval might be too high."

# --- 2. THE APP INTERFACE ---

st.title("🛡️ Audit Assistant: Monetary Unit Sampling")
st.markdown("Perform statutory audit sampling with confidence.")

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    st.header("2. Sampling Settings")
    
    # NEW: Toggle between methods
    method = st.radio("Selection Method:", ["Target Sample Size", "Manual Interval", "Confidence Calculator"])
    
    final_interval = 0.0
    target_sample_size = 0
    
    if method == "Target Sample Size":
        st.info("The app will calculate the Interval needed to get exactly this many items.")
        target_sample_size = st.number_input("Number of Items to Select", value=25, min_value=1)
        
    elif method == "Manual Interval":
        final_interval = st.number_input("Enter Sampling Interval ($)", value=100000.0, step=1000.0)
        
    else: # Confidence Calculator
        st.info("Based on Zero Expected Errors")
        conf_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        tol_misstatement = st.number_input("Tolerable Misstatement ($)", value=50000.0)
        
        if tol_misstatement > 0:
            # R-Factor Lookup
            lookup = {90: 2.31, 95: 3.00, 99: 4.61}
            r_factor = lookup.get(conf_level, 3.00)
            final_interval = tol_misstatement / r_factor
            st.write(f"**Calculated Interval:** ${final_interval:,.2f}")
            st.caption(f"Math: {tol_misstatement} / {r_factor}")

    st.markdown("---")
    st.header("3. Execution")
    random_seed = st.number_input("Random Seed", value=12345, step=1)
    run_btn = st.button("Run Sampling")

# Main Area
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Select Column (Allow all columns initially, in case pandas misidentified types)
        all_cols = df.columns.tolist()
        
        st.write("### Data Preview")
        st.dataframe(df.head(5)) # Showing 5 rows now
        
        col1, col2 = st.columns(2)
        with col1:
             amount_col = st.selectbox("Select Amount Column", all_cols)
        
        # DYNAMIC CALCULATION: Show Total Value immediately
        # We clean the data *just for the display* here to check if it works
        temp_clean = df[amount_col].apply(clean_currency)
        total_val = temp_clean.abs().sum()
        
        with col2:
            st.metric("Total Population Value", f"${total_val:,.2f}")
            if total_val == 0:
                st.error("⚠️ The total is 0. Please select the correct column containing financial amounts.")

        # LOGIC FOR TARGET SAMPLE SIZE
        if method == "Target Sample Size" and total_val > 0:
            # Formula: Interval = Population / Sample Size
            calculated_interval = total_val / target_sample_size
            final_interval = calculated_interval
            st.info(f"**Calculated Interval:** ${final_interval:,.2f} (to get ~{target_sample_size} items)")

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
