import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import os

# --- 1. SETTINGS & CONSISTENT DATA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "time_series_60min_singleindex.csv")

# We use 5 different years to test weather sensitivity
YEARS = [2015, 2016, 2017, 2018, 2019]
discount_rate = 0.07

def annuity(n, r):
    """Annuity factor for consistent cost calculation with Part A"""
    return r / (1. - 1. / (1. + r)**n) if r > 0 else 1 / n

# Technical Parameters (MATCHED EXACTLY TO PART A) 2020
tech_params = {
    'solar':       {'inv': 750000,   'fom': 0.03,  'life': 25, 'color': "#f1c40f", 'marg': 0.01},
    'onshorewind': {'inv': 1240000,  'fom': 0.03,  'life': 25, 'color': "#3498db", 'marg': 0.01},
    'nuclear':     {'inv': 6000000,  'fom': 0.025, 'life': 40, 'color': "#e74c3c", 'marg': 11.5, 'avail': 0.9},
    'coal':        {'inv': 1300000,  'fom': 0.02,  'life': 40, 'color': "#7f8c8d", 'marg': 51.0, 'avail': 1.0, 'co2': 0.34},
    'gas':         {'inv': 400000,   'fom': 0.04,  'life': 30, 'color': "#e67e22", 'eff': 0.39, 'co2': 0.19}
}

# Marginal cost for Gas based on Part A formula
fuel_cost_gas = 21.6 
vom_gas = 3.0
gas_marginal = (fuel_cost_gas / tech_params['gas']['eff']) + vom_gas

# --- 2. DATA LOADING ---
print("Loading time series data...")
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
df.index = df.index.tz_localize(None)

results_list = []

# --- EXTRACT BASELINE DEMAND ---
# Freezing the load profile to 2019 to strictly isolate weather/supply variations
print("Extracting baseline demand (2019) to keep load constant...")
baseline_demand = df.loc['2019-01-01':'2019-12-31', 'CZ_load_actual_entsoe_transparency'].copy().ffill().bfill().values

# --- 3. OPTIMIZATION LOOP (One run per weather year) ---
for year in YEARS:
    print(f"--- Optimizing Weather Year: {year} ---")
    cz = df.loc[f'{year}-01-01':f'{year}-12-31'].copy().ffill().bfill()
    
    n = pypsa.Network()
    n.set_snapshots(cz.index)
    n.add("Bus", "Czech Republic")
    
    # Handle leap year mismatches safely
    current_length = len(cz)
    if current_length > len(baseline_demand):
        demand_array = np.pad(baseline_demand, (0, current_length - len(baseline_demand)), 'edge')
    else:
        demand_array = baseline_demand[:current_length]

    n.add("Load", "Demand", bus="Czech Republic", p_set=demand_array)
    
    # Add Carriers & Generators 
    for name, data in tech_params.items():
        n.add("Carrier", name, color=data['color'])
        
        # Calculate annualized capital cost
        capital_cost = data['inv'] * (annuity(data['life'], discount_rate) + data['fom'])
        
        if name == 'gas':
            marg_cost = gas_marginal
            p_max = 1.0
        elif name == 'onshorewind':
            marg_cost = data['marg']
            p_max = (cz['CZ_wind_onshore_generation_actual'] / 339).clip(0, 1)
        elif name == 'solar':
            marg_cost = data['marg']
            p_max = (cz['CZ_solar_generation_actual'] / 2072).clip(0, 1)
        else: # Nuclear and Coal
            marg_cost = data['marg']
            p_max = data['avail']

        n.add("Generator", name,
              bus="Czech Republic",
              carrier=name,
              p_nom_extendable=True,
              capital_cost=capital_cost,
              marginal_cost=marg_cost,
              p_max_pu=p_max)

    # --- REMOVED GLOBAL CO2 CONSTRAINT ---
    # Optimization now runs purely on economic merit order as per Part A
    n.optimize(solver_name='highs')
    
    # Collect optimal capacities
    year_res = n.generators.p_nom_opt.to_dict()
    year_res['Year'] = year
    results_list.append(year_res)

# --- 4. PROCESSING RESULTS ---
results_df = pd.DataFrame(results_list).set_index('Year')

# --- NEW: FILTER OUT 0 CAPACITY TECHNOLOGIES ---
# Keep only columns (technologies) where the maximum capacity across the 5 years was > 0.01 MW
results_df = results_df.loc[:, results_df.max() > 0.01]

mean_cap = results_df.mean()
cv = (results_df.std() / mean_cap * 100).fillna(0) # Coefficient of Variation

# --- 5. PLOTTING ---
# The colors list now automatically adapts to only include the surviving technologies
colors = [tech_params[col]['color'] for col in results_df.columns]

# Plot 1: Total Capacity Mix per Year (Stacked Bar)
results_df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), edgecolor='black', alpha=0.8)
plt.title("Optimal Capacity Mix Variation Across Weather Years (No CO2 Cap)", fontsize=14)
plt.ylabel("Installed Capacity [MW]", fontsize=14)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

# Plot 2: Weather Sensitivity (Coefficient of Variation)
plt.figure(figsize=(10, 5))
plt.bar(cv.index, cv.values, color=colors, edgecolor='black')
plt.title("Sensitivity of Technology Selection to Weather Variability (CV %)", fontsize=14)
plt.ylabel("Coefficient of Variation [%]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 3: Capacity Trends (Line Chart)
fig, ax = plt.subplots(figsize=(10, 6))
for gen in results_df.columns:
    ax.plot(YEARS, results_df[gen], marker='o', label=gen, color=tech_params[gen]['color'], linewidth=2.5)

ax.set_title("Capacity Shifts Across Different Weather Years (Constant Demand)", fontsize=14)
ax.set_xticks(YEARS)
ax.set_ylabel("Installed Capacity [MW]", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()