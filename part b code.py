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

# Technical Parameters (Matched exactly to your Part A code)
tech_params = {
    'Solar':   {'inv': 425000,  'fom': 0.03, 'life': 25, 'color': "#f1c40f", 'marg': 0.01},
    'Wind':    {'inv': 1182000, 'fom': 0.03, 'life': 25, 'color': "#3498db", 'marg': 0.01},
    'Gas':     {'inv': 400000,  'fom': 0.04, 'life': 30, 'color': "#e67e22", 'eff': 0.39, 'co2': 0.19},
    'Nuclear': {'ann_cap': 6000000, 'marg': 11.5, 'avail': 0.9, 'color': "#e74c3c"},
    'Coal':    {'ann_cap': 1500000, 'marg': 51.0, 'avail': 1.0, 'color': "#7f8c8d", 'co2': 0.34}
}

fuel_cost_gas = 21.6 
vom_gas = 3.0
gas_marginal = (fuel_cost_gas / tech_params['Gas']['eff']) + vom_gas

# --- 2. DATA LOADING ---
print("Loading time series data...")
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
df.index = df.index.tz_localize(None)

results_list = []

# --- 3. OPTIMIZATION LOOP (One run per weather year) ---
for year in YEARS:
    print(f"--- Optimizing Weather Year: {year} ---")
    cz = df.loc[f'{year}-01-01':f'{year}-12-31'].copy().ffill().bfill()
    
    n = pypsa.Network()
    n.set_snapshots(cz.index)
    n.add("Bus", "Czech Republic")
    n.add("Load", "Demand", bus="Czech Republic", p_set=cz['CZ_load_actual_entsoe_transparency'])
    
    # Add Carriers & Generators 
    for name, data in tech_params.items():
        n.add("Carrier", name, color=data['color'], co2_emissions=data.get('co2', 0))
        
        if name in ['Solar', 'Wind', 'Gas']:
            cap_cost = annuity(data['life'], discount_rate) * data['inv'] * (1 + data['fom'])
            marg_cost = gas_marginal if name == 'Gas' else data['marg']
            
            # Normalize CFs based on the specific year's data
            if name == 'Solar':
                p_max = (cz['CZ_solar_generation_actual'] / 2072).clip(0, 1)
            elif name == 'Wind':
                p_max = (cz['CZ_wind_onshore_generation_actual'] / 339).clip(0, 1)
            else: # Gas
                p_max = 1.0
        else:
            # Fixed values for Nuclear/Coal (Part A logic)
            cap_cost = data['ann_cap']
            marg_cost = data['marg']
            p_max = data['avail']

        n.add("Generator", name,
              bus="Czech Republic",
              carrier=name,
              p_nom_extendable=True,
              capital_cost=cap_cost,
              marginal_cost=marg_cost,
              p_max_pu=p_max)

    # Add the Global CO2 Constraint (10 Million tonnes)
    n.add("GlobalConstraint", "co2_limit",
          type="primary_energy",
          carrier_attribute="co2_emissions",
          sense="<=",
          constant=10e6) 

    n.optimize(solver_name='highs')
    
    # Collect optimal capacities
    year_res = n.generators.p_nom_opt.to_dict()
    year_res['Year'] = year
    results_list.append(year_res)

# --- 4. PROCESSING RESULTS ---
results_df = pd.DataFrame(results_list).set_index('Year')
mean_cap = results_df.mean()
cv = (results_df.std() / mean_cap * 100).fillna(0) # Coefficient of Variation

# --- 5. PLOTTING ---
colors = [tech_params[col]['color'] for col in results_df.columns]

# Plot 1: Total Capacity Mix per Year (Stacked Bar)
results_df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), edgecolor='black', alpha=0.8)
plt.title("Optimal Capacity Mix Variation Across Weather Years (10Mt CO2 Cap)")
plt.ylabel("Installed Capacity [MW]")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 2: Weather Sensitivity (Coefficient of Variation)
plt.figure(figsize=(10, 5))
plt.bar(cv.index, cv.values, color=colors, edgecolor='black')
plt.title("Sensitivity of Technology Selection to Weather Variability (CV %)")
plt.ylabel("Coefficient of Variation [%]")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot 3: Capacity Trends (Line Chart)
fig, ax = plt.subplots(figsize=(10, 6))
for gen in results_df.columns:
    ax.plot(YEARS, results_df[gen], marker='o', label=gen, color=tech_params[gen]['color'], linewidth=2.5)
ax.set_title("Capacity Shifts Across Different Weather Years")
ax.set_xticks(YEARS)
ax.legend()
ax.set_ylabel("Installed Capacity [MW]")
plt.grid(True, alpha=0.3)
plt.show()