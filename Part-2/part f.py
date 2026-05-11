import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. SETTINGS & DATA LOADING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "time_series_60min_singleindex.csv")

target_columns = [
    'utc_timestamp',
    'CZ_load_actual_entsoe_transparency',
    'CZ_solar_generation_actual',
    'CZ_wind_onshore_generation_actual'
]

print("Loading and cleaning data...")
df = pd.read_csv(file_path, usecols=target_columns, index_col=0, parse_dates=True)
df.index = df.index.tz_localize(None)

# Slice for 2019
cz_2019 = df.loc['2019-01-01':'2019-12-31'].copy()
cz_2019 = cz_2019.ffill().bfill()

# --- QUICK FIX: RESAMPLING ---
# Resampling to 3-hour blocks reduces variables by 66% while maintaining profile accuracy
cz_2019 = cz_2019.resample('3H').mean()

# Normalization (Capacity Factors) based on 2019 actuals
solar_installed_mw = 2049
wind_installed_mw  = 316
cz_2019['solar_cf'] = (cz_2019['CZ_solar_generation_actual'] / solar_installed_mw).clip(0, 1)
cz_2019['wind_cf']  = (cz_2019['CZ_wind_onshore_generation_actual'] / wind_installed_mw).clip(0, 1)

# --- 2. COST & TECH SETTINGS ---
discount_rate = 0.07

def annuity(n, r):
    if r > 0:
        return r / (1. - 1. / (1. + r)**n)
    else:
        return 1 / n
    
tech_data = {
    'solar':       {'inv': 750000,   'fom': 0.03,  'life': 25, 'color': "#f1c40f", 'marg': 0.01},
    'onshorewind': {'inv': 1240000,  'fom': 0.03,  'life': 25, 'color': "#3498db", 'marg': 0.01},
    'nuclear':     {'inv': 6000000,  'fom': 0.025, 'life': 40, 'color': "#e74c3c", 'marg': 11.5, 'avail': 0.9},
    'coal': {'inv': 1300000, 'fom': 0.02, 'life': 40, 'color': "#7f8c8d", 'marg': 51.0, 'avail': 1.0, 'co2': 0.34, 'eff': 0.35},
    'gas':         {'inv': 400000,   'fom': 0.04,  'life': 30, 'color': "#e67e22", 'eff': 0.39, 'co2': 0.19}
}

fuel_cost_gas = 21.6
vom_gas = 3.0
gas_marginal = (fuel_cost_gas / tech_data['gas']['eff']) + vom_gas

storage_data = {
    'battery': {
        'inv': 600000, 'fom': 0.02, 'life': 15, 'eff_store': 0.95, 
        'eff_dispatch': 0.95, 'max_hours': 4, 'color': "#2ecc71",
    },
    'hydrogen': {
        'inv': 2000000, 'fom': 0.03, 'life': 20, 'eff_store': 0.70, 
        'eff_dispatch': 0.50, 'max_hours': 168, 'color': "#9b59b6",
    }
}

# --- 3. NETWORK BUILDER FUNCTION ---
def build_network():
    n = pypsa.Network()
    n.set_snapshots(cz_2019.index)
    
    # Crucial: Weight snapshots so total annual energy is correct (3 hours per snapshot)
    n.snapshot_weightings[:] = 8760 / len(n.snapshots)

    n.add("Bus", "Czech Republic")

    for tech, data in tech_data.items():
        n.add("Carrier", tech, color=data['color'], co2_emissions=data.get('co2', 0))

    n.add("Load", "Demand", bus="Czech Republic", p_set=cz_2019['CZ_load_actual_entsoe_transparency'])

    for tech, data in tech_data.items():
        capital_cost = data['inv'] * (annuity(data['life'], discount_rate) + data['fom'])
        
        if tech == 'gas':
            marginal_cost = gas_marginal
            p_max_pu = 1.0
        elif tech == 'onshorewind':
            marginal_cost = data['marg']
            p_max_pu = cz_2019['wind_cf']
        elif tech == 'solar':
            marginal_cost = data['marg']
            p_max_pu = cz_2019['solar_cf']
        else:
            marginal_cost = data['marg']
            p_max_pu = data.get('avail', 1.0)

        n.add("Generator", tech,
              bus="Czech Republic",
              carrier=tech,
              p_nom_extendable=True,
              capital_cost=capital_cost,
              marginal_cost=marginal_cost,
              p_max_pu=p_max_pu,
              efficiency=data.get('eff', 1.0))

    for name, s in storage_data.items():
        ann_factor = annuity(s['life'], discount_rate) + s['fom']
        capital_cost = ann_factor * s['inv']
        n.add("StorageUnit", name, bus="Czech Republic", carrier=name,
              capital_cost=capital_cost, marginal_cost=0,
              efficiency_store=s['eff_store'], efficiency_dispatch=s['eff_dispatch'],
              max_hours=s['max_hours'], cyclic_state_of_charge=True, p_nom_extendable=True)
    return n

# --- 4. OPTIMIZATION LOOP ---
print("Running Baseline...")
n = build_network()
n.optimize(solver_name='highs')

# Calculate baseline emissions
base_emissions = (n.generators_t.p / n.generators.efficiency * n.generators.carrier.map(n.carriers.co2_emissions)).sum().sum()

# Add a Global Constraint placeholder
n.add("GlobalConstraint", "co2_limit", carrier_attribute="co2_emissions", sense="<=", constant=base_emissions)

limits_pct = np.linspace(1.0, 0.05, 15) 
results_cap = [] # Changed to collect capacities

for limit in limits_pct:
    # Calculate exact CO2 limit in absolute terms (Tonnes and Megatonnes)
    co2_limit_abs = max(base_emissions * limit, 1.0)
    co2_mt = co2_limit_abs / 1e6
    
    print(f"Optimizing CO2 Cap: {limit*100:.0f}% ({co2_mt:.2f} Mt)")
    
    n.global_constraints.at["co2_limit", "constant"] = co2_limit_abs
    
    status, condition = n.optimize(solver_name='highs')
    if condition != "optimal": continue

    # Extract capacities and convert from MW to GW
    gen_cap = n.generators.p_nom_opt / 1000
    store_cap = n.storage_units.p_nom_opt / 1000
    
    # Combine generation and storage into a single column
    total_cap = pd.concat([gen_cap, store_cap])
    
    # Name the series with the absolute Mt value so it becomes the X-axis label
    total_cap.name = f"{co2_mt:.1f}" 
    results_cap.append(total_cap)


# --- 5. PLOTTING ---
# Build dataframe from results
df_cap = pd.DataFrame(results_cap)

# Filter out technologies that were never built (keeps only techs > 0.01 GW)
df_cap = df_cap.loc[:, df_cap.max() > 0.01]

# Combine color dictionaries to map colors correctly
all_colors = {**{k: v['color'] for k, v in tech_data.items()}, 
              **{k: v['color'] for k, v in storage_data.items()}}
colors = [all_colors[col] for col in df_cap.columns]

# Create the figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the stacked bar chart
df_cap.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black', width=0.85, alpha=0.85)

# --- ADD DATA LABELS INSIDE THE BARS ---
for c in ax.containers:
    # Only label segments that are larger than 0.3 GW (prevents text from overlapping on tiny slices)
    labels = [f"{v.get_height():.1f}" if v.get_height() > 0.3 else "" for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='black', weight='bold')

# Formatting
ax.set_xlabel("CO2 Limit (Mt/year)", fontsize=13)
ax.set_ylabel("Optimal Installed Capacity [GW]", fontsize=13)
ax.set_title("System Capacity Evolution under strict CO2 Constraints", fontsize=15)

# Rotate X-axis labels for better readability since they are now strings of numbers
plt.xticks(rotation=0)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Fix the Legend order so it matches the stack (top to bottom)
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels_leg), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)

plt.tight_layout()
plt.show()