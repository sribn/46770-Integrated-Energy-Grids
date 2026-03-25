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

# Normalization (Capacity Factors) based on 2019 actuals
solar_installed_mw = 2072 
wind_installed_mw  = 339  
cz_2019['solar_cf'] = (cz_2019['CZ_solar_generation_actual'] / solar_installed_mw).clip(0, 1)
cz_2019['wind_cf']  = (cz_2019['CZ_wind_onshore_generation_actual'] / wind_installed_mw).clip(0, 1)

# --- 2. COST CALCULATION (Tutorial Scheme) ---
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n and discount rate r."""
    if r > 0:
        return r / (1. - 1. / (1. + r)**n)
    else:
        return 1 / n

discount_rate = 0.07

# Technical Data Dictionary
# inv/fom/life for annuity calculation (Solar, Wind, Gas)
# ann_cap/marg for direct entry (Nuclear, Coal)
tech_data = {
    'solar':       {'inv': 425000,  'fom': 0.03, 'life': 25, 'color': "#f1c40f", 'type': 'calc'},
    'onshorewind': {'inv': 1182000, 'fom': 0.03, 'life': 25, 'color': "#3498db", 'type': 'calc'},
    'gas':         {'inv': 400000,  'fom': 0.04, 'life': 30, 'color': "#e67e22", 'eff': 0.39, 'co2': 0.19, 'type': 'calc'},
    'nuclear':     {'ann_cap': 6000000, 'marg': 11.5, 'avail': 0.9, 'color': "#e74c3c", 'type': 'fixed'},
    'coal':        {'ann_cap': 1500000, 'marg': 51.0, 'avail': 1.0, 'color': "#7f8c8d", 'co2': 0.34, 'type': 'fixed'}
}

fuel_cost_gas = 21.6 # EUR/MWh_th
vom_gas = 3.0        # EUR/MWh_el

# --- 3. NETWORK INITIALIZATION ---
n = pypsa.Network()
n.set_snapshots(cz_2019.index)
n.add("Bus", "Czech Republic")

# Add Carriers with CO2 emission factors (tCO2/MWh_thermal)
for tech, data in tech_data.items():
    co2 = data.get('co2', 0)
    n.add("Carrier", tech, color=data['color'], co2_emissions=co2)

# --- 4. ADDING COMPONENTS ---

# Demand
n.add("Load", "Demand", bus="Czech Republic", p_set=cz_2019['CZ_load_actual_entsoe_transparency'])

# Generators
for tech, data in tech_data.items():
    if data['type'] == 'calc':
        # Calculate annualized cost for new technologies
        capital_cost = annuity(data['life'], discount_rate) * data['inv'] * (1 + data['fom'])
        if tech == 'gas':
            marginal_cost = (fuel_cost_gas / data['eff']) + vom_gas
            p_max_pu = 1.0
        elif tech == 'onshorewind':
            marginal_cost = 0.01
            p_max_pu = cz_2019['wind_cf']
        else: # solar
            marginal_cost = 0.01
            p_max_pu = cz_2019['solar_cf']
    else:
        # Use your original fixed values for Nuclear and Coal
        capital_cost = data['ann_cap']
        marginal_cost = data['marg']
        p_max_pu = data['avail']

    n.add("Generator", tech,
          bus="Czech Republic",
          carrier=tech,
          p_nom_extendable=True,
          capital_cost=capital_cost,
          marginal_cost=marginal_cost,
          p_max_pu=p_max_pu)

# --- 5. GLOBAL CONSTRAINT (CO2 Limit) ---
# This forces the system to avoid "only gas/coal" by capping total emissions
# 10 Million tonnes is a reasonable starting point for CZ
n.add("GlobalConstraint", "co2_limit",
      type="primary_energy",
      carrier_attribute="co2_emissions",
      sense="<=",
      constant=10e6) 

# --- 6. OPTIMIZATION ---
print("Running optimization...")
n.optimize(solver_name='highs')

# --- 7. PLOTTING ---

# Plot A: Dispatch Time Series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
colors = [n.carriers.at[c, "color"] for c in n.generators.carrier]

n.generators_t.p.loc['2019-01-01':'2019-01-07'].plot.area(ax=ax1, color=colors, title="Winter Week Dispatch (CZ 2019)")
ax1.plot(n.loads_t.p_set.loc['2019-01-01':'2019-01-07', 'Demand'], color='black', linewidth=2, linestyle='--', label='Demand')
ax1.set_ylabel("Power [MW]")
ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

n.generators_t.p.loc['2019-07-01':'2019-07-07'].plot.area(ax=ax2, color=colors, title="Summer Week Dispatch (CZ 2019)")
ax2.plot(n.loads_t.p_set.loc['2019-07-01':'2019-07-07', 'Demand'], color='black', linewidth=2, linestyle='--', label='Demand')
ax2.set_ylabel("Power [MW]")
ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

fig.autofmt_xdate()
plt.subplots_adjust(hspace=0.4, right=0.85) 
plt.show()

# Plot B: Annual Electricity Mix
plt.figure(figsize=(8, 8))
n.generators_t.p.sum().plot(kind='pie', autopct='%1.1f%%', colors=colors, title="Optimal Annual Electricity Mix")
plt.ylabel("")
plt.show()

# Plot C: Generation Duration Curves
fig, ax = plt.subplots(figsize=(10, 6))
for i, col in enumerate(n.generators_t.p.columns):
    sorted_gen = n.generators_t.p[col].sort_values(ascending=False).values
    ax.plot(sorted_gen, label=col, color=colors[i])

sorted_load = n.loads_t.p_set['Demand'].sort_values(ascending=False).values
ax.plot(sorted_load, color='black', linewidth=2, linestyle='--', label='Load')
ax.set_title("Generation Duration Curves")
ax.legend()
plt.show()

# Plot D: Price Duration Curve
prices = n.buses_t.marginal_price["Czech Republic"]
prices_sorted = prices.sort_values(ascending=False).values

plt.figure(figsize=(10, 6))
plt.plot(prices_sorted, color='purple', linewidth=2)
plt.title("Price Duration Curve (Market Clearing Price)")
plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Hours of the Year")
plt.grid(True, alpha=0.3)
plt.show()

# --- 8. PRINT RESULTS ---
print(f"\nTotal Annual System Cost: {n.objective / 1e6:.2f} Million EUR")
for gen in n.generators.index:
    print(f"Optimal Capacity for {gen}: {n.generators.at[gen, 'p_nom_opt']:.2f} MW")