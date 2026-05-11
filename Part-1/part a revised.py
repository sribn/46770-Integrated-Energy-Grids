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
solar_installed_mw = 2049 #https://transparency.entsoe.eu/generation/installed/perType?appState=%7B%22sa%22%3A%5B%22BZN%7C10YCZ-CEPS-----N%22%5D%2C%22st%22%3A%22BZN%22%2C%22mm%22%3Atrue%2C%22ma%22%3Afalse%2C%22sp%22%3A%22HALF%22%2C%22dt%22%3A%22CHART%22%2C%22df%22%3A%5B%222019%22%2C%222019%22%5D%2C%22tz%22%3A%22CET%22%7D
wind_installed_mw  = 316 #same
cz_2019['solar_cf'] = (cz_2019['CZ_solar_generation_actual'] / solar_installed_mw).clip(0, 1)
cz_2019['wind_cf']  = (cz_2019['CZ_wind_onshore_generation_actual'] / wind_installed_mw).clip(0, 1)

# --- 2. COST CALCULATION (Source: DIW Berlin 2013) ---
def annuity(n, r):
    if r > 0:
        return r / (1. - 1. / (1. + r)**n)
    else:
        return 1 / n

discount_rate = 0.07

# Technical Data Dictionary - UPDATED with REAL DIW COSTS
tech_data = {
    'solar':       {'inv': 750000,   'fom': 0.03,  'life': 25, 'color': "#f1c40f", 'marg': 0.01},
    'onshorewind': {'inv': 1240000,  'fom': 0.03,  'life': 25, 'color': "#3498db", 'marg': 0.01},
    'nuclear':     {'inv': 6000000,  'fom': 0.025, 'life': 40, 'color': "#e74c3c", 'marg': 11.5, 'avail': 0.9},
    'coal':        {'inv': 1300000,  'fom': 0.02,  'life': 40, 'color': "#7f8c8d", 'marg': 51.0, 'avail': 1.0, 'co2': 0.34},
    'gas':         {'inv': 400000,   'fom': 0.04,  'life': 30, 'color': "#e67e22", 'eff': 0.39, 'co2': 0.19}
}
#https://www.diw.de/documents/publikationen/73/diw_01.c.424566.de/diw_datadoc_2013-068.pdf

fuel_cost_gas = 21.6 #https://www.sciencedirect.com/science/article/abs/pii/S036054421831288X?fr=RR-2&ref=pdf_download&rr=9ea337d92f634f62
vom_gas = 3.0        #same
gas_marginal = (fuel_cost_gas / tech_data['gas']['eff']) + vom_gas

# --- 3. NETWORK INITIALIZATION ---
n = pypsa.Network()
n.set_snapshots(cz_2019.index)
n.add("Bus", "Czech Republic")

for tech, data in tech_data.items():
    n.add("Carrier", tech, color=data['color'], co2_emissions=data.get('co2', 0))

# --- 4. ADDING COMPONENTS ---
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
    else: # Nuclear and Coal
        marginal_cost = data['marg']
        p_max_pu = data['avail']

    n.add("Generator", tech,
          bus="Czech Republic",
          carrier=tech,
          p_nom_extendable=True,
          capital_cost=capital_cost,
          marginal_cost=marginal_cost,
          p_max_pu=p_max_pu)

# --- 5. OPTIMIZATION ---
print("Running optimization...")
n.optimize(solver_name='highs')

# --- NEW: FILTER UNBUILT TECHNOLOGIES ---
# Identify only the technologies that were actually built (capacity > 0.01 MW)
active_gens = n.generators.index[n.generators.p_nom_opt > 0.01].tolist()
# Get the correct colors only for the active technologies
active_colors = [tech_data[g]['color'] for g in active_gens]


# --- 6. ALL PLOTS (RESTORED & FILTERED) ---

# Plot A: Dispatch Time Series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Only plot the active_gens
n.generators_t.p[active_gens].loc['2019-01-01':'2019-01-07'].plot.area(ax=ax1, color=active_colors, title="Winter Week Dispatch (CZ 2019)")
ax1.plot(n.loads_t.p_set.loc['2019-01-01':'2019-01-07', 'Demand'], color='green', linewidth=2, linestyle='--', label='Demand')
ax1.set_ylabel("Power [MW]")
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1)) # <--- MOVED CLOSER

n.generators_t.p[active_gens].loc['2019-07-01':'2019-07-07'].plot.area(ax=ax2, color=active_colors, title="Summer Week Dispatch (CZ 2019)")
ax2.plot(n.loads_t.p_set.loc['2019-07-01':'2019-07-07', 'Demand'], color='green', linewidth=2, linestyle='--', label='Demand')
ax2.set_ylabel("Power [MW]")
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1)) # <--- MOVED CLOSER

plt.tight_layout() # <--- FORCES PYTHON TO FIT EVERYTHING
plt.subplots_adjust(right=0.85) # <--- LEAVES 15% EMPTY SPACE ON THE RIGHT FOR THE LEGEND
plt.show()

# Plot B: Annual Electricity Mix
plt.figure(figsize=(8, 8))
# Sum only the active generators so 0% slices are completely removed
n.generators_t.p[active_gens].sum().plot(kind='pie', autopct='%1.1f%%', colors=active_colors, title="Optimal Annual Electricity Mix")
plt.ylabel("")
plt.show()

# Plot C: Generation Duration Curves
fig, ax = plt.subplots(figsize=(10, 6))
# Loop ONLY through the active generators
for col in active_gens:
    sorted_gen = n.generators_t.p[col].sort_values(ascending=False).values
    ax.plot(sorted_gen, label=col, color=tech_data[col]['color'])

sorted_load = n.loads_t.p_set['Demand'].sort_values(ascending=False).values
ax.plot(sorted_load, color='black', linewidth=2, linestyle='--', label='Load')

# --- NEW AXIS LABELS ADDED HERE ---
ax.set_title("Generation Duration Curves", fontsize=14)
ax.set_ylabel("Power [MW]", fontsize=12)
ax.set_xlabel("Hours of the Year", fontsize=12)
ax.grid(True, alpha=0.2) # Added a light grid so it's easier to read!

ax.legend()
plt.tight_layout() # Keeps the labels from getting cut off
plt.show()

# Plot D: Price Duration Curve (Cleaned and Zoomed)
prices = n.buses_t.marginal_price["Czech Republic"]
prices_sorted = prices.sort_values(ascending=False).values

fig, ax = plt.subplots(figsize=(12, 7))

# Plot the actual market prices
ax.plot(prices_sorted, color='purple', linewidth=3, label='Market Clearing Price (Result)', zorder=5)

# Add reference lines for ALL technologies in your tech_data
for tech, data in tech_data.items():
    if tech == 'gas':
        m_cost = gas_marginal
    else:
        m_cost = data['marg']
    
    ax.axhline(y=m_cost, color=data['color'], linestyle='--', alpha=0.8, 
               label=f"{tech.capitalize()} Marginal Cost ({m_cost:.2f})")

ax.set_xlim(left=0, right=len(prices_sorted)) 
ax.set_ylim([-5, 75])                         

ax.set_title("Price Duration Curve: Marginal Cost Identification", fontsize=14)
ax.set_ylabel("Price [EUR/MWh]", fontsize=12)
ax.set_xlabel("Hours of the Year", fontsize=12)
ax.grid(True, alpha=0.2)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# --- 7. FINAL RESULTS & 60MW VERIFICATION ---
print(f"\nTotal Annual System Cost: {n.objective / 1e6:.2f} Million EUR")
for gen in n.generators.index:
    print(f"Optimal Capacity for {gen}: {n.generators.at[gen, 'p_nom_opt']:.2f} MW")

print("\n" + "="*50)
print("PEAK CAPACITY VERIFICATION (Explaining the 60.00 MW)")
print("="*50)
res_load = n.loads_t.p_set['Demand'] - \
           (n.generators_t.p_max_pu['onshorewind'] * n.generators.at['onshorewind', 'p_nom_opt']) - \
           (n.generators_t.p_max_pu['solar'] * n.generators.at['solar', 'p_nom_opt'])

peak_value = res_load.max()
print(f"Max Residual Load Needed: {peak_value:.2f} MW")
print(f"Total Coal + Gas Provided: {(n.generators.at['coal', 'p_nom_opt'] + n.generators.at['gas', 'p_nom_opt']):.2f} MW")
print("="*50)