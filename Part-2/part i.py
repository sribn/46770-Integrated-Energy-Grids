import os
import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#data load and Preprocessing
def annuity(n_years, r):
    """Annuity factor (capital-recovery factor)."""
    return r / (1 - 1 / (1 + r) ** n_years) if r > 0 else 1 / n_years


CSV_NAME = "time_series_60min_singleindex (1).csv" 

try:
    _base = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _base = os.getcwd()

file_path = os.path.join(_base, CSV_NAME)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"\nCSV not found at:\n  {file_path}\n\n"
        f"Please either:\n"
        f"  1) Move '{CSV_NAME}' into the same folder as part_i.py, OR\n"
        f"  2) Set CSV_NAME above to the full absolute path, e.g.:\n"
        f"     CSV_NAME = r'C:\\Users\\kusha\\Downloads\\time_series_60min_singleindex.csv'"
    )

COUNTRIES = ["CZ", "DE", "AT", "PL"]

target_cols = ["utc_timestamp"]
for c in COUNTRIES:
    target_cols += [
        f"{c}_load_actual_entsoe_transparency",
        f"{c}_solar_generation_actual",
        f"{c}_wind_onshore_generation_actual",
    ]

print("Loading data …")
df = pd.read_csv(file_path, usecols=target_cols, index_col=0, parse_dates=True)
df.index = df.index.tz_localize(None)
df = df.loc["2019-01-01":"2019-12-31"].copy().ffill().bfill().fillna(0)

# 3-hourly resampling → 2920 snapshots (manageable for HiGHS)
df = df.resample("3H").mean()
snapshots = df.index

#ELECTRICITY TECH DATA 
DISCOUNT_RATE = 0.07

ELEC_TECH = {
    "solar":   {"inv": 750_000,   "fom": 0.030, "life": 25, "marg": 0.01, "co2": 0.00, "color": "#f1c40f"},
    "wind":    {"inv": 1_240_000, "fom": 0.030, "life": 25, "marg": 0.01, "co2": 0.00, "color": "#3498db"},
    "nuclear": {"inv": 6_000_000, "fom": 0.025, "life": 40, "marg": 11.5, "avail": 0.9, "co2": 0.00, "color": "#e74c3c"},
    "coal":    {"inv": 1_300_000, "fom": 0.020, "life": 40, "marg": 51.0, "avail": 1.0, "co2": 0.34, "color": "#7f8c8d"},
}

FUEL_COST_GAS = 21.6   # EUR/MWh_th
VOM_GAS       = 3.0    # EUR/MWh_el
GAS_EFF       = 0.39
GAS_CO2       = 0.19   # t_CO2 / MWh_th

#installed capacities for capacity-factor normalisation
INSTALLED = {
    "CZ": {"solar": 2_049,  "wind":    316},
    "DE": {"solar": 45_435, "wind": 52_946},
    "AT": {"solar":  1_193, "wind":  3_035},
    "PL": {"solar":    430, "wind":  5_808},
}

for c in COUNTRIES:
    df[f"{c}_solar_cf"] = (df[f"{c}_solar_generation_actual"] / INSTALLED[c]["solar"]).clip(0, 1)
    df[f"{c}_wind_cf"]  = (df[f"{c}_wind_onshore_generation_actual"]  / INSTALLED[c]["wind"]).clip(0, 1)

#HEATING DEMAND & COP
ANNUAL_HEAT_GWh = {"CZ": 100_000, "DE": 600_000, "AT": 70_000, "PL": 250_000}
TEMP_PARAMS = {
    "CZ": {"T_mean": 10, "T_amp": 12},
    "DE": {"T_mean": 10, "T_amp": 10},
    "AT": {"T_mean":  8, "T_amp": 12},
    "PL": {"T_mean":  9, "T_amp": 13},
}

T_THRESHOLD = 17   
T_SINK      = 55  

SNAPSHOT_WEIGHT = 8760 / len(snapshots) 

for c in COUNTRIES:
    doy = snapshots.dayofyear.values
    T   = TEMP_PARAMS[c]["T_mean"] - TEMP_PARAMS[c]["T_amp"] * np.cos(2 * np.pi * doy / 365)

    # Heating Degree Hours profile
    HDH = np.maximum(T_THRESHOLD - T, 0)

    annual_MWh = ANNUAL_HEAT_GWh[c] * 1e3          
    hw_annual  = 0.15 * annual_MWh
    sh_annual  = annual_MWh - hw_annual

    hdh_sum    = HDH.sum() * SNAPSHOT_WEIGHT        
    scale      = sh_annual / hdh_sum

    heat_demand = scale * HDH + (hw_annual / 8760)  
    df[f"{c}_heat_demand"] = heat_demand

    dT  = T_SINK - T
    COP = np.clip(6.81 - 0.121 * dT + 0.00063 * dT**2, 1.5, 6.0)
    df[f"{c}_COP"] = COP

print("Heating demand summary [GWh/year]:")
for c in COUNTRIES:
    total = df[f"{c}_heat_demand"].sum() * SNAPSHOT_WEIGHT / 1e3
    print(f"  {c}: {total:.0f} GWh  (target: {ANNUAL_HEAT_GWh[c]} GWh)")

print("\nMean COP by country:")
for c in COUNTRIES:
    print(f"  {c}: {df[f'{c}_COP'].mean():.2f}")

#CAPITAL COSTS (heating technologies)
def heat_capex(inv, fom, life):
    return inv * (annuity(life, DISCOUNT_RATE) + fom)

CAPEX = {
    "heat_pump":  heat_capex(1_500_000, 0.030, 20),   # EUR/MW_heat
    "gas_boiler": heat_capex(  200_000, 0.015, 20),   # EUR/MW_heat
    "resistive":  heat_capex(  100_000, 0.010, 25),   # EUR/MW_heat
    "TES":        heat_capex(   30_000, 0.010, 20),   # EUR/MWh_heat  (energy capital cost)
    "gas_plant":  400_000 * (annuity(30, DISCOUNT_RATE) + 0.04),  # EUR/MW_el
}
for k, v in ELEC_TECH.items():
    CAPEX[k] = v["inv"] * (annuity(v["life"], DISCOUNT_RATE) + v["fom"])

#BUILD NETWORK
def build_network():
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.snapshot_weightings[:] = SNAPSHOT_WEIGHT

    #Buses
    n.add("Bus", COUNTRIES)                          
    n.add("Bus", "gas_hub", carrier="gas")           
    for c in COUNTRIES:
        n.add("Bus", f"{c}_heat", carrier="heat")    

    #Carriers
    carrier_cfg = {
        "solar":      (0.00, "#f1c40f"),
        "wind":       (0.00, "#3498db"),
        "nuclear":    (0.00, "#e74c3c"),
        "coal":       (0.34, "#7f8c8d"),
        "gas":        (0.19, "#e67e22"),
        "heat":       (0.00, "#c0392b"),
        "heat pump":  (0.00, "#2ecc71"),
        "gas boiler": (0.00, "#d35400"),
        "resistive":  (0.00, "#f39c12"),
        "TES":        (0.00, "#95a5a6"),
    }
    for name, (co2, color) in carrier_cfg.items():
        n.add("Carrier", name, co2_emissions=co2, color=color)

    #Electricity loads
    for c in COUNTRIES:
        n.add("Load", f"{c}_elec",
              bus=c,
              p_set=df[f"{c}_load_actual_entsoe_transparency"])

    #Heat loads 
    for c in COUNTRIES:
        n.add("Load", f"{c}_heat_load",
              bus=f"{c}_heat",
              p_set=df[f"{c}_heat_demand"])

    #Electricity generators
    for c in COUNTRIES:
        for tech, d in ELEC_TECH.items():
            p_max = df[f"{c}_{tech}_cf"] if tech in ("solar", "wind") else d.get("avail", 1.0)
            n.add("Generator", f"{c}_{tech}",
                  bus=c, carrier=tech,
                  p_nom_extendable=True,
                  capital_cost=CAPEX[tech],
                  marginal_cost=d["marg"],
                  p_max_pu=p_max)

    
    n.add("Generator", "gas_supply",
          bus="gas_hub", carrier="gas",
          p_nom_extendable=True,
          marginal_cost=FUEL_COST_GAS)

    for c in COUNTRIES:
        n.add("Link", f"{c}_gas_power",
              bus0="gas_hub", bus1=c,
              carrier="gas",
              efficiency=GAS_EFF,
              p_nom_extendable=True,
              capital_cost=CAPEX["gas_plant"],
              marginal_cost=VOM_GAS)

    for c in COUNTRIES:

        n.add("Link", f"{c}_heat_pump",
              bus0=c, bus1=f"{c}_heat",
              carrier="heat pump",
              efficiency=df[f"{c}_COP"],          
              p_nom_extendable=True,
              capital_cost=CAPEX["heat_pump"],
              marginal_cost=0.5)

        n.add("Link", f"{c}_gas_boiler",
              bus0="gas_hub", bus1=f"{c}_heat",
              carrier="gas boiler",
              efficiency=0.90,
              p_nom_extendable=True,
              capital_cost=CAPEX["gas_boiler"],
              marginal_cost=VOM_GAS)

        n.add("Link", f"{c}_resistive",
              bus0=c, bus1=f"{c}_heat",
              carrier="resistive",
              efficiency=1.0,
              p_nom_extendable=True,
              capital_cost=CAPEX["resistive"],
              marginal_cost=0.5)

        n.add("StorageUnit", f"{c}_TES",
              bus=f"{c}_heat", carrier="TES",
              capital_cost=CAPEX["TES"],
              marginal_cost=0,
              efficiency_store=0.95,
              efficiency_dispatch=0.95,
              standing_loss=0.015,
              max_hours=8,
              cyclic_state_of_charge=True,
              p_nom_extendable=True)

    elec_lines = [
        ("CZ", "DE", 3_500),
        ("CZ", "AT", 2_500),
        ("CZ", "PL", 2_500),
        ("DE", "PL", 3_000),
    ]
    for b0, b1, cap in elec_lines:
        n.add("Line", f"{b0}-{b1}",
              bus0=b0, bus1=b1,
              x=1, s_nom=cap)

    return n

#OPTIMISE
print("\n─── Step 1: unconstrained baseline (to compute CO2 budget) ───")
n_base = build_network()
n_base.optimize(solver_name="highs")

def compute_co2(net):
    co2 = 0.0
    if "gas_supply" in net.generators.index:
        gas_gen = net.generators_t.p["gas_supply"].sum() * SNAPSHOT_WEIGHT
        co2 += gas_gen * GAS_CO2
    for link in net.links.index:
        if "gas_power" in link or "gas_boiler" in link:
            gas_input = net.links_t.p0[link].sum() * SNAPSHOT_WEIGHT
            co2 += gas_input * GAS_CO2
    for g in net.generators.index:
        if net.generators.at[g, "carrier"] == "coal":
            co2 += net.generators_t.p[g].sum() * SNAPSHOT_WEIGHT * 0.34
    return co2

baseline_co2 = compute_co2(n_base)
print(f"Baseline CO2: {baseline_co2/1e6:.2f} Mt/year")

CO2_LIMIT = 0.50 * baseline_co2
print(f"CO2 limit (50 % reduction): {CO2_LIMIT/1e6:.2f} Mt/year")

print("\n─── Step 2: sector-coupled optimisation with CO2 constraint ───")
n = build_network()

n.add("GlobalConstraint", "co2_limit",
      carrier_attribute="co2_emissions",
      sense="<=",
      constant=CO2_LIMIT)

# NOTE: carrier_attribute="co2_emissions" uses n.carriers.co2_emissions per MWh
# of primary fuel consumed. Gas carrier has co2=0.19 t/MWh_th.
# PyPSA computes emissions as: sum_g( p_g * co2_g / eff_g ) * weight

status, condition = n.optimize(solver_name="highs")
print(f"Solver status: {status} | {condition}")

#RESULTS ANALYSIS
print("\n═══ RESULTS ═══")
print(f"Total system cost: {n.objective/1e9:.3f} B€/year")

#Electricity capacities 
print("\nElectricity generator capacities [GW]:")
elec_caps = n.generators.p_nom_opt / 1000
print(elec_caps[elec_caps > 0.001].round(2))

print("\nHeat technology capacities [GW_heat]:")
heat_links = [l for l in n.links.index if any(k in l for k in ["heat_pump", "gas_boiler", "resistive"])]
for c in COUNTRIES:
    print(f"\n  {c}:")
    for l in heat_links:
        if l.startswith(c):
            cap = n.links.at[l, "p_nom_opt"] / 1000
            if cap > 0.001:
                print(f"    {l}: {cap:.2f} GW")
    tes_cap = n.storage_units.at[f"{c}_TES", "p_nom_opt"] / 1000
    print(f"    {c}_TES: {tes_cap:.2f} GW  ({tes_cap*8:.1f} GWh)")

print("\nAnnual heat supply [TWh]:")
for c in COUNTRIES:
    hp_heat  = (n.links_t.p1[f"{c}_heat_pump"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6
    gb_heat  = (n.links_t.p1[f"{c}_gas_boiler"].abs() * SNAPSHOT_WEIGHT).sum() / 1e6
    res_heat = (n.links_t.p1[f"{c}_resistive"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6
    tes_dis  = (n.storage_units_t.p[f"{c}_TES"].clip(lower=0) * SNAPSHOT_WEIGHT).sum() / 1e6
    total    = ANNUAL_HEAT_GWh[c] / 1e3
    print(f"  {c} (demand: {total:.0f} TWh):  HP={hp_heat:.2f}  Boiler={gb_heat:.2f}  Resistive={res_heat:.2f}  TES={tes_dis:.2f} TWh")

print("\nExtra electricity consumed by heat sector [TWh]:")
for c in COUNTRIES:
    hp_elec  = (n.links_t.p0[f"{c}_heat_pump"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6
    res_elec = (n.links_t.p0[f"{c}_resistive"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6
    orig_dem = (df[f"{c}_load_actual_entsoe_transparency"] * SNAPSHOT_WEIGHT).sum() / 1e6
    print(f"  {c}: HP={hp_elec:.2f} TWh + Resistive={res_elec:.2f} TWh  ({(hp_elec+res_elec)/orig_dem*100:.1f}% of electricity demand)")

#PLOTS
plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    120,
})

HEAT_COLORS = {
    "heat_pump":  "#2ecc71",
    "gas_boiler": "#d35400",
    "resistive":  "#f39c12",
    "TES":        "#95a5a6",
}
HEAT_LABELS = {
    "heat_pump":  "Heat Pump",
    "gas_boiler": "Gas Boiler",
    "resistive":  "Resistive Heater",
    "TES":        "Thermal Storage",
}

ELEC_COLORS = {k: v["color"] for k, v in ELEC_TECH.items()}
ELEC_COLORS["gas"] = "#e67e22"

def savefig(fname):
    try:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  Note: could not save {fname} ({e})")


#Optimal installed capacities (electricity + heat)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.35)

# Electricity
elec_data = {}
for c in COUNTRIES:
    row = {}
    for tech in list(ELEC_TECH.keys()) + ["gas"]:
        name = f"{c}_{tech}" if tech != "gas" else f"{c}_gas_power"
        if name in n.generators.index:
            row[tech] = n.generators.at[name, "p_nom_opt"] / 1000
        elif name in n.links.index:
            row[tech] = n.links.at[name, "p_nom_opt"] / 1000
        else:
            row[tech] = 0.0
    elec_data[c] = row
df_elec = pd.DataFrame(elec_data).T
df_elec = df_elec.loc[:, df_elec.max() > 0.01]
df_elec.rename(columns={"solar": "Solar", "wind": "Wind", "nuclear": "Nuclear",
                         "coal": "Coal", "gas": "Gas (CCGT)"}, inplace=True)
elec_color_map = {"Solar": "#f1c40f", "Wind": "#3498db", "Nuclear": "#e74c3c",
                  "Coal": "#7f8c8d", "Gas (CCGT)": "#e67e22"}

df_elec.plot(kind="bar", stacked=True, ax=axes[0],
             color=[elec_color_map.get(t, "gray") for t in df_elec.columns],
             edgecolor="white", linewidth=0.4, width=0.55)
axes[0].set_title("Optimal Electricity Generation Capacities", fontsize=12, pad=10)
axes[0].set_xlabel("Country", labelpad=6)
axes[0].set_ylabel("Installed Capacity [GW]")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.7)
axes[0].grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
axes[0].set_axisbelow(True)

# Heat
heat_data = {}
for c in COUNTRIES:
    row = {}
    for tech_key in ["heat_pump", "gas_boiler", "resistive"]:
        row[HEAT_LABELS[tech_key]] = n.links.at[f"{c}_{tech_key}", "p_nom_opt"] / 1000
    row[HEAT_LABELS["TES"]] = n.storage_units.at[f"{c}_TES", "p_nom_opt"] / 1000
    heat_data[c] = row
df_heat = pd.DataFrame(heat_data).T

df_heat.plot(kind="bar", stacked=True, ax=axes[1],
             color=[HEAT_COLORS["heat_pump"], HEAT_COLORS["gas_boiler"],
                    HEAT_COLORS["resistive"], HEAT_COLORS["TES"]],
             edgecolor="white", linewidth=0.4, width=0.55)
axes[1].set_title("Optimal Heat Technology Capacities", fontsize=12, pad=10)
axes[1].set_xlabel("Country", labelpad=6)
axes[1].set_ylabel("Installed Capacity [GW$_{heat}$]")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(loc="upper left", fontsize=9, framealpha=0.7)
axes[1].grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
axes[1].set_axisbelow(True)

savefig("capacities.png")
plt.show()


#Czech Republic winter week – electricity + heat dispatch
WINTER_SLICE = slice("2019-01-07", "2019-01-13")
c = "CZ"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

# Electricity panel
gens_cz  = [g for g in n.generators.index if g.startswith(f"{c}_")]
gen_data = n.generators_t.p[gens_cz].loc[WINTER_SLICE].copy()
gen_data.columns = [g.replace(f"{c}_", "").capitalize() for g in gens_cz]
gen_data = gen_data.loc[:, gen_data.max() > 1]
gen_data.plot.area(ax=ax1,
    color=[ELEC_COLORS.get(col.lower(), "gray") for col in gen_data.columns],
    alpha=0.82, linewidth=0)

n.links_t.p1[f"{c}_gas_power"].loc[WINTER_SLICE].clip(lower=0).plot(
    ax=ax1, label="Gas (to power)", color="#e67e22", linewidth=1.8)
(-n.links_t.p0[f"{c}_heat_pump"].loc[WINTER_SLICE]).plot(
    ax=ax1, label="Heat pump (elec.)", color="#2ecc71", linewidth=1.8, linestyle="--")
n.loads_t.p_set[f"{c}_elec"].loc[WINTER_SLICE].plot(
    ax=ax1, color="black", linewidth=2.0, linestyle=":", label="Electricity demand")

ax1.set_ylabel("Power [MW]", fontsize=11)
ax1.set_title("Czech Republic – Electricity Dispatch (Winter Week)", fontsize=12, pad=8)
ax1.legend(ncol=3, fontsize=8, loc="upper right", framealpha=0.8)
ax1.grid(axis="y", linestyle="--", alpha=0.3); ax1.set_axisbelow(True)
ax1.set_xlabel("")

# Heat panel
hp_h  = n.links_t.p1[f"{c}_heat_pump"].abs().loc[WINTER_SLICE]
gb_h  = n.links_t.p1[f"{c}_gas_boiler"].abs().loc[WINTER_SLICE]
res_h = n.links_t.p1[f"{c}_resistive"].abs().loc[WINTER_SLICE]
tes_d = n.storage_units_t.p[f"{c}_TES"].clip(lower=0).loc[WINTER_SLICE]
tes_c = n.storage_units_t.p[f"{c}_TES"].clip(upper=0).loc[WINTER_SLICE]

pd.DataFrame({
    "Heat Pump":        hp_h.values,
    "Gas Boiler":       gb_h.values,
    "Resistive Heater": res_h.values,
    "TES (discharge)":  tes_d.values,
}, index=hp_h.index).plot.area(ax=ax2,
    color=[HEAT_COLORS["heat_pump"], HEAT_COLORS["gas_boiler"],
           HEAT_COLORS["resistive"], HEAT_COLORS["TES"]],
    alpha=0.82, linewidth=0)

tes_c.plot(ax=ax2, color=HEAT_COLORS["TES"], linewidth=1.5,
           linestyle="--", label="TES (charge)")
n.loads_t.p_set[f"{c}_heat_load"].loc[WINTER_SLICE].plot(
    ax=ax2, color="black", linewidth=2.0, linestyle=":", label="Heat demand")

ax2.set_ylabel("Heat [MW]", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.set_title("Czech Republic – Heat Supply Dispatch (Winter Week)", fontsize=12, pad=8)
ax2.legend(ncol=3, fontsize=8, loc="upper right", framealpha=0.8)
ax2.grid(axis="y", linestyle="--", alpha=0.3); ax2.set_axisbelow(True)

savefig("CZ_winter_dispatch.png")
plt.show()


#Annual heat supply mix per country 
fig, ax = plt.subplots(figsize=(8, 5))

mix = {}
for c in COUNTRIES:
    mix[c] = {
        "Heat Pump":        (n.links_t.p1[f"{c}_heat_pump"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6,
        "Gas Boiler":       (n.links_t.p1[f"{c}_gas_boiler"].abs() * SNAPSHOT_WEIGHT).sum() / 1e6,
        "Resistive Heater": (n.links_t.p1[f"{c}_resistive"].abs()  * SNAPSHOT_WEIGHT).sum() / 1e6,
    }
df_mix = pd.DataFrame(mix).T
df_mix.plot(kind="bar", stacked=True, ax=ax,
    color=[HEAT_COLORS["heat_pump"], HEAT_COLORS["gas_boiler"], HEAT_COLORS["resistive"]],
    edgecolor="white", linewidth=0.4, width=0.55)

ax.set_title("Annual Heat Supply Mix by Technology", fontsize=12, pad=10)
ax.set_xlabel("Country", labelpad=6)
ax.set_ylabel("Heat Energy [TWh/year]")
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.35); ax.set_axisbelow(True)

savefig("heat_mix.png")
plt.show()


#COP profile and heat pump output – Czech Republic
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.subplots_adjust(hspace=0.08)

df["CZ_COP"].plot(ax=ax1, color="#2ecc71", linewidth=0.9, label="Hourly COP")
ax1.axhline(df["CZ_COP"].mean(), color="navy", linestyle="--", linewidth=1.4,
            label=f"Annual mean COP = {df['CZ_COP'].mean():.2f}")
ax1.set_ylabel("COP [–]", fontsize=11)
ax1.set_title("Czech Republic – Air-source Heat Pump COP (2019)", fontsize=12, pad=8)
ax1.legend(fontsize=10, framealpha=0.8)
ax1.grid(axis="y", linestyle="--", alpha=0.3); ax1.set_axisbelow(True)
ax1.set_xlabel("")

n.links_t.p1["CZ_heat_pump"].abs().plot(ax=ax2, color="#2ecc71", linewidth=0.7, label="Heat pump output")
df["CZ_heat_demand"].plot(ax=ax2, color="black", linestyle="--", linewidth=1.2, label="Heat demand")
ax2.set_ylabel("Power [MW]", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.set_title("Czech Republic – Heat Pump Output vs Heat Demand", fontsize=12, pad=8)
ax2.legend(fontsize=10, framealpha=0.8)
ax2.grid(axis="y", linestyle="--", alpha=0.3); ax2.set_axisbelow(True)

savefig("COP_and_HP.png")
plt.show()


#Electricity demand increase due to heating electrification 
fig, ax = plt.subplots(figsize=(8, 5))

orig = {c: df[f"{c}_load_actual_entsoe_transparency"].sum() * SNAPSHOT_WEIGHT / 1e6 for c in COUNTRIES}
extra = {c: (n.links_t.p0[f"{c}_heat_pump"].abs().sum() +
             n.links_t.p0[f"{c}_resistive"].abs().sum()) * SNAPSHOT_WEIGHT / 1e6
         for c in COUNTRIES}

x = np.arange(len(COUNTRIES))
w = 0.38
b1 = ax.bar(x - w/2, [orig[c]  for c in COUNTRIES], w,
            label="Original electricity demand", color="#3498db", alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, [extra[c] for c in COUNTRIES], w,
            label="Additional electricity for heating", color="#2ecc71", alpha=0.85, edgecolor="white")

for bar in b2:
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.0f} TWh", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(COUNTRIES)
ax.set_ylabel("Annual Energy [TWh]", fontsize=11)
ax.set_xlabel("Country", labelpad=6)
ax.set_title("Electricity Demand Increase from Heating Sector Coupling", fontsize=12, pad=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.35); ax.set_axisbelow(True)

savefig("elec_increase.png")
plt.show()

print("\nDone.")
