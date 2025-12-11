# EDA on ICAO Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
def_path = Path("/Users/arnavpatil/Desktop/JetEngineSimulation/data/")


# Ensure CANTERA_DATA points to the directory containing the file
# os.environ["CANTERA_DATA"] = str(def_path)
df = pd.read_csv("/Users/arnavpatil/Desktop/JetEngineSimulation/data/icao_engine_data.csv")


print("Shape:", df.shape)
print(df.columns)
print(df.head())

print(df.info())
print(df.isna().sum())

df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

print(df.describe())

print("Unique engines:", df.Engine_ID.nunique())
print("Modes:", df.Mode.unique())

num_cols = ["Fuel_Flow_kg/s", "HC_g/kg", "CO_g/kg", "NOx_g/kg", "Smoke_Number", "Pressure_Ratio", "Rated_Thrust_kN"]
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Trent 1000 Emission Metrics)")
plt.show()

avg_by_mode = df.groupby("Mode")[["Fuel_Flow_kg/s", "NOx_g/kg", "CO_g/kg", "Smoke_Number"]].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data=avg_by_mode, x="Mode", y="Fuel_Flow_kg/s", ax=axes[0])
sns.barplot(data=avg_by_mode, x="Mode", y="NOx_g/kg", ax=axes[1])
sns.barplot(data=avg_by_mode, x="Mode", y="CO_g/kg", ax=axes[2])
axes[0].set_title("Fuel Flow vs Mode")
axes[1].set_title("NOx Emissions vs Mode")
axes[2].set_title("CO Emissions vs Mode")
plt.tight_layout()
plt.show()

sns.pairplot(df, vars=["Pressure_Ratio", "Bypass_Ratio", "Fuel_Flow_kg/s", "NOx_g/kg", "CO_g/kg"], hue="Mode")
plt.suptitle("Parameter Relationships by Operating Mode", y=1.02)
plt.show()

df["NOx_per_thrust"] = df["NOx_g/kg"] / df["Rated_Thrust_kN"]
df["CO_per_thrust"] = df["CO_g/kg"] / df["Rated_Thrust_kN"]

sns.scatterplot(data=df, x="Fuel_Flow_kg/s", y="NOx_per_thrust", hue="Mode", style="Engine_ID")
plt.title("NOx per Thrust vs Fuel Flow")
plt.show()
