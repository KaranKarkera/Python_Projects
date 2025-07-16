import customtkinter as ctk
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tkinter import messagebox

# Set the appearance mode and default color theme
ctk.set_appearance_mode("dark")  # Modes: "dark", "light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue", "dark-blue", "green"

def backloop():
    l = []

    periodic_table = {
        "H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 12,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,
        "S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,
        "Cu": 29,"Zn": 30,"Ga": 31,"Ge": 32,"As": 33,"Se": 34,"Br": 35,"Kr": 36,"Rb": 37,"Sr": 38,"Y": 39,"Zr": 40,"Nb": 41,
        "Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53,"Xe": 54,
        "Cs": 55,"Ba": 56,"La": 57,"Ce": 58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,
        "Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,"Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,
        "Tl": 81,"Pb": 82,"Bi": 83,"Po": 84,"At": 85,"Rn": 86,"Fr": 87,"Ra": 88,"Ac": 89,"Th": 90,"Pa": 91,"U": 92,"Np": 93,
        "Pu": 94,"Am": 95,"Cm": 96,"Bk": 97,"Cf": 98,"Es": 99,"Fm": 120,"Md": 121,"No": 122,"Lr": 123,"Rf": 124,"Db": 125,"Sg": 126,
        "Bh": 127,"Hs": 128,"Mt": 129,"Ds": 112,"Rg": 111,"Cn": 112,"Nh": 113,"Fl": 114,"Mc": 115,"Lv": 116,"Ts": 117,"Og": 118
    }

    def parse_compound(compound):
        element_pattern = r'([A-Z][a-z]*)(\d*)'
        elements = re.findall(element_pattern, compound)
        element_counts = {}
        for (element, count) in elements:
            if count == "":
                count = 1
            else:
                count = int(count)
            if element in element_counts:
                element_counts[element] += count
            else:
                element_counts[element] = count
        return element_counts

    def calculate_total_atomic_number(species):
        compounds = species.split('-')
        total_atomic_number = 0
        indv_atomic_number = 0
        for compound in compounds:
            element_counts = parse_compound(compound)
            for element, count in element_counts.items():
                total_atomic_number += periodic_table[element] * count
                indv_atomic_number += periodic_table[element] * count
            l.append(indv_atomic_number)
            indv_atomic_number = 0  
        return total_atomic_number


    species = catalyst_type_entry.get()
    total_atomic_number = calculate_total_atomic_number(species)
    var=""
    for i in l:
        var=var+str(i)
    return var

# Function to predict based on user input
def predict_values():
    try:
        # Get user input from Entry widgets
        flow_rate = float(flow_rate_entry.get())
        carbon_hydrogen_ratio = float(carbon_hydrogen_ratio_entry.get())
        CO = int(CO_entry.get() or 0)
        H2 = int(H2_entry.get() or 0)
        CO2 = int(CO2_entry.get() or 0)
        CH4 = int(CH4_entry.get() or 0)
        O = int(O_entry.get() or 0)
        N2 = int(N2_entry.get() or 0)
        syngas_composition = int(str(CO) + str(H2) + str(CO2) + str(CH4) + str(O) + str(N2))
        catalyst_type_1 = backloop()
        catalyst_type = catalyst_type_1

        catalyst_tube_diameter = float(catalyst_tube_diameter_entry.get())

        # Construct input data for prediction
        new_data = pd.DataFrame({
            'flow_rate': [flow_rate],
            'carbon_to_hydrogen_ratio': [carbon_hydrogen_ratio],
            'syngas_composition': [syngas_composition],
            'catalyst_type': [catalyst_type],
            'catalyst_tube_diameter': [catalyst_tube_diameter]
        })

        # Encode categorical 'catalyst_type' if necessary
        if isinstance(catalyst_type, str):
            new_data['catalyst_type'] = LabelEncoder().fit_transform(new_data['catalyst_type'])

        # Predict optimal temperature, pressure, and methanol production rate
        predicted_temperature = model_temperature.predict(new_data)
        predicted_pressure = model_pressure.predict(new_data)
        predicted_methanol_production_rate = model_methanol_production_rate.predict(new_data)

        # Update output labels with predicted values
        temperature_label.configure(text=f'Predicted optimal temperature: {predicted_temperature[0]:.2f}')
        pressure_label.configure(text=f'Predicted optimal pressure: {predicted_pressure[0]:.2f}')
        methanol_production_label.configure(text=f'Predicted methanol production rate: {predicted_methanol_production_rate[0]:.2f}')

    except ValueError as ve:
        messagebox.showerror("Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Load and Prepare Data (same as before)
df = pd.read_csv('methanol_production_data.csv')

# Reorder columns according to the new order
df = df[['catalyst_type', 'catalyst_tube_diameter', 'syngas_composition', 
         'carbon_to_hydrogen_ratio', 'temperature', 'pressure', 
         'flow_rate', 'methanol_production_rate']]

# Ensure numeric data types for features and targets
numeric_columns = ['flow_rate', 'carbon_to_hydrogen_ratio', 'syngas_composition',
                   'catalyst_tube_diameter', 'temperature', 'pressure', 'methanol_production_rate']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Handle categorical variable 'catalyst_type'
# If 'catalyst_type' is categorical (string), encode it numerically
if df['catalyst_type'].dtype == 'object':
    df['catalyst_type'] = LabelEncoder().fit_transform(df['catalyst_type'])

# Extract features and targets
X = df[['flow_rate', 'carbon_to_hydrogen_ratio', 'syngas_composition', 
        'catalyst_type', 'catalyst_tube_diameter']]
y_temperature = df['temperature']
y_pressure = df['pressure']
y_methanol_production_rate = df['methanol_production_rate']

# Train Random Forest models
model_temperature = RandomForestRegressor()
model_temperature.fit(X, y_temperature)

model_pressure = RandomForestRegressor()
model_pressure.fit(X, y_pressure)

model_methanol_production_rate = RandomForestRegressor()
model_methanol_production_rate.fit(X, y_methanol_production_rate)

# Create main window
root = ctk.CTk()
root.title("Methanol Production Prediction Model")

# Create frame for inputs
input_frame = ctk.CTkFrame(root)
input_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Create labels and entry fields
ctk.CTkLabel(input_frame, text="Flow rate (ml/min):", font=("Hero", 12)).grid(row=0, column=0, padx=12, pady=5, sticky="w")
flow_rate_entry = ctk.CTkEntry(input_frame)
flow_rate_entry.grid(row=0, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="Carbon to Hydrogen ratio:", font=("Hero", 12)).grid(row=1, column=0, padx=12, pady=5, sticky="w")
carbon_hydrogen_ratio_entry = ctk.CTkEntry(input_frame)
carbon_hydrogen_ratio_entry.grid(row=1, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="CO (Carbon Monoxide) (%):", font=("Hero", 12)).grid(row=2, column=0, padx=12, pady=5, sticky="w")
CO_entry = ctk.CTkEntry(input_frame)
CO_entry.grid(row=2, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="H2 (Hydrogen) (%):", font=("Hero", 12)).grid(row=3, column=0, padx=12, pady=5, sticky="w")
H2_entry = ctk.CTkEntry(input_frame)
H2_entry.grid(row=3, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="CO2 (Carbon Dioxide) (%):", font=("Hero", 12)).grid(row=4, column=0, padx=12, pady=5, sticky="w")
CO2_entry = ctk.CTkEntry(input_frame)
CO2_entry.grid(row=4, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="CH4 (Methane) (%):", font=("Hero", 12)).grid(row=5, column=0, padx=12, pady=5, sticky="w")
CH4_entry = ctk.CTkEntry(input_frame)
CH4_entry.grid(row=5, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="O2 (Oxygen) (%):", font=("Hero", 12)).grid(row=6, column=0, padx=12, pady=5, sticky="w")
O_entry = ctk.CTkEntry(input_frame)
O_entry.grid(row=6, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="N2 (Nitrogen) (%):", font=("Hero", 12)).grid(row=7, column=0, padx=12, pady=5, sticky="w")
N2_entry = ctk.CTkEntry(input_frame)
N2_entry.grid(row=7, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="Catalyst type (Elements e.g., CuO-ZnO-Al2O3):", font=("Hero", 12)).grid(row=8, column=0, padx=12, pady=5, sticky="w")
catalyst_type_entry = ctk.CTkEntry(input_frame)
catalyst_type_entry.grid(row=8, column=1, padx=12, pady=5)

ctk.CTkLabel(input_frame, text="Catalyst tube diameter (mm):", font=("Hero", 12)).grid(row=9, column=0, padx=12, pady=5, sticky="w")
catalyst_tube_diameter_entry = ctk.CTkEntry(input_frame)
catalyst_tube_diameter_entry.grid(row=9, column=1, padx=12, pady=5)

# Create predict button
predict_button = ctk.CTkButton(input_frame, text="Predict", command=predict_values)
predict_button.grid(row=12, column=0, columnspan=2, pady=20)

# Create frame for outputs
output_frame = ctk.CTkFrame(root)
output_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Create labels for outputs
temperature_label = ctk.CTkLabel(output_frame, text="Predicted optimal temperature (°C): ", font=("Hero", 12))
temperature_label.pack(pady=12)

pressure_label = ctk.CTkLabel(output_frame, text="Predicted optimal pressure (bar): ", font=("Hero", 12))
pressure_label.pack(pady=12)

methanol_production_label = ctk.CTkLabel(output_frame, text="Predicted methanol production rate (%): ", font=("Hero", 12))
methanol_production_label.pack(pady=12)

# Run the application
root.mainloop()