import customtkinter as ctk
import re

# Set the appearance mode and default color theme
ctk.set_appearance_mode("dark")  # Modes: "dark", "light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue", "dark-blue", "green"

# Create main window
root = ctk.CTk()
root.title("Unique Code Generator")

# Dictionary for periodic table elements and their atomic numbers
periodic_table = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
    "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108,
    "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116,
    "Ts": 117, "Og": 118
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

def calculate_unique_code(species):
    compounds = species.split('-')
    unique_code = []
    for compound in compounds:
        element_counts = parse_compound(compound)
        total_atomic_number = sum(periodic_table[element] * count for element, count in element_counts.items())
        unique_code.append(str(total_atomic_number))
    return "".join(unique_code)

def generate_unique_code():
    species = catalyst_type_entry.get()
    unique_code = calculate_unique_code(species)
    unique_code_label.configure(text=f"Unique code: {unique_code}")

# Create frame for inputs
input_frame = ctk.CTkFrame(root)
input_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Create labels and entry fields
catalyst_type_label = ctk.CTkLabel(input_frame, text="Catalyst type (CuO-ZnO-Al2O3):", font=("Hero", 14))
catalyst_type_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')

catalyst_type_entry = ctk.CTkEntry(input_frame, width=250)
catalyst_type_entry.grid(row=0, column=1, padx=10, pady=10)

# Create button to generate unique code
generate_button = ctk.CTkButton(input_frame, text="Generate Code", font=("Hero", 14, "bold"), command=generate_unique_code)
generate_button.grid(row=1, column=0, columnspan=2, padx=10, pady=20)

# Create label to display unique code
unique_code_label = ctk.CTkLabel(input_frame, text="Unique code: ", font=("Hero", 14))
unique_code_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()