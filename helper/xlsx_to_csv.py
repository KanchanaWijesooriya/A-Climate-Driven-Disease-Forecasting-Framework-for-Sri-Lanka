import pandas as pd
import os

# Setup
excel_file = '/home/chanuka002/Research/Patient counts raw data/Patient counts raw data 2025.xlsx'
output_folder = '/home/chanuka002/Research/Patient counts raw data/CSV/Patient counts raw data 2025'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read and convert
sheets_dict = pd.read_excel(excel_file, sheet_name=None)

for sheet_name, df in sheets_dict.items():
    clean_name = sheet_name.replace(' ', '_').replace('/', '_')
    csv_path = os.path.join(output_folder, f'{clean_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')