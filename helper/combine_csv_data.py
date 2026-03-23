import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def parse_date_from_filename(filename):
    """
    Extract date range from filename like '2020-02-15_to_2020-02-21.csv'
    Returns the end date of the range
    """
    basename = os.path.basename(filename)
    date_part = basename.replace('.csv', '')
    # Extract the end date (after 'to_')
    end_date_str = date_part.split('_to_')[1]
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    return end_date

def shift_date_back_one_week(date):
    """Shift date back by 7 days"""
    return date - timedelta(days=7)

# Load the weather dataset
weather_df = pd.read_csv('/home/chanuka002/Research/weather_weekly_with_seasonality_lags.csv')

# Strip whitespace from column names
weather_df.columns = weather_df.columns.str.strip()

# Strip whitespace from district names
weather_df['district'] = weather_df['district'].str.strip()

# Convert start_date and end_date to datetime
weather_df['start_date'] = pd.to_datetime(weather_df['start_date'])
weather_df['end_date'] = pd.to_datetime(weather_df['end_date'])

# Get all patient count CSV files from all years
base_path = '/home/chanuka002/Research/Patient counts raw data/CSV'
all_csv_files = []

for year_folder in ['Patient counts raw data 2020', 
                    'Patient counts raw data 2021', 
                    'Patient counts raw data 2022', 
                    'Patient counts raw data 2023', 
                    'Patient counts raw data 2024',
                    'Patient counts raw data 2025']:
    year_path = os.path.join(base_path, year_folder)
    if os.path.exists(year_path):
        csv_files = glob.glob(os.path.join(year_path, '*.csv'))
        all_csv_files.extend(csv_files)

print(f"Found {len(all_csv_files)} CSV files to process")

# Create a list to store all patient count data
patient_data_list = []

# Process each CSV file
for csv_file in sorted(all_csv_files):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Remove unnamed columns and empty rows
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(subset=['District'])
        
        # Get the report date range from filename
        report_end_date = parse_date_from_filename(csv_file)
        
        # Shift back one week to get actual patient count period
        actual_end_date = shift_date_back_one_week(report_end_date)
        actual_start_date = actual_end_date - timedelta(days=6)
        
        # Add date columns to the dataframe
        df['patient_start_date'] = actual_start_date
        df['patient_end_date'] = actual_end_date
        
        # Filter out 'Kalmune' district
        df = df[df['District'] != 'Kalmune']
        
        # Clean district names (remove spaces, make consistent)
        df['District'] = df['District'].str.strip()
        
        # Add to list
        patient_data_list.append(df)
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        continue

# Combine all patient data
if patient_data_list:
    patient_df = pd.concat(patient_data_list, ignore_index=True)
    
    # Convert date columns to datetime
    patient_df['patient_start_date'] = pd.to_datetime(patient_df['patient_start_date'])
    patient_df['patient_end_date'] = pd.to_datetime(patient_df['patient_end_date'])
    
    print(f"\nPatient data shape: {patient_df.shape}")
    print(f"Date range: {patient_df['patient_start_date'].min()} to {patient_df['patient_end_date'].max()}")
    print(f"Unique districts in patient data: {sorted(patient_df['District'].unique())}")
    print(f"Unique districts in weather data: {sorted(weather_df['district'].unique())}")
    
    # Standardize district names for matching
    # Create a mapping dictionary for any district name variations
    patient_df['District'] = patient_df['District'].str.strip()
    
    # Standardize district name variations
    patient_df['District'] = patient_df['District'].replace({
        'NuwaraEliya': 'Nuwara Eliya'
    })
    weather_df['district'] = weather_df['district'].replace({
        'NuwaraEliya': 'Nuwara Eliya'
    })
    
    # Create a temporary dataframe for merging with only disease counts
    patient_merge_df = patient_df[['District', 'patient_start_date', 'patient_end_date', 
                                     'Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']].copy()
    
    # Rename columns to match weather dataset for merging
    patient_merge_df.columns = ['district', 'start_date', 'end_date', 
                                 'Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']
    
    # Merge: Add only the 4 disease count columns to weather data
    final_df = weather_df.merge(
        patient_merge_df,
        on=['district', 'start_date', 'end_date'],
        how='left'
    )
    
    # Keep NaN values as null (don't fill with 0)
    # NaN = data unavailable, 0 = actual patient count of zero
    patient_columns = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']
    
    # Save the final dataset
    output_path = '/home/chanuka002/Research/Final_Data_Counts.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Final dataset saved to: {output_path}")
    print(f"Final shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Show sample with patient counts
    print(f"\nSample of merged data (first few rows with patient counts):")
    sample_with_counts = final_df[final_df['Leptospirosis'] > 0].head(5)
    if len(sample_with_counts) > 0:
        print(sample_with_counts[['district', 'start_date', 'end_date', 'Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']])
    
    # Show summary statistics
    print("\nPatient count statistics:")
    for col in patient_columns:
        if col in final_df.columns:
            non_zero = (final_df[col] > 0).sum()
            print(f"{col}: min={final_df[col].min()}, max={final_df[col].max()}, "
                  f"mean={final_df[col].mean():.2f}, non-zero records={non_zero}")
    
    # Check matching rates
    total_weather_records = len(final_df)
    matched_records = (final_df['Leptospirosis'] > 0).sum()
    print(f"\nMatch rate: {matched_records}/{total_weather_records} "
          f"({matched_records/total_weather_records*100:.1f}% of weather records have patient data)")
    
else:
    print("No patient data files found!")
    