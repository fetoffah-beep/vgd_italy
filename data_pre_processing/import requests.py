import requests
from datetime import datetime, timedelta

# Replace with your CLMS API credentials
USERNAME = 'your_username'
PASSWORD = 'your_password'

# Example: Download a dataset (replace with your dataset info)
DATASET_ID = 'clc2018'
FILE_ID = 'CLC2018_CLC2018_V2020_20u1.tif'
# Set the dataset and file info for Soil Moisture (SSM) for Italy, 2018-2022
# You need to know the exact file naming convention and path for the CLMS soil moisture product.
# Example filename pattern (update as needed):
# SSM_ITALY_YYYYMMDD.tif


START_DATE = datetime(2018, 1, 1)
END_DATE = datetime(2022, 12, 31)

country_code = 'ITALY'  # Update if a different code is required by the provider

def generate_file_ids(start_date, end_date, country_code):
    current = start_date
    file_ids = []
    while current <= end_date:
        file_id = f'SSM_{country_code}_{current.strftime("%Y%m%d")}.tif'
        file_ids.append(file_id)
        current += timedelta(days=1)
    return file_ids

file_ids = generate_file_ids(START_DATE, END_DATE, country_code)

BASE_URL = 'https://land.copernicus.vgt.vito.be/PDF/datapool/CLMS/soilmoisture/'

# Example: Download the first file (update loop as needed)
FILE_ID = file_ids[0]
DOWNLOAD_URL = f'{BASE_URL}{FILE_ID}'

# Authenticate and download
with requests.Session() as s:
    s.auth = (USERNAME, PASSWORD)
    response = s.get(DOWNLOAD_URL, stream=True)
    if response.status_code == 200:
        with open(FILE_ID, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'Downloaded {FILE_ID}')
    else:
        print(f'Failed to download: {response.status_code} {response.reason}')


        