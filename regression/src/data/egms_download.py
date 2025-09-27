# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:59:48 2024

@author: 39351
"""

import os
import requests
import zipfile

data_path = f'../data/'

# The token is to be obtained directly from the EGMS website 
my_token = f'44dbda79e4114afb9033108f132a2031'

def egms_download(pswd, usn):
    
    
    # Construct the url and download the Dataset
    for east in range(40, 52):
        for north in range(15,27):
            filename = f'EGMS_L3_E{east}N{north}_100km_U_2018_2022_1'
            if filename == 'EGMS_L3_E47N23_100km_U_2018_2022_1':
                continue
            file_path = os.path.join(data_path, filename)
            
            if os.path.exists(f'{file_path}.zip'):
                print(f'file{file_path}.zip already exists')
                continue
            else:
                file_url = f"https://egms.land.copernicus.eu/insar-api/archive/download/{filename}.zip?id={my_token}"
                print(f'downloading file {filename}')
                
                # Send the request
                response = requests.get(file_url, auth=('usnm', 'psw'))
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Save the file locally
                    with open(f'{file_path}.zip', 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                file.write(chunk)
                    print(f"File downloaded successfully as {filename}")
                    
                    # print(f'Extracting {filename}.zip ...')
                    # with zipfile.ZipFile(f'{file_path}.zip', 'r') as zip_ref:
                    #     zip_ref.extract(f'{filename}.csv', path=data_path)
                        
                    # # remove the non-csv files
                    # os.remove(f'{file_path}.zip')
                
                else:
                    print(f"Failed to download file. Status code: {response.status_code}")
                
    # files = os.listdir(data_path) 
    # for file in files:
    #     if os.path.exists(os.path.join(data_path,file.split('.')[0] + '.csv')):
    #         continue
    #     if file.endswith('.zip'):
    #         print(os.path.join(data_path, file))
    #         with zipfile.ZipFile(os.path.join(data_path, file), 'r') as zip_ref:
    #             zip_ref.extract(file.split('.')[0] + '.csv', path=data_path)
    #             os.remove(f'{file_path}.zip')



