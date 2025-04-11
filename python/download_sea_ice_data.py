# python/download_sea_ice_data.py
import requests
import os
from datetime import datetime

# Base URL pattern
# The placeholders {year}, {year_short}, {month_zero}, {day_zero} will be filled in the loop

# Arctic
# BASE_URL = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/netcdf/{year}/asi-AMSR2-n6250-{year}{month_zero}{day_zero}-v5.4.nc"

# Southern Hemisphere
BASE_URL = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/netcdf/{year}/asi-AMSR2-s6250-{year}{month_zero}{day_zero}-v5.4.nc"

# Define the date range
START_YEAR = 2012
START_MONTH = 7
END_YEAR = 2024
END_MONTH = 12
DAY_TO_FETCH = 15 # We only want the 15th of each month

# Define the output directory
OUTPUT_DIR = "data/sea-ice-SH-bremen"

def download_file(url, output_path):
    """Downloads a file from a URL to a specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up potentially incomplete file if download failed
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")
        # Clean up potentially incomplete file if download failed
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    """Main function to iterate through dates and download files."""
    print(f"Starting download process...")
    print(f"Target directory: {OUTPUT_DIR}")
    
    # Ensure the base output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files_attempted = 0
    successful_downloads = 0
    skipped_files = 0

    # Iterate through each year in the range
    for year in range(START_YEAR, END_YEAR + 1):
        # Determine the range of months for the current year
        start_m = START_MONTH if year == START_YEAR else 1
        end_m = END_MONTH if year == END_YEAR else 12

        # Iterate through each month in the determined range
        for month in range(start_m, end_m + 1):
            total_files_attempted += 1
            
            # Format month and day with leading zeros
            month_zero = f"{month:02d}"
            day_zero = f"{DAY_TO_FETCH:02d}" # Day is always 15

            # Construct the specific URL for the date
            file_url = BASE_URL.format(
                year=year,
                month_zero=month_zero,
                day_zero=day_zero
            )

            # Construct the output filename and path
            filename = os.path.basename(file_url)
            output_path = os.path.join(OUTPUT_DIR, filename)

            # Check if the file already exists
            if os.path.exists(output_path):
                print(f"Skipping already downloaded file: {output_path}")
                skipped_files += 1
                continue

            # Attempt to download the file
            print(f"Attempting to download: {file_url}")
            if download_file(file_url, output_path):
                successful_downloads += 1

    print("\nDownload process finished.")
    print(f"Total files attempted: {total_files_attempted}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Skipped (already existing): {skipped_files}")
    print(f"Failed downloads: {total_files_attempted - successful_downloads - skipped_files}")

if __name__ == "__main__":
    main()
