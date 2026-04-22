"""
Real Data Fetcher from data.gov.in API
Fetches real Indian government data using official API
"""
import os
import pandas as pd
import numpy as np
from datagovindia import DataGovIndia

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY_FILE = os.path.join(DATA_DIR, 'api_key.txt')


def get_api_key():
    """Get API key from file or return None"""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    return None


def save_api_key(key):
    """Save API key to file"""
    with open(API_KEY_FILE, 'w') as f:
        f.write(key)


def fetch_from_datagov(resource_id, description="Data"):
    """Fetch data from data.gov.in using resource ID"""
    api_key = get_api_key()
    
    if not api_key:
        print(f"No API key found. Fetching {description} using fallack data.")
        return None
    
    try:
        print(f"Connecting to data.gov.in API for {description}...")
        datagovin = DataGovIndia(api_key)
        data = datagovin.get_data(resource_id)
        print(f"Got {len(data)} records from data.gov.in")
        return data
    except Exception as e:
        print(f"API Error: {e}")
        print(f"Using fallback data for {description}")
        return None


def fetch_army_api():
    """Fetch real Army data from data.gov.in"""
    
    # Try to fetch from API if key available
    resource_id = "year-wise-number-candidates-recruited-indian-army-2017-18-2021-22"
    
    # First check for cached API data
    api_cache = os.path.join(DATA_DIR, 'army_api_data.csv')
    
    if os.path.exists(api_cache):
        print("Loading cached API data for Army...")
        return pd.read_csv(api_cache)
    
    # Try API
    data = fetch_from_datagov(resource_id, "Army Recruitment")
    
    if data is not None and len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(api_cache, index=False)
        return df
    
    return None


def fetch_education_api():
    """Fetch real Education data from data.gov.in"""
    api_cache = os.path.join(DATA_DIR, 'education_api_data.csv')
    
    if os.path.exists(api_cache):
        print("Loading cached API data for Education...")
        return pd.read_csv(api_cache)
    
    # Education loans API
    resource_id = "stateut-wise-details-educational-loans-disbursed-public-sector-banks-2021-22-2023-24"
    data = fetch_from_datagov(resource_id, "Education Loans")
    
    if data is not None and len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(api_cache, index=False)
        return df
    
    return None


def fetch_bank_api():
    """Fetch real Bank Loan data from data.gov.in"""
    api_cache = os.path.join(DATA_DIR, 'bank_api_data.csv')
    
    if os.path.exists(api_cache):
        print("Loading cached API data for Bank...")
        return pd.read_csv(api_cache)
    
    # RBI education loan data
    resource_id = "year-wise-outstanding-balance-towards-education-loan-all-scheduled-commercial-banks"
    data = fetch_from_datagov(resource_id, "Bank Education Loans")
    
    if data is not None and len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(api_cache, index=False)
        return df
    
    return None


def setup_api():
    """Guide user to set up API key"""
    print("=" * 50)
    print("To use real data from data.gov.in, you need an API key:")
    print("=" * 50)
    print("1. Go to https://data.gov.in")
    print("2. Register / Login")
    print("3. Go to 'My Account'")
    print("4. Get your API key")
    print("5. Enter it below")
    print("=" * 50)
    
    key = input("Enter your API key (or press Enter to skip): ").strip()
    
    if key:
        save_api_key(key)
        print("API key saved!")
    else:
        print("Skipping API key. Using fallback data.")
    
    return key


if __name__ == '__main__':
    setup_api()