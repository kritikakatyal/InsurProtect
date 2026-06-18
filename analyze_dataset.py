import pandas as pd

def analyze_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("--- Info ---")
        print(df.info())
        print("\n--- Basic Statistics ---")
        print(df.describe())
        print("\n--- Null Values ---")
        print(df.isnull().sum())
        print("\n--- Unique Values in Categorical/Binary Columns ---")
        binary_cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']
        for col in binary_cols:
            if col in df.columns:
                print(f"{col}: {df[col].unique()}")
        
        print("\n--- Correlation with Target (PremiumPrice) ---")
        if 'PremiumPrice' in df.columns:
            print(df.corr()['PremiumPrice'].sort_values(ascending=False))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_data('Medicalpremium.csv')
