import pandas as pd

# Wczytaj plik, separator to tabulator
inflation_df = pd.read_csv('inflation.tsv', sep='\t')
unemployment_df = pd.read_csv('unemployment.tsv', sep='\t')

# Wyświetl kilka pierwszych wierszy
print(inflation_df.head())

print(unemployment_df.head())