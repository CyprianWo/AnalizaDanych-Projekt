# Analiza danych: Inflacja i Bezrobocie
#
# Ten skrypt zawiera analizę danych dotyczących inflacji i bezrobocia. Analiza obejmuje:
# 1. Wczytanie i czyszczenie danych
# 2. Analizę eksploracyjną danych (EDA)
# 3. Wizualizację danych
# 4. Analizę korelacji między inflacją a bezrobociem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ustawienia dla wykresów
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Ustawienie wyświetlania liczb zmiennoprzecinkowych
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Ustawienie wyświetlania wszystkich kolumn
pd.set_option('display.max_columns', None)

# Wczytanie danych
print("Wczytywanie danych...")
df_inflation = pd.read_csv('inflation_small.tsv', sep='\t')
df_unemployment = pd.read_csv('unemployment_small.tsv', sep='\t')

print("Dane o inflacji:")
print(f"Wymiary: {df_inflation.shape}")
print("\nPierwsze wiersze danych:")
print(df_inflation.head())

print("\nDane o bezrobociu:")
print(f"Wymiary: {df_unemployment.shape}")
print("\nPierwsze wiersze danych:")
print(df_unemployment.head())

# Wyodrębnienie kodu geograficznego z pierwszej kolumny
df_inflation['geo'] = df_inflation[df_inflation.columns[0]].str.split(',').str[-1].str.strip()
df_unemployment['geo'] = df_unemployment[df_unemployment.columns[0]].str.split(',').str[-1].str.strip()

# Usunięcie oryginalnej kolumny z metadanymi
df_inflation = df_inflation.drop(columns=df_inflation.columns[0])
df_unemployment = df_unemployment.drop(columns=df_unemployment.columns[0])

# Funkcje pomocnicze do czyszczenia danych
# def clean_numeric_data(value):
#     if pd.isna(value):
#         return np.nan
#     if isinstance(value, str):
#         value = value.strip()
#         if value == ':':
#             return np.nan
#         # Handle values like '6.3 b' by taking the first part
#         value = value.split(' ')[0]
#         value = value.replace(',', '.')
#         try:
#             return float(value)
#         except ValueError:
#             return np.nan
#     return value

# # Czyszczenie danych
# # Ostatnia kolumna to 'geo', więc ją pomijamy
# data_cols_inflation = df_inflation.columns[:-1]
# for col in data_cols_inflation:
#     df_inflation[col] = df_inflation[col].apply(clean_numeric_data)

# data_cols_unemployment = df_unemployment.columns[:-1]
# for col in data_cols_unemployment:
#     df_unemployment[col] = df_unemployment[col].apply(clean_numeric_data)

# Przekształcenie danych z formatu szerokiego na długi (melt)
df_inflation_long = df_inflation.melt(id_vars=['geo'], var_name='period', value_name='inflation_rate')
df_unemployment_long = df_unemployment.melt(id_vars=['geo'], var_name='period', value_name='unemployment_rate')

# Konwersja kolumny 'period' na typ datetime
df_inflation_long['period'] = pd.to_datetime(df_inflation_long['period'], format='%Y-%m', errors='coerce')
df_unemployment_long['period'] = pd.to_datetime(df_unemployment_long['period'], format='%Y-%m', errors='coerce')

# Usunięcie wierszy z brakującymi danymi po transformacji
df_inflation_long.dropna(inplace=True)
df_unemployment_long.dropna(inplace=True)

print("Wymiary po czyszczeniu i transformacji:")
print(f"Inflacja: {df_inflation_long.shape}")
print(f"Bezrobocie: {df_unemployment_long.shape}")
print("\nDane o inflacji (po transformacji):")
print(df_inflation_long.head())
print("\nDane o bezrobociu (po transformacji):")
print(df_unemployment_long.head())

# Podstawowe statystyki
print("\nPodstawowe statystyki dla inflacji:")
print(df_inflation_long.describe())
print("\nPodstawowe statystyki dla bezrobocia:")
print(df_unemployment_long.describe())

# Połączenie ramek danych w jedną, aby znaleźć wspólne okresy i kraje
df_merged = pd.merge(df_inflation_long, df_unemployment_long, on=['geo', 'period'], how='inner')

# Usunięcie wierszy z brakującymi danymi po transformacji i połączeniu
df_merged.dropna(inplace=True)

print("Wymiary po czyszczeniu i transformacji:")
print(f"Połączone dane: {df_merged.shape}")
print(df_merged.head())

# Podstawowe statystyki
print("\nPodstawowe statystyki dla połączonych danych:")
print(df_merged.describe())

# Wizualizacja danych

# Wybierzmy jeden region do wizualizacji trendów, np. pierwszy z listy
if not df_merged.empty:
    sample_geo = df_merged['geo'].iloc[0]
    df_sample = df_merged[df_merged['geo'] == sample_geo]

    # Wykres trendu inflacji
    plt.figure(figsize=(15, 6))
    plt.title(f'Trend inflacji w czasie dla {sample_geo}')
    plt.plot(df_sample['period'], df_sample['inflation_rate'], marker='o', label='Inflacja')
    plt.plot(df_sample['period'], df_sample['unemployment_rate'], marker='o', color='red', label='Bezrobocie')
    plt.xlabel('Okres')
    plt.ylabel('Wartość (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Wykres pudełkowy (box plot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=df_merged['inflation_rate'], ax=ax1)
ax1.set_title('Rozkład inflacji')
ax1.set_ylabel('Inflacja (%)')

sns.boxplot(data=df_merged['unemployment_rate'], ax=ax2, color='red')
ax2.set_title('Rozkład bezrobocia')
ax2.set_ylabel('Stopa bezrobocia (%)')

plt.tight_layout()
plt.show()

# Histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(data=df_merged['inflation_rate'], ax=ax1, kde=True)
ax1.set_title('Histogram inflacji')
ax1.set_xlabel('Inflacja (%)')

sns.histplot(data=df_merged['unemployment_rate'], ax=ax2, kde=True, color='red')
ax2.set_title('Histogram bezrobocia')
ax2.set_xlabel('Stopa bezrobocia (%)')

plt.tight_layout()
plt.show()

# Analiza korelacji
if not df_merged.empty:
    # Obliczanie współczynnika korelacji Pearsona
    correlation, p_value = stats.pearsonr(df_merged['inflation_rate'], df_merged['unemployment_rate'])
    print(f"Współczynnik korelacji Pearsona: {correlation:.3f}")
    print(f"Wartość p: {p_value:.3f}")

    # Wykres rozproszenia (scatter plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged, x='inflation_rate', y='unemployment_rate', alpha=0.5)
    plt.xlabel('Inflacja (%)')
    plt.ylabel('Stopa bezrobocia (%)')
    plt.title('Zależność między inflacją a bezrobociem')

    # Dodanie linii trendu
    sns.regplot(data=df_merged, x='inflation_rate', y='unemployment_rate', scatter=False, color='red', line_kws={'linestyle':'--'})

    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Brak wspólnych danych do analizy korelacji i wizualizacji.") 