import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych o bezrobociu
df_bezrobocie = pd.read_csv('estat_tps00203.tsv', sep='\t', encoding='utf-8')

# Wczytanie danych o inflacji
df_inflacja = pd.read_csv('estat_tec00118.tsv', sep='\t', encoding='utf-8')

# Przygotowanie danych o bezrobociu
# Rozdzielenie pierwszej kolumny na osobne kolumny
df_bezrobocie[['freq', 'age', 'unit', 'sex', 'geo']] = df_bezrobocie.iloc[:, 0].str.split(',', expand=True)
df_bezrobocie = df_bezrobocie.drop(df_bezrobocie.columns[0], axis=1)

# Filtrowanie danych
df_bezrobocie = df_bezrobocie[df_bezrobocie['unit'] == 'PC_ACT']  # Wybieramy tylko dane w procentach aktywnych
df_bezrobocie = df_bezrobocie[df_bezrobocie['sex'] == 'T']  # Wybieramy dane dla obu płci
df_bezrobocie = df_bezrobocie[df_bezrobocie['age'] == 'Y15-74']  # Wybieramy dane dla wieku 15-74

# Przygotowanie danych o inflacji
# Rozdzielenie pierwszej kolumny na osobne kolumny
df_inflacja[['freq', 'unit', 'coicop', 'geo']] = df_inflacja.iloc[:, 0].str.split(',', expand=True)
df_inflacja = df_inflacja.drop(df_inflacja.columns[0], axis=1)

# Filtrowanie danych o inflacji
df_inflacja = df_inflacja[df_inflacja['coicop'] == 'CP00']  # Wybieramy dane o inflacji

# Czyszczenie nazw kolumn (usuwanie spacji)
df_bezrobocie.columns = df_bezrobocie.columns.str.strip()
df_inflacja.columns = df_inflacja.columns.str.strip()

# Konwersja kolumn z latami na wartości numeryczne
lata = [str(rok) for rok in range(2013, 2025)]
for rok in lata:
    if rok in df_bezrobocie.columns:
        df_bezrobocie[rok] = pd.to_numeric(df_bezrobocie[rok].str.replace(':', 'NaN').str.replace('b', '').str.replace('u', ''), errors='coerce')
    if rok in df_inflacja.columns:
        df_inflacja[rok] = pd.to_numeric(df_inflacja[rok].str.replace(':', 'NaN').str.replace('d', ''), errors='coerce')

# Przygotowanie danych do analizy korelacji
kraje = df_bezrobocie['geo'].unique()
kraje = [kraj for kraj in kraje if kraj in df_inflacja['geo'].unique()]

# Tworzenie DataFrame z korelacjami
korelacje = []

for kraj in kraje:
    bezrobocie = df_bezrobocie[df_bezrobocie['geo'] == kraj][lata].values.flatten()
    inflacja = df_inflacja[df_inflacja['geo'] == kraj][lata].values.flatten()
    
    # Usuwanie wartości NaN
    mask = ~(np.isnan(bezrobocie) | np.isnan(inflacja))
    bezrobocie = bezrobocie[mask]
    inflacja = inflacja[mask]
    
    if len(bezrobocie) > 0 and len(inflacja) > 0:
        korelacja = np.corrcoef(bezrobocie, inflacja)[0, 1]
        korelacje.append({
            'Kraj': kraj,
            'Korelacja': korelacja
        })

df_korelacje = pd.DataFrame(korelacje)

# Sortowanie po wartości korelacji
df_korelacje = df_korelacje.sort_values('Korelacja', ascending=False)

# Wyświetlenie wyników
print("\nKorelacja między stopą bezrobocia a inflacją w latach 2013-2024:")
print(df_korelacje)

# Wykres korelacji dla wybranych krajów
plt.figure(figsize=(15, 8))
sns.barplot(data=df_korelacje.head(10), x='Korelacja', y='Kraj')
plt.title('Top 10 krajów z najwyższą korelacją między bezrobociem a inflacją')
plt.xlabel('Współczynnik korelacji')
plt.ylabel('Kraj')
plt.tight_layout()
plt.savefig('korelacja_bezrobocie_inflacja.png')
plt.close()

# Obliczenie średniej korelacji
srednia_korelacja = df_korelacje['Korelacja'].mean()
print(f"\nŚrednia korelacja dla wszystkich krajów: {srednia_korelacja:.3f}")

# Zapisanie wyników do pliku CSV
df_korelacje.to_csv('korelacje_bezrobocie_inflacja.csv', index=False) 