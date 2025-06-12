import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ustawienia wykresów i pandas
plt.style.use('seaborn-v0_8')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

# 1. Wczytanie danych
print("Wczytywanie danych...")
df_infl = pd.read_csv('inflation_small.tsv', sep='\t')
df_unemp = pd.read_csv('unemployment_small.tsv', sep='\t')

# Usuń spacje z nazw kolumn (poza pierwszą kolumną z metadanymi)
df_infl.columns = [col.strip() for col in df_infl.columns]
df_unemp.columns = [col.strip() for col in df_unemp.columns]

print("Kolumny w inflacji:", df_infl.columns.tolist())
print("Kolumny w bezrobociu:", df_unemp.columns.tolist())

# 2. Wyodrębnienie kodu kraju
# Ostatni element w pierwszej kolumnie to kod kraju
for df in [df_infl, df_unemp]:
    df['geo'] = df[df.columns[0]].str.split(',').str[-1].str.strip()
    df.drop(columns=df.columns[0], inplace=True)

# 3. Przekształcenie do formatu długiego
infl_long = df_infl.melt(id_vars=['geo'], var_name='period', value_name='inflation')
unemp_long = df_unemp.melt(id_vars=['geo'], var_name='period', value_name='unemployment')

# 4. Konwersja okresu na datetime
infl_long['period'] = pd.to_datetime(infl_long['period'], format='%Y-%m', errors='coerce')
unemp_long['period'] = pd.to_datetime(unemp_long['period'], format='%Y-%m', errors='coerce')

# DEBUG: Sprawdź unikalne kody krajów i okresy
print("\nUnikalne kody krajów w inflacji:", sorted(infl_long['geo'].unique())[:10], '...')
print("Unikalne kody krajów w bezrobociu:", sorted(unemp_long['geo'].unique())[:10], '...')
print("Wspólne kraje:", sorted(set(infl_long['geo']).intersection(set(unemp_long['geo'])))[:10], '...')

print("\nPrzykładowe okresy w inflacji:", sorted(infl_long['period'].dropna().unique())[:5], '...')
print("Przykładowe okresy w bezrobociu:", sorted(unemp_long['period'].dropna().unique())[:5], '...')
print("Wspólne okresy:", sorted(set(infl_long['period'].dropna()).intersection(set(unemp_long['period'].dropna())))[:5], '...')

# 5. Czyszczenie wartości liczbowych
# Zamiana ':' i innych nietypowych wartości na NaN, konwersja na float
def clean_numeric(val):
    if pd.isna(val): return np.nan
    if isinstance(val, str):
        val = val.replace(',', '.').split(' ')[0].replace(':', '')
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val

infl_long['inflation'] = infl_long['inflation'].apply(clean_numeric)
unemp_long['unemployment'] = unemp_long['unemployment'].apply(clean_numeric)

# 6. Usunięcie braków i połączenie danych
infl_long.dropna(subset=['period', 'inflation'], inplace=True)
unemp_long.dropna(subset=['period', 'unemployment'], inplace=True)
df_merged = pd.merge(infl_long, unemp_long, on=['geo', 'period'], how='inner')
df_merged.dropna(subset=['inflation', 'unemployment'], inplace=True)

print(f"Połączone dane: {df_merged.shape}")
print(df_merged.head())

# 7. Korelacja globalna
if not df_merged.empty:
    pearson_corr, pearson_p = stats.pearsonr(df_merged['inflation'], df_merged['unemployment'])
    spearman_corr, spearman_p = stats.spearmanr(df_merged['inflation'], df_merged['unemployment'])
    print(f"\nGlobalna korelacja Pearsona: {pearson_corr:.3f} (p={pearson_p:.3g})")
    print(f"Globalna korelacja Spearmana: {spearman_corr:.3f} (p={spearman_p:.3g})")
else:
    print("Brak wspólnych danych do analizy korelacji.")

# 8. Korelacje dla krajów
results = []
for country in sorted(df_merged['geo'].unique()):
    sub = df_merged[df_merged['geo'] == country]
    if len(sub) > 2:
        pearson, p_p = stats.pearsonr(sub['inflation'], sub['unemployment'])
        spearman, p_s = stats.spearmanr(sub['inflation'], sub['unemployment'])
        results.append({'geo': country, 'pearson': pearson, 'pearson_p': p_p, 'spearman': spearman, 'spearman_p': p_s, 'n': len(sub)})

corr_df = pd.DataFrame(results)
corr_df.to_csv('wyniki_korelacji_kraj.csv', index=False)
print("\nTop 10 krajów wg korelacji Pearsona:")
print(corr_df.sort_values('pearson', ascending=False).head(10))

# 9. Wizualizacje
sns.set(font_scale=1.1)

# a) Heatmapa korelacji krajów
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df.pivot_table(index='geo', values='pearson').sort_values('pearson', ascending=False), annot=True, cmap='coolwarm', center=0)
plt.title('Korelacja Pearsona inflacja–bezrobocie (kraje)')
plt.tight_layout()
plt.savefig('heatmapa_korelacji.png')
plt.close()

# b) Wykres rozrzutu globalny
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_merged, x='inflation', y='unemployment', alpha=0.5)
sns.regplot(data=df_merged, x='inflation', y='unemployment', scatter=False, color='red', line_kws={'linestyle':'--'})
plt.xlabel('Inflacja (%)')
plt.ylabel('Stopa bezrobocia (%)')
plt.title('Zależność inflacja–bezrobocie (wszystkie kraje i okresy)')
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_global.png')
plt.close()

# c) Trendy dla przykładowego kraju
if not df_merged.empty:
    sample_geo = df_merged['geo'].value_counts().idxmax()
    df_sample = df_merged[df_merged['geo'] == sample_geo]
    plt.figure(figsize=(15, 6))
    plt.plot(df_sample['period'], df_sample['inflation'], marker='o', label='Inflacja')
    plt.plot(df_sample['period'], df_sample['unemployment'], marker='o', label='Bezrobocie')
    plt.title(f'Trendy inflacji i bezrobocia: {sample_geo}')
    plt.xlabel('Okres')
    plt.ylabel('Wartość (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trendy_sample.png')
    plt.close()

# d) Boxploty i histogramy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(data=df_merged['inflation'], ax=axes[0,0])
axes[0,0].set_title('Boxplot inflacji')
sns.boxplot(data=df_merged['unemployment'], ax=axes[0,1], color='red')
axes[0,1].set_title('Boxplot bezrobocia')
sns.histplot(data=df_merged['inflation'], ax=axes[1,0], kde=True)
axes[1,0].set_title('Histogram inflacji')
sns.histplot(data=df_merged['unemployment'], ax=axes[1,1], kde=True, color='red')
axes[1,1].set_title('Histogram bezrobocia')
plt.tight_layout()
plt.savefig('box_hist.png')
plt.close()

print("\nWyniki i wykresy zapisano do plików: wyniki_korelacji_kraj.csv, heatmapa_korelacji.png, scatter_global.png, trendy_sample.png, box_hist.png") 