#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zaawansowana analiza korelacji między bezrobociem a inflacją
Dane źródłowe: inflation_small.csv i unemployment_esmall.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, jarque_bera, shapiro
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja wyświetlania
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
sns.set_palette("Set2")

def zaawansowane_czyszczenie_danych(df, kolumna_wartosci):
    """
    Zaawansowane czyszczenie danych z detekcją outlierów
    """
    print(f"\nZaawansowane czyszczenie danych dla kolumny: {kolumna_wartosci}")
    print(f"Rozmiar przed czyszczeniem: {len(df)}")
    
    # Usunięcie wartości null i nieskończonych
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=[kolumna_wartosci])
    
    # Detekcja outlierów metodą IQR
    Q1 = df_clean[kolumna_wartosci].quantile(0.25)
    Q3 = df_clean[kolumna_wartosci].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df_clean[kolumna_wartosci] < lower_bound) | (df_clean[kolumna_wartosci] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    print(f"Znaleziono {outliers_count} outlierów ({outliers_count/len(df_clean)*100:.1f}%)")
    print(f"Granice outlierów: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Opcjonalnie usuń skrajne outliers (powyżej 3 odchyleń standardowych)
    mean_val = df_clean[kolumna_wartosci].mean()
    std_val = df_clean[kolumna_wartosci].std()
    extreme_outliers = (abs(df_clean[kolumna_wartosci] - mean_val) > 3 * std_val)
    
    if extreme_outliers.sum() > 0:
        print(f"Usuwam {extreme_outliers.sum()} skrajnych outlierów (>3σ)")
        df_clean = df_clean[~extreme_outliers]
    
    print(f"Rozmiar po czyszczeniu: {len(df_clean)}")
    
    return df_clean

def wczytaj_dane_inflacja(plik_csv):
    """
    Wczytuje i przetwarza dane o inflacji z zaawansowanym czyszczeniem
    """
    print("="*60)
    print("WCZYTYWANIE I CZYSZCZENIE DANYCH INFLACJI")
    print("="*60)
    
    df_inflacja = pd.read_csv(plik_csv)
    
    print(f"Pierwotny kształt danych: {df_inflacja.shape}")
    print(f"Kolumny: {list(df_inflacja.columns)}")
    
    # Podstawowe informacje o danych
    print("\nPodstawowe informacje:")
    print(f"- Zakres dat: {df_inflacja['TIME_PERIOD'].min()} do {df_inflacja['TIME_PERIOD'].max()}")
    print(f"- Liczba krajów: {df_inflacja['geo'].nunique()}")
    print(f"- Wartości null w OBS_VALUE: {df_inflacja['OBS_VALUE'].isnull().sum()}")
    
    # Konwersja TIME_PERIOD na format daty
    df_inflacja['TIME_PERIOD'] = pd.to_datetime(df_inflacja['TIME_PERIOD'])
    
    # Selekcja kolumn
    df_inflacja_clean = df_inflacja[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_inflacja_clean.columns = ['Kraj', 'Data', 'Inflacja']
    
    # Zaawansowane czyszczenie
    df_inflacja_clean = zaawansowane_czyszczenie_danych(df_inflacja_clean, 'Inflacja')
    
    # Dodanie dodatkowych kolumn czasowych
    df_inflacja_clean['Rok'] = df_inflacja_clean['Data'].dt.year
    df_inflacja_clean['Miesiac'] = df_inflacja_clean['Data'].dt.month
    df_inflacja_clean['Kwartal'] = df_inflacja_clean['Data'].dt.quarter
    
    print(f"\nFinalne dane inflacji: {df_inflacja_clean.shape}")
    
    return df_inflacja_clean

def wczytaj_dane_bezrobocie(plik_csv):
    """
    Wczytuje i przetwarza dane o bezrobociu z zaawansowanym czyszczeniem
    """
    print("\n" + "="*60)
    print("WCZYTYWANIE I CZYSZCZENIE DANYCH BEZROBOCIA")
    print("="*60)
    
    df_bezrobocie = pd.read_csv(plik_csv)
    
    print(f"Pierwotny kształt danych: {df_bezrobocie.shape}")
    print(f"Kolumny: {list(df_bezrobocie.columns)}")
    
    # Podstawowe informacje
    print(f"\nGrupy demograficzne: {df_bezrobocie['age'].unique()}")
    print(f"Płcie: {df_bezrobocie['sex'].unique()}")
    print(f"Jednostki: {df_bezrobocie['unit'].unique()}")
    
    # Konwersja daty
    df_bezrobocie['TIME_PERIOD'] = pd.to_datetime(df_bezrobocie['TIME_PERIOD'])
    
    # Agregacja po kraju i dacie
    df_bezrobocie_agg = df_bezrobocie.groupby(['geo', 'TIME_PERIOD']).agg({
        'OBS_VALUE': ['sum', 'mean', 'count']
    }).reset_index()
    
    # Spłaszczenie kolumn
    df_bezrobocie_agg.columns = ['Kraj', 'Data', 'Bezrobocie_suma', 'Bezrobocie_srednia', 'Liczba_grup']
    
    # Wybór głównej metryki (suma)
    df_bezrobocie_clean = df_bezrobocie_agg[['Kraj', 'Data', 'Bezrobocie_suma']].copy()
    df_bezrobocie_clean.columns = ['Kraj', 'Data', 'Bezrobocie']
    
    # Zaawansowane czyszczenie
    df_bezrobocie_clean = zaawansowane_czyszczenie_danych(df_bezrobocie_clean, 'Bezrobocie')
    
    # Dodanie kolumn czasowych
    df_bezrobocie_clean['Rok'] = df_bezrobocie_clean['Data'].dt.year
    df_bezrobocie_clean['Miesiac'] = df_bezrobocie_clean['Data'].dt.month
    df_bezrobocie_clean['Kwartal'] = df_bezrobocie_clean['Data'].dt.quarter
    
    print(f"\nFinalne dane bezrobocia: {df_bezrobocie_clean.shape}")
    
    return df_bezrobocie_clean

def lacz_i_przygotuj_dane(df_inflacja, df_bezrobocie):
    """
    Łączy dane i przeprowadza dodatkowe przygotowania
    """
    print("\n" + "="*60)
    print("ŁĄCZENIE I PRZYGOTOWANIE DANYCH")
    print("="*60)
    
    # Wyświetl informacje o krajach
    kraje_inflacja = set(df_inflacja['Kraj'].unique())
    kraje_bezrobocie = set(df_bezrobocie['Kraj'].unique())
    wspolne_kraje = kraje_inflacja.intersection(kraje_bezrobocie)
    
    print(f"Kraje tylko w danych inflacji: {len(kraje_inflacja - kraje_bezrobocie)}")
    print(f"Kraje tylko w danych bezrobocia: {len(kraje_bezrobocie - kraje_inflacja)}")
    print(f"Wspólne kraje: {len(wspolne_kraje)}")
    
    # Łączenie danych
    df_merged = pd.merge(df_inflacja, df_bezrobocie, on=['Kraj', 'Data'], how='inner')
    
    if len(df_merged) == 0:
        print("UWAGA: Brak wspólnych obserwacji!")
        return df_merged
    
    # Dodanie dodatkowych zmiennych
    df_merged['Inflacja_lag1'] = df_merged.groupby('Kraj')['Inflacja'].shift(1)
    df_merged['Bezrobocie_lag1'] = df_merged.groupby('Kraj')['Bezrobocie'].shift(1)
    
    # Normalizacja danych (z-score)
    scaler = StandardScaler()
    df_merged['Inflacja_norm'] = scaler.fit_transform(df_merged[['Inflacja']])
    df_merged['Bezrobocie_norm'] = scaler.fit_transform(df_merged[['Bezrobocie']])
    
    # Kategoryzacja krajów według wielkości gospodarki (na podstawie bezrobocia)
    mediana_bezrobocia = df_merged.groupby('Kraj')['Bezrobocie'].median()
    df_merged['Kategoria_kraju'] = df_merged['Kraj'].map(
        lambda x: 'Wysokie bezrobocie' if mediana_bezrobocia[x] > mediana_bezrobocia.median() 
        else 'Niskie bezrobocie'
    )
    
    print(f"Finalne połączone dane: {df_merged.shape}")
    print(f"Zakres dat: {df_merged['Data'].min()} do {df_merged['Data'].max()}")
    
    return df_merged

def testy_statystyczne(df):
    """
    Przeprowadza zaawansowane testy statystyczne
    """
    print("\n" + "="*60)
    print("ZAAWANSOWANE TESTY STATYSTYCZNE")
    print("="*60)
    
    # Test normalności
    print("\n1. TESTY NORMALNOŚCI:")
    
    # Shapiro-Wilk (dla małych próbek)
    if len(df) <= 5000:
        stat_shapiro_inf, p_shapiro_inf = shapiro(df['Inflacja'])
        stat_shapiro_bez, p_shapiro_bez = shapiro(df['Bezrobocie'])
        print(f"Shapiro-Wilk test:")
        print(f"  Inflacja: statystyka={stat_shapiro_inf:.4f}, p-value={p_shapiro_inf:.4f}")
        print(f"  Bezrobocie: statystyka={stat_shapiro_bez:.4f}, p-value={p_shapiro_bez:.4f}")
    
    # Jarque-Bera test
    stat_jb_inf, p_jb_inf = jarque_bera(df['Inflacja'])
    stat_jb_bez, p_jb_bez = jarque_bera(df['Bezrobocie'])
    print(f"\nJarque-Bera test:")
    print(f"  Inflacja: statystyka={stat_jb_inf:.4f}, p-value={p_jb_inf:.4f}")
    print(f"  Bezrobocie: statystyka={stat_jb_bez:.4f}, p-value={p_jb_bez:.4f}")
    
    # Test homoskedastyczności (Levene)
    print("\n2. TEST HOMOSKEDASTYCZNOŚCI:")
    # Podziel dane na grupy według mediany inflacji
    mediana_inflacji = df['Inflacja'].median()
    grupa1 = df[df['Inflacja'] <= mediana_inflacji]['Bezrobocie']
    grupa2 = df[df['Inflacja'] > mediana_inflacji]['Bezrobocie']
    
    stat_levene, p_levene = stats.levene(grupa1, grupa2)
    print(f"Levene test: statystyka={stat_levene:.4f}, p-value={p_levene:.4f}")
    
    # Korelacje różnego typu
    print("\n3. RÓŻNE TYPY KORELACJI:")
    
    pearson_r, pearson_p = stats.pearsonr(df['Bezrobocie'], df['Inflacja'])
    spearman_r, spearman_p = stats.spearmanr(df['Bezrobocie'], df['Inflacja'])
    kendall_r, kendall_p = stats.kendalltau(df['Bezrobocie'], df['Inflacja'])
    
    print(f"Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")
    print(f"Kendall:  r={kendall_r:.4f}, p={kendall_p:.4f}")
    
    return {
        'pearson': (pearson_r, pearson_p),
        'spearman': (spearman_r, spearman_p),
        'kendall': (kendall_r, kendall_p)
    }

def analiza_regresji(df):
    """
    Przeprowadza analizę regresji liniowej
    """
    print("\n4. ANALIZA REGRESJI LINIOWEJ:")
    
    X = df[['Bezrobocie']].values
    y = df['Inflacja'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"Współczynnik determinacji (R²): {r2:.4f}")
    print(f"Błąd średniokwadratowy (MSE): {mse:.4f}")
    print(f"Współczynnik regresji: {model.coef_[0]:.4f}")
    print(f"Wyraz wolny: {model.intercept_:.4f}")
    
    return model, r2

def analiza_skupien(df):
    """
    Przeprowadza analizę skupień K-means
    """
    print("\n5. ANALIZA SKUPIEŃ (K-MEANS):")
    
    # Przygotowanie danych
    X = df[['Bezrobocie_norm', 'Inflacja_norm']].values
    
    # Znajdź optymalną liczbę skupień (metoda łokcia)
    inertias = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Wybierz k=3 jako dobry kompromis
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Klaster'] = kmeans_final.fit_predict(X)
    
    print(f"Liczba obserwacji w każdym klastrze:")
    print(df['Klaster'].value_counts().sort_index())
    
    return df

def zaawansowane_wizualizacje(df, korelacje):
    """
    Tworzy zaawansowane wizualizacje
    """
    print("\n6. TWORZENIE ZAAWANSOWANYCH WIZUALIZACJI...")
    
    # Ustawienia kolorów
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Główny wykres - siatka 3x3
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Scatter plot z regresją i klastrami
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(df['Bezrobocie'], df['Inflacja'], 
                         c=df['Klaster'], cmap='viridis', alpha=0.7, s=80)
    
    # Linia regresji
    z = np.polyfit(df['Bezrobocie'], df['Inflacja'], 1)
    p = np.poly1d(z)
    ax1.plot(df['Bezrobocie'], p(df['Bezrobocie']), "r--", alpha=0.8, linewidth=3)
    
    ax1.set_xlabel('Bezrobocie (tys. osób)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inflacja (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Bezrobocie vs Inflacja (z klastrami)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Klaster')
    
    # 2. Histogram 2D
    ax2 = fig.add_subplot(gs[0, 1])
    hist = ax2.hist2d(df['Bezrobocie'], df['Inflacja'], bins=20, cmap='Blues')
    ax2.set_xlabel('Bezrobocie (tys. osób)', fontweight='bold')
    ax2.set_ylabel('Inflacja (%)', fontweight='bold')
    ax2.set_title('Histogram 2D', fontsize=14, fontweight='bold')
    plt.colorbar(hist[3], ax=ax2, label='Częstość')
    
    # 3. Boxplot porównawczy
    ax3 = fig.add_subplot(gs[0, 2])
    df_melt = pd.melt(df, value_vars=['Inflacja_norm', 'Bezrobocie_norm'], 
                      var_name='Zmienna', value_name='Wartość_znormalizowana')
    sns.boxplot(data=df_melt, x='Zmienna', y='Wartość_znormalizowana', ax=ax3)
    ax3.set_title('Porównanie rozkładów (znormalizowane)', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(['Inflacja', 'Bezrobocie'])
    
    # 4. Korelacje po krajach
    ax4 = fig.add_subplot(gs[1, 0])
    korelacje_kraje = []
    kraje_z_danymi = []
    
    for kraj in df['Kraj'].unique():
        dane_kraj = df[df['Kraj'] == kraj]
        if len(dane_kraj) >= 3:
            r, p = stats.pearsonr(dane_kraj['Bezrobocie'], dane_kraj['Inflacja'])
            korelacje_kraje.append(r)
            kraje_z_danymi.append(kraj)
    
    if korelacje_kraje:
        y_pos = np.arange(len(kraje_z_danymi))
        bars = ax4.barh(y_pos, korelacje_kraje, color=colors[:len(korelacje_kraje)])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(kraje_z_danymi, fontsize=10)
        ax4.set_xlabel('Współczynnik korelacji', fontweight='bold')
        ax4.set_title('Korelacja po krajach', fontsize=14, fontweight='bold')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.grid(True, alpha=0.3)
    
    # 5. Szeregi czasowe
    ax5 = fig.add_subplot(gs[1, 1])
    df_czasowy = df.groupby('Data').agg({
        'Inflacja': 'mean',
        'Bezrobocie': lambda x: x.mean() / 1000  # Skalowanie dla lepszej wizualizacji
    }).reset_index()
    
    ax5.plot(df_czasowy['Data'], df_czasowy['Inflacja'], 
             color='red', linewidth=3, label='Inflacja (%)', marker='o')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(df_czasowy['Data'], df_czasowy['Bezrobocie'], 
                  color='blue', linewidth=3, label='Bezrobocie (tys.)', marker='s')
    
    ax5.set_xlabel('Data', fontweight='bold')
    ax5.set_ylabel('Inflacja (%)', color='red', fontweight='bold')
    ax5_twin.set_ylabel('Bezrobocie (tys.)', color='blue', fontweight='bold')
    ax5.set_title('Trendy czasowe', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Violin plot
    ax6 = fig.add_subplot(gs[1, 2])
    df_kategorie = df.melt(id_vars=['Kategoria_kraju'], 
                          value_vars=['Inflacja', 'Bezrobocie'],
                          var_name='Wskaźnik', value_name='Wartość')
    
    sns.violinplot(data=df_kategorie, x='Kategoria_kraju', y='Wartość', 
                   hue='Wskaźnik', ax=ax6, split=True)
    ax6.set_title('Rozkłady według kategorii krajów', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Macierz korelacji rozszerzona
    ax7 = fig.add_subplot(gs[2, 0])
    corr_cols = ['Inflacja', 'Bezrobocie', 'Inflacja_lag1', 'Bezrobocie_lag1']
    correlation_matrix = df[corr_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax7, fmt='.3f', cbar_kws={'shrink': 0.8})
    ax7.set_title('Macierz korelacji (z opóźnieniami)', fontsize=14, fontweight='bold')
    
    # 8. Q-Q plot
    ax8 = fig.add_subplot(gs[2, 1])
    stats.probplot(df['Inflacja'], dist="norm", plot=ax8)
    ax8.set_title('Q-Q Plot - Inflacja', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Residuals plot
    ax9 = fig.add_subplot(gs[2, 2])
    X = df[['Bezrobocie']].values
    y = df['Inflacja'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    ax9.scatter(y_pred, residuals, alpha=0.6, color='purple')
    ax9.axhline(y=0, color='red', linestyle='--')
    ax9.set_xlabel('Wartości przewidywane', fontweight='bold')
    ax9.set_ylabel('Residua', fontweight='bold')
    ax9.set_title('Wykres residuów', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('ZAAWANSOWANA ANALIZA KORELACJI: BEZROBOCIE vs INFLACJA', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('zaawansowana_analiza_korelacji.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Główny wykres zapisany jako 'zaawansowana_analiza_korelacji.png'")

def wykres_interaktywny(df):
    """
    Tworzy interaktywny wykres z Plotly
    """
    print("\n7. TWORZENIE WYKRESÓW INTERAKTYWNYCH...")
    
    # Interaktywny scatter plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scatter Plot z Klastrami', 'Trendy Czasowe', 
                       'Korelacja po Krajach', 'Histogram 3D'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"type": "scatter3d"}]]
    )
    
    # Scatter plot z klastrami
    for klaster in df['Klaster'].unique():
        dane_klaster = df[df['Klaster'] == klaster]
        fig.add_trace(
            go.Scatter(x=dane_klaster['Bezrobocie'], y=dane_klaster['Inflacja'],
                      mode='markers', name=f'Klaster {klaster}',
                      marker=dict(size=8, opacity=0.7)),
            row=1, col=1
        )
    
    # Trendy czasowe
    df_czasowy = df.groupby('Data').agg({
        'Inflacja': 'mean',
        'Bezrobocie': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=df_czasowy['Data'], y=df_czasowy['Inflacja'],
                  mode='lines+markers', name='Inflacja', line=dict(color='red')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=df_czasowy['Data'], y=df_czasowy['Bezrobocie'],
                  mode='lines+markers', name='Bezrobocie', 
                  line=dict(color='blue'), yaxis='y2'),
        row=1, col=2, secondary_y=True
    )
    
    # Aktualizacja układu
    fig.update_layout(height=800, showlegend=True,
                     title_text="Interaktywna Analiza Bezrobocie vs Inflacja")
    
    # Zapisz jako HTML
    fig.write_html("interaktywna_analiza.html")
    print("   Interaktywny wykres zapisany jako 'interaktywna_analiza.html'")

def raport_koncowy(df, korelacje, model, r2):
    """
    Generuje szczegółowy raport końcowy
    """
    print("\n" + "="*80)
    print("SZCZEGÓŁOWY RAPORT KOŃCOWY")
    print("="*80)
    
    print(f"\n📊 PODSUMOWANIE DANYCH:")
    print(f"   • Liczba obserwacji: {len(df):,}")
    print(f"   • Liczba krajów: {df['Kraj'].nunique()}")
    print(f"   • Zakres czasowy: {df['Data'].min().strftime('%Y-%m')} - {df['Data'].max().strftime('%Y-%m')}")
    print(f"   • Średnia inflacja: {df['Inflacja'].mean():.2f}% (±{df['Inflacja'].std():.2f})")
    print(f"   • Średnie bezrobocie: {df['Bezrobocie'].mean():.0f} tys. osób (±{df['Bezrobocie'].std():.0f})")
    
    print(f"\n📈 WYNIKI KORELACJI:")
    pearson_r, pearson_p = korelacje['pearson']
    spearman_r, spearman_p = korelacje['spearman']
    kendall_r, kendall_p = korelacje['kendall']
    
    print(f"   • Pearson:  r = {pearson_r:+.4f} (p = {pearson_p:.4f})")
    print(f"   • Spearman: r = {spearman_r:+.4f} (p = {spearman_p:.4f})")
    print(f"   • Kendall:  r = {kendall_r:+.4f} (p = {kendall_p:.4f})")
    
    # Interpretacja
    if abs(pearson_r) < 0.1:
        sila = "bardzo słaba"
    elif abs(pearson_r) < 0.3:
        sila = "słaba"
    elif abs(pearson_r) < 0.5:
        sila = "umiarkowana"
    elif abs(pearson_r) < 0.7:
        sila = "silna"
    else:
        sila = "bardzo silna"
    
    kierunek = "pozytywna" if pearson_r > 0 else "negatywna"
    istotnosc = "istotna statystycznie" if pearson_p < 0.05 else "nieistotna statystycznie"
    
    print(f"\n💡 INTERPRETACJA:")
    print(f"   • Korelacja jest {sila} i {kierunek}")
    print(f"   • Korelacja jest {istotnosc} (α = 0.05)")
    
    print(f"\n📐 MODEL REGRESJI:")
    print(f"   • R² = {r2:.4f} ({r2*100:.1f}% wariancji wyjaśnione)")
    print(f"   • Równanie: Inflacja = {model.intercept_:.3f} + {model.coef_[0]:.6f} × Bezrobocie")
    
    print(f"\n🎯 WNIOSKI EKONOMICZNE:")
    if pearson_r < -0.2:
        print("   • Wyniki sugerują negatywną korelację zgodną z krzywą Phillipsa")
        print("   • Wzrost bezrobocia może być związany ze spadkiem inflacji")
    elif pearson_r > 0.2:
        print("   • Wyniki wskazują na pozytywną korelację")
        print("   • Może to świadczyć o stagflacji lub innych czynnikach strukturalnych")
    else:
        print("   • Korelacja jest słaba, co może oznaczać:")
        print("     - Brak prostej liniowej zależności")
        print("     - Wpływ innych czynników ekonomicznych")
        print("     - Różnice w mechanizmach między krajami")
    
    print(f"\n📁 WYGENEROWANE PLIKI:")
    print("   • zaawansowana_analiza_korelacji.png - główne wizualizacje")
    print("   • interaktywna_analiza.html - interaktywny wykres")
    print("   • dane_oczyszczone.csv - oczyszczone dane do dalszych analiz")
    
    # Zapisz oczyszczone dane
    df.to_csv('dane_oczyszczone.csv', index=False)
    print("   • Dane zapisane w formacie CSV")

def main():
    """
    Główna funkcja programu z rozszerzoną analizą
    """
    print("🔍 ZAAWANSOWANA ANALIZA KORELACJI: BEZROBOCIE vs INFLACJA")
    print("=" * 80)
    
    try:
        # 1. Wczytanie i czyszczenie danych
        df_inflacja = wczytaj_dane_inflacja('inflation_small.csv')
        df_bezrobocie = wczytaj_dane_bezrobocie('unemployment_esmall.csv')
        
        # 2. Łączenie danych
        df = lacz_i_przygotuj_dane(df_inflacja, df_bezrobocie)
        
        if len(df) == 0:
            print("❌ BŁĄD: Brak danych do analizy!")
            return
        
        # 3. Zaawansowane testy statystyczne
        korelacje = testy_statystyczne(df)
        
        # 4. Analiza regresji
        model, r2 = analiza_regresji(df)
        
        # 5. Analiza skupień
        df = analiza_skupien(df)
        
        # 6. Zaawansowane wizualizacje
        zaawansowane_wizualizacje(df, korelacje)
        
        # 7. Wykres interaktywny
        try:
            wykres_interaktywny(df)
        except ImportError:
            print("   Plotly nie jest dostępny - pomijam interaktywne wykresy")
        
        # 8. Raport końcowy
        raport_koncowy(df, korelacje, model, r2)
        
        print(f"\n✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
        
    except FileNotFoundError as e:
        print(f"❌ BŁĄD: Nie znaleziono pliku: {e}")
        print("Upewnij się, że pliki są w odpowiednim folderze.")
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 