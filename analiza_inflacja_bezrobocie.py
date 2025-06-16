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
    Funkcja do zaawansowanego czyszczenia danych:
    - Usuwa wartości null i nieskończone
    - Wykrywa i obsługuje wartości odstające
    - Usuwa skrajne wartości odstające
    """
    print(f"\nZaawansowane czyszczenie danych dla kolumny: {kolumna_wartosci}")
    print(f"Rozmiar przed czyszczeniem: {len(df)}")

    # Inicjalizacja df_clean
    df_clean = df.copy()

    # Usunięcie wartości null i nieskończonych
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

    print(f"Znaleziono {outliers_count} outlierów ({outliers_count / len(df_clean) * 100:.1f}%)")
    print(f"Granice outlierów: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Usunięcie skrajnych outlierów (>3 odchylenia standardowe)
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
    Wczytuje i przetwarza dane o inflacji:
    - Konwertuje daty
    - Wybiera istotne kolumny
    - Dodaje kolumny czasowe (rok, miesiąc, kwartał)
    - Czyści dane z wartości odstających
    """
    print("=" * 60)
    print("WCZYTYWANIE I CZYSZCZENIE DANYCH INFLACJI")
    print("=" * 60)

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

    # Usunięcie danych dla United States
    df_inflacja = df_inflacja[df_inflacja['geo'] != 'United States'].copy()

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
    Wczytuje i przetwarza dane o bezrobociu:
    - Agreguje dane po kraju i dacie
    - Oblicza sumę i średnią wartości
    - Dodaje kolumny czasowe
    - Czyści dane z wartości odstających
    - Zachowuje informacje o płci
    """
    print("\n" + "=" * 60)
    print("WCZYTYWANIE I CZYSZCZENIE DANYCH BEZROBOCIA")
    print("=" * 60)

    df_bezrobocie = pd.read_csv(plik_csv)

    print(f"Pierwotny kształt danych: {df_bezrobocie.shape}")
    print(f"Kolumny: {list(df_bezrobocie.columns)}")

    # Podstawowe informacje
    print(f"\nGrupy demograficzne: {df_bezrobocie['age'].unique()}")
    print(f"Płcie: {df_bezrobocie['sex'].unique()}")
    print(f"Jednostki: {df_bezrobocie['unit'].unique()}")

    # Konwersja daty
    df_bezrobocie['TIME_PERIOD'] = pd.to_datetime(df_bezrobocie['TIME_PERIOD'])

    # Usunięcie danych dla United States
    df_bezrobocie = df_bezrobocie[df_bezrobocie['geo'] != 'United States'].copy()
    print(f"Usunięto dane dla United States. Rozmiar po usunięciu: {len(df_bezrobocie)}")

    # Zachowanie danych o płci
    df_bezrobocie_sex = df_bezrobocie[df_bezrobocie['sex'].isin(['Females', 'Males'])].copy()
    
    # Agregacja po kraju, dacie i płci
    df_bezrobocie_sex_agg = df_bezrobocie_sex.groupby(['geo', 'TIME_PERIOD', 'sex']).agg({
        'OBS_VALUE': 'mean'
    }).reset_index()

    # Pivotowanie danych dla analizy różnic płci
    df_bezrobocie_sex_pivot = df_bezrobocie_sex_agg.pivot_table(
        index=['geo', 'TIME_PERIOD'],
        columns='sex',
        values='OBS_VALUE'
    ).reset_index()

    # Obliczenie różnicy między płciami (Males - Females)
    df_bezrobocie_sex_pivot['Gender_Gap'] = df_bezrobocie_sex_pivot['Males'] - df_bezrobocie_sex_pivot['Females']
    
    # Agregacja po kraju i dacie (dla głównej analizy)
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

    return df_bezrobocie_clean, df_bezrobocie_sex_pivot


def lacz_i_przygotuj_dane(df_inflacja, df_bezrobocie):
    """
    Łączy dane o inflacji i bezrobociu:
    - Znajduje wspólne kraje
    - Łączy dane po kraju i dacie
    - Dodaje zmienne opóźnione (lag)
    - Normalizuje dane
    - Kategoryzuje kraje według poziomu bezrobocia
    """
    print("\n" + "=" * 60)
    print("ŁĄCZENIE I PRZYGOTOWANIE DANYCH")
    print("=" * 60)

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

    # Dodanie zmiennych opóźnionych
    df_merged['Inflacja_lag1'] = df_merged.groupby('Kraj')['Inflacja'].shift(1)
    df_merged['Bezrobocie_lag1'] = df_merged.groupby('Kraj')['Bezrobocie'].shift(1)

    # Normalizacja danych (z-score)
    scaler = StandardScaler()
    df_merged['Inflacja_norm'] = scaler.fit_transform(df_merged[['Inflacja']])
    df_merged['Bezrobocie_norm'] = scaler.fit_transform(df_merged[['Bezrobocie']])

    # Kategoryzacja krajów według poziomu bezrobocia
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
    Przeprowadza zaawansowane testy statystyczne:
    - Testy normalności (Shapiro-Wilk, Jarque-Bera)
    - Test homoskedastyczności (Levene)
    - Różne typy korelacji (Pearson, Spearman, Kendall)
    """
    print("\n" + "=" * 60)
    print("ZAAWANSOWANE TESTY STATYSTYCZNE")
    print("=" * 60)

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
    Przeprowadza analizę regresji liniowej:
    - Buduje model regresji liniowej
    - Oblicza współczynnik determinacji R²
    - Oblicza błąd średniokwadratowy MSE
    - Zwraca model i współczynnik determinacji
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
    Przeprowadza analizę skupień K-means:
    - Znajduje optymalną liczbę skupień metodą łokcia
    - Przypisuje obserwacje do klastrów
    - Zwraca dane z dodaną kolumną klastrów
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


def zaawansowane_wizualizacje(df, korelacje, df_bezrobocie_sex=None):
    """
    Tworzy zaawansowane wizualizacje:
    1. Wykres rozproszenia z regresją i klastrami
    2. Szereg czasowy inflacji i bezrobocia
    3. Korelacje w poszczególnych krajach
    4. Analiza różnic płci w bezrobociu (jeśli dostępne dane)
    """
    print("\n6. TWORZENIE WIZUALIZACJI...")

    if df_bezrobocie_sex is not None:
        fig, axes = plt.subplots(2, 2, figsize=(22, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        axes = axes.flatten()

    # 1. Wykres rozproszenia z regresją i klastrami
    scatter = axes[0].scatter(df['Bezrobocie'], df['Inflacja'],
                              c=df['Klaster'], cmap='viridis', alpha=0.7, s=80)
    z = np.polyfit(df['Bezrobocie'], df['Inflacja'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['Bezrobocie'], p(df['Bezrobocie']), "r--", label='Regresja liniowa')
    axes[0].set_title('Inflacja vs Bezrobocie (z klastrami)')
    axes[0].set_xlabel('Bezrobocie (tys.)')
    axes[0].set_ylabel('Inflacja (%)')
    axes[0].legend()
    plt.colorbar(scatter, ax=axes[0], label='Klaster')
    axes[0].grid(True)

    # 2. Szereg czasowy inflacji i bezrobocia
    df_czasowy = df.groupby('Data').agg({
        'Inflacja': 'mean',
        'Bezrobocie': lambda x: x.mean() / 1000
    }).reset_index()

    axes[1].plot(df_czasowy['Data'], df_czasowy['Inflacja'], 'r-', label='Inflacja (%)')
    ax2_twin = axes[1].twinx()
    ax2_twin.plot(df_czasowy['Data'], df_czasowy['Bezrobocie'], 'b--', label='Bezrobocie (mln.)')
    axes[1].set_title('Trendy inflacji i bezrobocia w czasie')
    axes[1].set_xlabel('Data')
    axes[1].set_ylabel('Inflacja (%)', color='red')
    ax2_twin.set_ylabel('Bezrobocie (mln.)', color='blue')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(True)

    # 3. Korelacje w krajach
    kraje = []
    korelacje_krajowe = []
    for kraj in df['Kraj'].unique():
        dane = df[df['Kraj'] == kraj]
        if len(dane) >= 3:
            r, _ = stats.pearsonr(dane['Bezrobocie'], dane['Inflacja'])
            kraje.append(kraj)
            korelacje_krajowe.append(r)

    axes[2].barh(kraje, korelacje_krajowe, color='skyblue')
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].set_title('Korelacja inflacja–bezrobocie w krajach')
    axes[2].set_xlabel('Współczynnik korelacji')
    axes[2].grid(True, axis='x')

    # 4. Analiza różnic płci w bezrobociu (jeśli dostępne dane)
    if df_bezrobocie_sex is not None:
        # Obliczenie średniej różnicy płci dla każdego kraju
        gender_gap_by_country = df_bezrobocie_sex.groupby('geo')['Gender_Gap'].mean().sort_values()
        
        # Wykres różnic płci
        bars = axes[3].barh(gender_gap_by_country.index, gender_gap_by_country.values, color='purple')
        axes[3].axvline(0, color='red', linestyle='--', label='Brak różnicy')
        
        
        axes[3].set_title('Różnica w bezrobociu między płciami (Males-Females)')
        axes[3].set_xlabel('Różnica w stopie bezrobocia (Males-Females) [tys. osób]')
        axes[3].grid(True, axis='x')

    # Finalizacja wykresów
    plt.suptitle('Analiza inflacji i bezrobocia', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('wizualizacja_skoncentrowana.png', dpi=300)
    plt.show()

    print("   Zapisano uproszczoną wizualizację jako 'wizualizacja_skoncentrowana.png'")


def raport_koncowy(df, korelacje, model, r2):
    """
    Generuje szczegółowy raport końcowy zawierający:
    - Podsumowanie danych
    - Wyniki korelacji
    - Interpretację wyników
    - Wnioski ekonomiczne
    - Informacje o wygenerowanych plikach
    """
    print("\n" + "=" * 80)
    print("SZCZEGÓŁOWY RAPORT KOŃCOWY")
    print("=" * 80)

    print(f"\nPODSUMOWANIE DANYCH:")
    print(f"   • Liczba obserwacji: {len(df):,}")
    print(f"   • Liczba krajów: {df['Kraj'].nunique()}")
    print(f"   • Zakres czasowy: {df['Data'].min().strftime('%Y-%m')} - {df['Data'].max().strftime('%Y-%m')}")
    print(f"   • Średnia inflacja: {df['Inflacja'].mean():.2f}% (±{df['Inflacja'].std():.2f})")
    print(f"   • Średnie bezrobocie: {df['Bezrobocie'].mean():.0f} tys. osób (±{df['Bezrobocie'].std():.0f})")

    print(f"\nWYNIKI KORELACJI:")
    pearson_r, pearson_p = korelacje['pearson']
    spearman_r, spearman_p = korelacje['spearman']
    kendall_r, kendall_p = korelacje['kendall']

    print(f"   • Pearson:  r = {pearson_r:+.4f} (p = {pearson_p:.4f})")
    print(f"   • Spearman: r = {spearman_r:+.4f} (p = {spearman_p:.4f})")
    print(f"   • Kendall:  r = {kendall_r:+.4f} (p = {kendall_p:.4f})")

    # Interpretacja siły i kierunku korelacji
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

    print(f"\nINTERPRETACJA:")
    print(f"   • Korelacja jest {sila} i {kierunek}")
    print(f"   • Korelacja jest {istotnosc} (α = 0.05)")

    print(f"\nMODEL REGRESJI:")
    print(f"   • R² = {r2:.4f} ({r2 * 100:.1f}% wariancji wyjaśnione)")
    print(f"   • Równanie: Inflacja = {model.intercept_:.3f} + {model.coef_[0]:.6f} × Bezrobocie")

    print(f"\nWNIOSKI EKONOMICZNE:")
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

    print(f"\n WYGENEROWANE PLIKI:")
    print("   • zaawansowana_analiza_korelacji.png - główne wizualizacje")
    print("   • dane_oczyszczone.csv - oczyszczone dane do dalszych analiz")

    # Zapisz oczyszczone dane
    df.to_csv('dane_oczyszczone.csv', index=False)
    print("   • Dane zapisane w formacie CSV")


def main():
    """
    Główna funkcja programu:
    1. Wczytuje i czyści dane
    2. Łączy dane
    3. Przeprowadza testy statystyczne
    4. Wykonuje analizę regresji
    5. Przeprowadza analizę skupień
    6. Tworzy wizualizacje
    7. Generuje raport końcowy
    """
    print("ZAAWANSOWANA ANALIZA KORELACJI: BEZROBOCIE vs INFLACJA")
    print("=" * 80)

    try:
        # 1. Wczytanie i czyszczenie danych
        df_inflacja = wczytaj_dane_inflacja('hicp_full.csv')
        df_bezrobocie, df_bezrobocie_sex = wczytaj_dane_bezrobocie('full_un.csv')

        # 2. Łączenie danych
        df = lacz_i_przygotuj_dane(df_inflacja, df_bezrobocie)

        if len(df) == 0:
            print("BŁĄD: Brak danych do analizy!")
            return

        # 3. Zaawansowane testy statystyczne
        korelacje = testy_statystyczne(df)

        # 4. Analiza regresji
        model, r2 = analiza_regresji(df)

        # 5. Analiza skupień
        df = analiza_skupien(df)

        # 6. Zaawansowane wizualizacje
        zaawansowane_wizualizacje(df, korelacje, df_bezrobocie_sex)

        # 7. Raport końcowy
        raport_koncowy(df, korelacje, model, r2)

        print(f"\nANALIZA ZAKOŃCZONA POMYŚLNIE!")

    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku: {e}")
        print("Upewnij się, że pliki są w odpowiednim folderze.")
    except Exception as e:
        print(f"BŁĄD: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 