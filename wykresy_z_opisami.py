#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator osobnych wykresów z opisami dla analizy bezrobocia i inflacji
Każdy wykres jest zapisywany jako osobny plik z prostym opisem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja wykresów
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def przygotuj_dane():
    """
    Przygotowuje dane do analiz (uproszczona wersja)
    """
    print("📂 Przygotowywanie danych...")
    
    # Wczytanie danych inflacji
    df_inflacja = pd.read_csv('inflation_small.csv')
    df_inflacja['TIME_PERIOD'] = pd.to_datetime(df_inflacja['TIME_PERIOD'])
    df_inflacja = df_inflacja[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_inflacja.columns = ['Kraj', 'Data', 'Inflacja']
    df_inflacja = df_inflacja.dropna()
    
    # Wczytanie danych bezrobocia
    df_bezrobocie = pd.read_csv('unemployment_esmall.csv')
    df_bezrobocie['TIME_PERIOD'] = pd.to_datetime(df_bezrobocie['TIME_PERIOD'])
    df_bezrobocie_agg = df_bezrobocie.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].sum().reset_index()
    df_bezrobocie_agg.columns = ['Kraj', 'Data', 'Bezrobocie']
    
    # Łączenie danych
    df = pd.merge(df_inflacja, df_bezrobocie_agg, on=['Kraj', 'Data'], how='inner')
    
    # Usunięcie skrajnych outlierów
    for col in ['Inflacja', 'Bezrobocie']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Dodanie normalizowanych danych
    scaler = StandardScaler()
    df['Inflacja_norm'] = scaler.fit_transform(df[['Inflacja']])
    df['Bezrobocie_norm'] = scaler.fit_transform(df[['Bezrobocie']])
    
    print(f"✅ Przygotowano {len(df)} obserwacji z {df['Kraj'].nunique()} krajów")
    return df

def wykres_1_scatter_podstawowy(df):
    """
    WYKRES 1: Podstawowy scatter plot
    """
    print("📊 Tworzenie wykresu 1: Podstawowy scatter plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(df['Bezrobocie'], df['Inflacja'], alpha=0.6, s=100, color='#3498db')
    
    # Linia trendu
    z = np.polyfit(df['Bezrobocie'], df['Inflacja'], 1)
    p = np.poly1d(z)
    plt.plot(df['Bezrobocie'], p(df['Bezrobocie']), "r-", linewidth=3, alpha=0.8)
    
    # Obliczenie korelacji
    correlation, p_value = stats.pearsonr(df['Bezrobocie'], df['Inflacja'])
    
    plt.xlabel('Bezrobocie (tysiące osób)', fontweight='bold', fontsize=14)
    plt.ylabel('Inflacja (%)', fontweight='bold', fontsize=14)
    plt.title('Związek między bezrobociem a inflacją', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Dodanie tekstu z korelacją
    plt.text(0.05, 0.95, f'Korelacja: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('1_scatter_podstawowy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Opis
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Każda kropka to jeden kraj w jednym miesiącu
• Oś X (pozioma): poziom bezrobocia w tysiącach osób
• Oś Y (pionowa): stopa inflacji w procentach
• Czerwona linia: pokazuje ogólny trend związku
• Jeśli linia idzie w górę → więcej bezrobocia = więcej inflacji
• Jeśli linia idzie w dół → więcej bezrobocia = mniej inflacji

💡 JAK TO INTERPRETOWAĆ:
""")
    if correlation > 0.1:
        print("✅ Korelacja pozytywna: kraje z wyższym bezrobociem mają wyższą inflację")
    elif correlation < -0.1:
        print("✅ Korelacja negatywna: kraje z wyższym bezrobociem mają niższą inflację")
    else:
        print("✅ Brak wyraźnego związku między bezrobociem a inflacją")

def wykres_2_histogram_porownawczy(df):
    """
    WYKRES 2: Porównanie rozkładów
    """
    print("📊 Tworzenie wykresu 2: Porównanie rozkładów...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram inflacji
    ax1.hist(df['Inflacja'], bins=25, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Inflacja (%)', fontweight='bold')
    ax1.set_ylabel('Liczba obserwacji', fontweight='bold')
    ax1.set_title('Rozkład inflacji', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Dodanie linii średniej
    mean_inf = df['Inflacja'].mean()
    ax1.axvline(mean_inf, color='red', linestyle='--', linewidth=2, 
                label=f'Średnia: {mean_inf:.1f}%')
    ax1.legend()
    
    # Histogram bezrobocia
    ax2.hist(df['Bezrobocie'], bins=25, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.set_xlabel('Bezrobocie (tys. osób)', fontweight='bold')
    ax2.set_ylabel('Liczba obserwacji', fontweight='bold')
    ax2.set_title('Rozkład bezrobocia', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Dodanie linii średniej
    mean_bez = df['Bezrobocie'].mean()
    ax2.axvline(mean_bez, color='blue', linestyle='--', linewidth=2, 
                label=f'Średnia: {mean_bez:.0f} tys.')
    ax2.legend()
    
    plt.suptitle('Jak często występują różne wartości bezrobocia i inflacji', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig('2_histogram_porownawczy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Lewy wykres: jak często występują różne poziomy inflacji
• Prawy wykres: jak często występują różne poziomy bezrobocia
• Wysokie słupki = częste wartości, niskie słupki = rzadkie wartości
• Czerwona/niebieska linia = średnia wartość

💡 JAK TO INTERPRETOWAĆ:
• Czy większość krajów ma podobną inflację/bezrobocie?
• Czy są kraje z bardzo wysokimi lub bardzo niskimi wartościami?
• Gdzie znajduje się "typowy" kraj?
""")

def wykres_3_trendy_czasowe(df):
    """
    WYKRES 3: Jak zmieniają się wskaźniki w czasie
    """
    print("📊 Tworzenie wykresu 3: Trendy czasowe...")
    
    # Grupowanie po miesiącach
    df_czasowy = df.groupby('Data').agg({
        'Inflacja': 'mean',
        'Bezrobocie': 'mean'
    }).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Wykres inflacji
    color = '#e74c3c'
    ax1.set_xlabel('Data', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Średnia inflacja (%)', color=color, fontweight='bold', fontsize=14)
    line1 = ax1.plot(df_czasowy['Data'], df_czasowy['Inflacja'], 
                     color=color, linewidth=3, marker='o', markersize=8, label='Inflacja')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Druga oś dla bezrobocia
    ax2 = ax1.twinx()
    color = '#3498db'
    ax2.set_ylabel('Średnie bezrobocie (tys. osób)', color=color, fontweight='bold', fontsize=14)
    line2 = ax2.plot(df_czasowy['Data'], df_czasowy['Bezrobocie'], 
                     color=color, linewidth=3, marker='s', markersize=8, label='Bezrobocie')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Tytuł i legenda
    plt.title('Jak zmieniały się inflacja i bezrobocie w czasie', 
              fontweight='bold', fontsize=16, pad=20)
    
    # Dodanie legendy
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('3_trendy_czasowe.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Czerwona linia: jak zmieniała się średnia inflacja w poszczególnych miesiącach
• Niebieska linia: jak zmieniało się średnie bezrobocie w tym samym czasie
• Oś X: czas (miesiące i lata)
• Dwie różne osie Y bo inflacja (%) i bezrobocie (tys. osób) mają różne skale

💡 JAK TO INTERPRETOWAĆ:
• Czy inflacja i bezrobocie rosną/maleją w tym samym czasie?
• Czy są widoczne trendy sezonowe?
• Kiedy były najwyższe/najniższe wartości?
""")

def wykres_4_korelacja_po_krajach(df):
    """
    WYKRES 4: Korelacja dla poszczególnych krajów
    """
    print("📊 Tworzenie wykresu 4: Korelacja po krajach...")
    
    # Obliczenie korelacji dla każdego kraju
    korelacje_kraje = []
    kraje_nazwy = []
    
    for kraj in df['Kraj'].unique():
        dane_kraj = df[df['Kraj'] == kraj]
        if len(dane_kraj) >= 4:  # Minimum 4 obserwacje
            r, p = stats.pearsonr(dane_kraj['Bezrobocie'], dane_kraj['Inflacja'])
            korelacje_kraje.append(r)
            kraje_nazwy.append(kraj)
    
    if not korelacje_kraje:
        print("❌ Brak wystarczających danych dla analizy po krajach")
        return
    
    # Sortowanie krajów według korelacji
    sorted_data = sorted(zip(kraje_nazwy, korelacje_kraje), key=lambda x: x[1], reverse=True)
    kraje_sorted, korelacje_sorted = zip(*sorted_data)
    
    plt.figure(figsize=(12, max(8, len(kraje_sorted) * 0.4)))
    
    # Kolorowanie słupków
    colors = ['#e74c3c' if r > 0 else '#3498db' for r in korelacje_sorted]
    
    bars = plt.barh(range(len(kraje_sorted)), korelacje_sorted, color=colors, alpha=0.7)
    
    # Dodanie linii zerowej
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    plt.yticks(range(len(kraje_sorted)), kraje_sorted)
    plt.xlabel('Siła związku (korelacja)', fontweight='bold', fontsize=14)
    plt.ylabel('Kraj', fontweight='bold', fontsize=14)
    plt.title('Jak silny jest związek bezrobocie-inflacja w różnych krajach', 
              fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Dodanie legendy
    plt.text(0.7, 0.95, 'Czerwony = pozytywny związek\nNiebieski = negatywny związek', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('4_korelacja_po_krajach.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Każdy słupek = jeden kraj
• Długość słupka = jak silny jest związek między bezrobociem a inflacją w tym kraju
• Czerwone słupki: pozytywny związek (więcej bezrobocia = więcej inflacji)
• Niebieskie słupki: negatywny związek (więcej bezrobocia = mniej inflacji)
• Słupki bliżej zera = słaby związek

💡 JAK TO INTERPRETOWAĆ:
• W których krajach związek jest najsilniejszy?
• Czy większość krajów ma podobny typ związku?
• Które kraje są wyjątkami?
""")

def wykres_5_grupy_krajow(df):
    """
    WYKRES 5: Grupowanie krajów według podobieństwa
    """
    print("📊 Tworzenie wykresu 5: Grupy podobnych krajów...")
    
    # Przygotowanie danych dla grupowania
    X = df[['Bezrobocie_norm', 'Inflacja_norm']].values
    
    # Grupowanie K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Grupa'] = kmeans.fit_predict(X)
    
    plt.figure(figsize=(12, 8))
    
    # Różne kolory dla grup
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    group_names = ['Grupa A', 'Grupa B', 'Grupa C']
    
    for i in range(3):
        mask = df['Grupa'] == i
        plt.scatter(df[mask]['Bezrobocie'], df[mask]['Inflacja'], 
                   c=colors[i], label=group_names[i], alpha=0.7, s=100)
    
    # Dodanie centrów grup
    centers_original = kmeans.cluster_centers_
    # Przekształcenie z powrotem do oryginalnej skali
    scaler = StandardScaler()
    scaler.fit(df[['Bezrobocie']])
    centers_bezrobocie = scaler.inverse_transform(centers_original[:, 0].reshape(-1, 1)).flatten()
    
    scaler.fit(df[['Inflacja']])
    centers_inflacja = scaler.inverse_transform(centers_original[:, 1].reshape(-1, 1)).flatten()
    
    plt.scatter(centers_bezrobocie, centers_inflacja, 
               c='black', marker='X', s=300, linewidths=2, label='Centra grup')
    
    plt.xlabel('Bezrobocie (tys. osób)', fontweight='bold', fontsize=14)
    plt.ylabel('Inflacja (%)', fontweight='bold', fontsize=14)
    plt.title('Grupowanie krajów według podobnych charakterystyk ekonomicznych', 
              fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_grupy_krajow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analiza grup
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Komputer automatycznie pogrupował kraje o podobnych charakterystykach
• Różne kolory = różne grupy krajów
• Czarne X = "średni" kraj w każdej grupie
• Kraje w tej samej grupie mają podobne poziomy bezrobocia i inflacji

💡 JAK TO INTERPRETOWAĆ:
""")
    
    for i in range(3):
        grupa_data = df[df['Grupa'] == i]
        avg_bez = grupa_data['Bezrobocie'].mean()
        avg_inf = grupa_data['Inflacja'].mean()
        count = len(grupa_data)
        print(f"• {group_names[i]}: {count} obserwacji, średnie bezrobocie: {avg_bez:.0f} tys., średnia inflacja: {avg_inf:.1f}%")

def wykres_6_histogram_2d(df):
    """
    WYKRES 6: Mapa gęstości - gdzie skupiają się kraje
    """
    print("📊 Tworzenie wykresu 6: Mapa gęstości...")
    
    plt.figure(figsize=(12, 8))
    
    # Histogram 2D
    h = plt.hist2d(df['Bezrobocie'], df['Inflacja'], bins=20, cmap='YlOrRd')
    plt.colorbar(h[3], label='Liczba krajów w tym obszarze')
    
    plt.xlabel('Bezrobocie (tys. osób)', fontweight='bold', fontsize=14)
    plt.ylabel('Inflacja (%)', fontweight='bold', fontsize=14)
    plt.title('Gdzie skupiają się kraje pod względem bezrobocia i inflacji', 
              fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('6_histogram_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• "Mapa ciepła" pokazująca gdzie skupiają się kraje
• Żółte/pomarańczowe obszary = mało krajów
• Czerwone/ciemne obszary = dużo krajów
• Im ciemniejszy kolor, tym więcej krajów ma podobne wartości

💡 JAK TO INTERPRETOWAĆ:
• Gdzie są najczęstsze kombinacje bezrobocia i inflacji?
• Czy kraje tworzą wyraźne skupiska?
• Które kombinacje są rzadkie?
""")

def wykres_7_boxplot_porownawczy(df):
    """
    WYKRES 7: Porównanie rozkładów wartości
    """
    print("📊 Tworzenie wykresu 7: Porównanie rozkładów...")
    
    # Przygotowanie danych do boxplotu
    df_melt = pd.melt(df, value_vars=['Inflacja_norm', 'Bezrobocie_norm'], 
                      var_name='Wskaźnik', value_name='Wartość_znormalizowana')
    
    plt.figure(figsize=(10, 8))
    
    # Boxplot
    bp = plt.boxplot([df['Inflacja_norm'], df['Bezrobocie_norm']], 
                     labels=['Inflacja\n(znormalizowana)', 'Bezrobocie\n(znormalizowane)'],
                     patch_artist=True)
    
    # Kolorowanie boxplotów
    colors = ['#e74c3c', '#3498db']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Wartość znormalizowana', fontweight='bold', fontsize=14)
    plt.title('Porównanie rozrzutu danych: inflacja vs bezrobocie', 
              fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Dodanie opisu elementów boxplotu
    plt.text(0.02, 0.98, 
             'Elementy boxplotu:\n• Linia środkowa = mediana\n• Pudełko = 50% danych\n• Wąsy = zakres danych\n• Kropki = wartości odstające', 
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('7_boxplot_porownawczy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Porównanie "rozrzutu" danych inflacji i bezrobocia
• Pudełka pokazują gdzie znajduje się 50% obserwacji
• Linia w środku pudełka = wartość środkowa (mediana)
• Wąsy pokazują zakres typowych wartości
• Kropki = wartości bardzo wysokie lub bardzo niskie

💡 JAK TO INTERPRETOWAĆ:
• Który wskaźnik ma większą zmienność między krajami?
• Czy są wartości ekstremalnie wysokie lub niskie?
• Jak różnią się rozkłady obu wskaźników?
""")

def wykres_8_regresja_z_przedzialem(df):
    """
    WYKRES 8: Model przewidywania z przedziałem ufności
    """
    print("📊 Tworzenie wykresu 8: Model przewidywania...")
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(df['Bezrobocie'], df['Inflacja'], alpha=0.6, s=80, color='#3498db', label='Obserwacje')
    
    # Model regresji
    X = df[['Bezrobocie']].values
    y = df['Inflacja'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Przewidywania
    X_plot = np.linspace(df['Bezrobocie'].min(), df['Bezrobocie'].max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    # Linia regresji
    plt.plot(X_plot, y_plot, color='red', linewidth=3, label='Linia trendu')
    
    # Obliczenie R²
    y_pred = model.predict(X)
    r2 = np.corrcoef(y, y_pred)[0, 1]**2
    
    plt.xlabel('Bezrobocie (tys. osób)', fontweight='bold', fontsize=14)
    plt.ylabel('Inflacja (%)', fontweight='bold', fontsize=14)
    plt.title('Model przewidywania inflacji na podstawie bezrobocia', 
              fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Dodanie informacji o modelu
    equation = f'Inflacja = {model.intercept_:.2f} + {model.coef_[0]:.4f} × Bezrobocie'
    plt.text(0.05, 0.95, f'Równanie: {equation}\nDokładność modelu (R²): {r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('8_regresja_z_przedzialem.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"""
🔍 CO POKAZUJE TEN WYKRES:
• Czerwona linia = najlepsze "dopasowanie" do danych
• Model matematyczny pozwalający przewidzieć inflację na podstawie bezrobocia
• R² = {r2:.3f} oznacza, że model wyjaśnia {r2*100:.1f}% zmienności danych

💡 JAK TO INTERPRETOWAĆ:
• Czy model dobrze przewiduje inflację?
• R² bliskie 1.0 = bardzo dobry model
• R² bliskie 0.0 = słaby model
• Równanie pozwala obliczyć przewidywaną inflację dla danego poziomu bezrobocia
""")

def wykres_9_macierz_korelacji(df):
    """
    WYKRES 9: Macierz wszystkich korelacji
    """
    print("📊 Tworzenie wykresu 9: Macierz korelacji...")
    
    # Przygotowanie danych z opóźnieniami
    df_corr = df.copy()
    df_corr['Inflacja_poprzedni_miesiac'] = df_corr.groupby('Kraj')['Inflacja'].shift(1)
    df_corr['Bezrobocie_poprzedni_miesiac'] = df_corr.groupby('Kraj')['Bezrobocie'].shift(1)
    
    # Wybór kolumn do analizy
    cols_to_corr = ['Inflacja', 'Bezrobocie', 'Inflacja_poprzedni_miesiac', 'Bezrobocie_poprzedni_miesiac']
    correlation_matrix = df_corr[cols_to_corr].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Maska dla górnego trójkąta
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Heatmapa
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    
    plt.title('Macierz korelacji - jak różne wskaźniki są ze sobą powiązane', 
              fontweight='bold', fontsize=16)
    
    # Zmiana etykiet na bardziej zrozumiałe
    new_labels = ['Inflacja\n(obecna)', 'Bezrobocie\n(obecne)', 
                  'Inflacja\n(poprzedni miesiąc)', 'Bezrobocie\n(poprzedni miesiąc)']
    plt.gca().set_xticklabels(new_labels, rotation=45, ha='right')
    plt.gca().set_yticklabels(new_labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig('9_macierz_korelacji.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
🔍 CO POKAZUJE TEN WYKRES:
• Tabela pokazująca siłę związku między wszystkimi parami wskaźników
• Czerwony kolor = pozytywny związek (jedno rośnie, drugie też)
• Niebieski kolor = negatywny związek (jedno rośnie, drugie maleje)
• Liczby = siła związku (od -1.0 do +1.0)

💡 JAK TO INTERPRETOWAĆ:
• Wartości bliskie +1.0 = bardzo silny pozytywny związek
• Wartości bliskie -1.0 = bardzo silny negatywny związek
• Wartości bliskie 0.0 = brak związku
• Czy przeszłe wartości wpływają na obecne?
""")

def stworz_podsumowanie(df):
    """
    Tworzy podsumowanie wszystkich analiz
    """
    print("📊 Tworzenie podsumowania analiz...")
    
    # Obliczenia podstawowe
    korelacja, p_value = stats.pearsonr(df['Bezrobocie'], df['Inflacja'])
    
    plt.figure(figsize=(14, 10))
    
    # Tytuł
    plt.suptitle('PODSUMOWANIE ANALIZY: Bezrobocie vs Inflacja', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Usunięcie osi
    plt.axis('off')
    
    # Tekst podsumowania
    summary_text = f"""
📊 PODSTAWOWE STATYSTYKI:
• Przeanalizowano {len(df):,} obserwacji z {df['Kraj'].nunique()} krajów
• Średnie bezrobocie: {df['Bezrobocie'].mean():.0f} tys. osób (od {df['Bezrobocie'].min():.0f} do {df['Bezrobocie'].max():.0f})
• Średnia inflacja: {df['Inflacja'].mean():.1f}% (od {df['Inflacja'].min():.1f}% do {df['Inflacja'].max():.1f}%)

🔗 GŁÓWNE WYNIKI:
• Korelacja między bezrobociem a inflacją: {korelacja:+.3f}
• Statystyczna istotność: {'TAK' if p_value < 0.05 else 'NIE'} (p-value: {p_value:.3f})

💡 CO TO OZNACZA:
"""
    
    if abs(korelacja) < 0.1:
        interpretation = "• Bardzo słaby związek - bezrobocie i inflacja nie są ze sobą powiązane"
    elif abs(korelacja) < 0.3:
        interpretation = "• Słaby związek - niewielka zależność między bezrobociem a inflacją"
    elif abs(korelacja) < 0.5:
        interpretation = "• Umiarkowany związek - zauważalna zależność między wskaźnikami"
    elif abs(korelacja) < 0.7:
        interpretation = "• Silny związek - wyraźna zależność między bezrobociem a inflacją"
    else:
        interpretation = "• Bardzo silny związek - bezrobocie i inflacja są mocno powiązane"
    
    if korelacja > 0:
        direction = "• Związek pozytywny: wyższe bezrobocie → wyższa inflacja"
    else:
        direction = "• Związek negatywny: wyższe bezrobocie → niższa inflacja"
    
    summary_text += interpretation + "\n" + direction
    
    # Dodanie kontekstu ekonomicznego
    if korelacja < -0.2:
        context = """
🏛️ KONTEKST EKONOMICZNY:
• Wynik zgodny z klasyczną krzywą Phillipsa
• Gdy bezrobocie rośnie, inflacja zwykle maleje
• Może wskazywać na typowe cykle gospodarcze"""
    elif korelacja > 0.2:
        context = """
🏛️ KONTEKST EKONOMICZNY:
• Wynik może wskazywać na stagflację
• Jednoczesny wzrost bezrobocia i inflacji
• Możliwe wpływy zewnętrzne (np. ceny surowców)"""
    else:
        context = """
🏛️ KONTEKST EKONOMICZNY:
• Brak prostej zależności Phillipsa
• Inne czynniki mogą dominować
• Różne mechanizmy w różnych krajach"""
    
    summary_text += context
    
    plt.text(0.05, 0.85, summary_text, fontsize=14, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    # Lista wygenerowanych wykresów
    files_text = """
📁 WYGENEROWANE WYKRESY:

1️⃣ scatter_podstawowy.png - Podstawowy związek bezrobocie-inflacja
2️⃣ histogram_porownawczy.png - Jak często występują różne wartości  
3️⃣ trendy_czasowe.png - Zmiany w czasie
4️⃣ korelacja_po_krajach.png - Różnice między krajami
5️⃣ grupy_krajow.png - Podobne kraje pogrupowane
6️⃣ histogram_2d.png - Mapa gęstości obserwacji
7️⃣ boxplot_porównawczy.png - Porównanie rozrzutu danych
8️⃣ regresja_z_przedzialem.png - Model przewidywania
9️⃣ macierz_korelacji.png - Wszystkie korelacje naraz
"""
    
    plt.text(0.05, 0.35, files_text, fontsize=12, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('0_PODSUMOWANIE_ANALIZY.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("""
✅ ANALIZA ZAKOŃCZONA!

📁 Sprawdź wszystkie wygenerowane pliki PNG w bieżącym folderze.
Każdy wykres ma własny plik z numerem i opisową nazwą.
""")

def main():
    """
    Główna funkcja generująca wszystkie wykresy
    """
    print("🎯 GENERATOR WYKRESÓW Z OPISAMI")
    print("="*50)
    print("Każdy wykres będzie zapisany jako osobny plik PNG z prostym opisem")
    print()
    
    try:
        # Przygotowanie danych
        df = przygotuj_dane()
        
        if len(df) == 0:
            print("❌ Brak danych do analizy!")
            return
        
        print(f"\n🚀 Rozpoczynam tworzenie {9} wykresów...\n")
        
        # Generowanie wszystkich wykresów
        stworz_podsumowanie(df)
        wykres_1_scatter_podstawowy(df)
        wykres_2_histogram_porownawczy(df)
        wykres_3_trendy_czasowe(df)
        wykres_4_korelacja_po_krajach(df)
        wykres_5_grupy_krajow(df)
        wykres_6_histogram_2d(df)
        wykres_7_boxplot_porownawczy(df)
        wykres_8_regresja_z_przedzialem(df)
        wykres_9_macierz_korelacji(df)
        
        print("\n" + "="*50)
        print("✅ WSZYSTKIE WYKRESY ZOSTAŁY UTWORZONE!")
        print("📁 Sprawdź pliki PNG w bieżącym folderze")
        print("📊 Każdy wykres ma własny opis i interpretację")
        
    except FileNotFoundError as e:
        print(f"❌ BŁĄD: Nie znaleziono pliku: {e}")
        print("Upewnij się, że pliki inflation_small.csv i unemployment_esmall.csv są w folderze")
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 