#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza korelacji między bezrobociem a inflacją
Dane źródłowe: inflation_small.csv i unemployment_esmall.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja wyświetlania
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

def wczytaj_dane_inflacja(plik_csv):
    """
    Wczytuje i przetwarza dane o inflacji
    """
    print("Wczytuję dane o inflacji...")
    df_inflacja = pd.read_csv(plik_csv)
    
    # Sprawdź strukturę danych
    print(f"Kształt danych inflacji: {df_inflacja.shape}")
    print(f"Kolumny: {list(df_inflacja.columns)}")
    
    # Konwersja TIME_PERIOD na format daty
    df_inflacja['TIME_PERIOD'] = pd.to_datetime(df_inflacja['TIME_PERIOD'])
    
    # Filtrowanie danych - tylko te z wartościami inflacji
    df_inflacja = df_inflacja.dropna(subset=['OBS_VALUE'])
    
    # Selekcja tylko potrzebnych kolumn
    df_inflacja_clean = df_inflacja[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_inflacja_clean.columns = ['Kraj', 'Data', 'Inflacja']
    
    print(f"Dane po czyszczeniu: {df_inflacja_clean.shape}")
    print(f"Kraje w danych inflacji: {df_inflacja_clean['Kraj'].nunique()}")
    
    return df_inflacja_clean

def wczytaj_dane_bezrobocie(plik_csv):
    """
    Wczytuje i przetwarza dane o bezrobociu
    """
    print("\nWczytuję dane o bezrobociu...")
    df_bezrobocie = pd.read_csv(plik_csv)
    
    # Sprawdź strukturę danych
    print(f"Kształt danych bezrobocia: {df_bezrobocie.shape}")
    print(f"Kolumny: {list(df_bezrobocie.columns)}")
    
    # Konwersja TIME_PERIOD na format daty
    df_bezrobocie['TIME_PERIOD'] = pd.to_datetime(df_bezrobocie['TIME_PERIOD'])
    
    # Filtrowanie danych - tylko te z wartościami bezrobocia
    df_bezrobocie = df_bezrobocie.dropna(subset=['OBS_VALUE'])
    
    # Agregacja danych po kraju i dacie (suma dla wszystkich grup demograficznych)
    df_bezrobocie_agg = df_bezrobocie.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].sum().reset_index()
    df_bezrobocie_agg.columns = ['Kraj', 'Data', 'Bezrobocie']
    
    print(f"Dane po agregacji: {df_bezrobocie_agg.shape}")
    print(f"Kraje w danych bezrobocia: {df_bezrobocie_agg['Kraj'].nunique()}")
    
    return df_bezrobocie_agg

def znajdz_wspolne_dane(df_inflacja, df_bezrobocie):
    """
    Łączy dane o inflacji i bezrobociu dla wspólnych krajów i okresów
    """
    print("\nŁączenie danych...")
    
    # Wyświetl unikalne kraje w każdym zbiorze danych
    kraje_inflacja = set(df_inflacja['Kraj'].unique())
    kraje_bezrobocie = set(df_bezrobocie['Kraj'].unique())
    
    print(f"Kraje w danych inflacji: {len(kraje_inflacja)}")
    print(f"Kraje w danych bezrobocia: {len(kraje_bezrobocie)}")
    
    # Wspólne kraje
    wspolne_kraje = kraje_inflacja.intersection(kraje_bezrobocie)
    print(f"Wspólne kraje: {len(wspolne_kraje)}")
    print(f"Lista wspólnych krajów: {sorted(list(wspolne_kraje))}")
    
    # Łączenie danych
    df_polaczone = pd.merge(
        df_inflacja, 
        df_bezrobocie, 
        on=['Kraj', 'Data'], 
        how='inner'
    )
    
    print(f"Kształt połączonych danych: {df_polaczone.shape}")
    
    return df_polaczone

def analiza_korelacji(df):
    """
    Przeprowadza analizę korelacji między bezrobociem a inflacją
    """
    print("\n" + "="*50)
    print("ANALIZA KORELACJI BEZROBOCIE - INFLACJA")
    print("="*50)
    
    # Podstawowe statystyki
    print("\n1. PODSTAWOWE STATYSTYKI:")
    print(df[['Inflacja', 'Bezrobocie']].describe())
    
    # Korelacja Pearsona
    korelacja_pearson, p_value_pearson = stats.pearsonr(df['Bezrobocie'], df['Inflacja'])
    print(f"\n2. KORELACJA PEARSONA:")
    print(f"   Współczynnik korelacji: {korelacja_pearson:.4f}")
    print(f"   Wartość p: {p_value_pearson:.4f}")
    
    # Korelacja Spearmana (nieparametryczna)
    korelacja_spearman, p_value_spearman = stats.spearmanr(df['Bezrobocie'], df['Inflacja'])
    print(f"\n3. KORELACJA SPEARMANA:")
    print(f"   Współczynnik korelacji: {korelacja_spearman:.4f}")
    print(f"   Wartość p: {p_value_spearman:.4f}")
    
    # Interpretacja korelacji
    print(f"\n4. INTERPRETACJA:")
    if abs(korelacja_pearson) < 0.1:
        sila = "bardzo słaba"
    elif abs(korelacja_pearson) < 0.3:
        sila = "słaba"
    elif abs(korelacja_pearson) < 0.5:
        sila = "umiarkowana"
    elif abs(korelacja_pearson) < 0.7:
        sila = "silna"
    else:
        sila = "bardzo silna"
    
    kierunek = "pozytywna" if korelacja_pearson > 0 else "negatywna"
    print(f"   Korelacja jest {sila} i {kierunek}")
    
    if p_value_pearson < 0.05:
        print(f"   Korelacja jest statystycznie istotna (p < 0.05)")
    else:
        print(f"   Korelacja nie jest statystycznie istotna (p >= 0.05)")
    
    return korelacja_pearson, p_value_pearson

def wizualizacja_danych(df):
    """
    Tworzy wizualizacje danych
    """
    print("\n5. TWORZENIE WIZUALIZACJI...")
    
    # Utworzenie siatki wykresów
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analiza korelacji: Bezrobocie vs Inflacja', fontsize=16, fontweight='bold')
    
    # Wykres 1: Scatter plot z linią trendu
    ax1 = axes[0, 0]
    ax1.scatter(df['Bezrobocie'], df['Inflacja'], alpha=0.6, s=60)
    
    # Dodanie linii trendu
    z = np.polyfit(df['Bezrobocie'], df['Inflacja'], 1)
    p = np.poly1d(z)
    ax1.plot(df['Bezrobocie'], p(df['Bezrobocie']), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Bezrobocie (tys. osób)')
    ax1.set_ylabel('Inflacja (%)')
    ax1.set_title('Scatter plot: Bezrobocie vs Inflacja')
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Histogram inflacji
    ax2 = axes[0, 1]
    ax2.hist(df['Inflacja'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Inflacja (%)')
    ax2.set_ylabel('Częstość')
    ax2.set_title('Rozkład inflacji')
    ax2.grid(True, alpha=0.3)
    
    # Wykres 3: Histogram bezrobocia
    ax3 = axes[1, 0]
    ax3.hist(df['Bezrobocie'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('Bezrobocie (tys. osób)')
    ax3.set_ylabel('Częstość')
    ax3.set_title('Rozkład bezrobocia')
    ax3.grid(True, alpha=0.3)
    
    # Wykres 4: Heatmapa korelacji
    ax4 = axes[1, 1]
    correlation_matrix = df[['Inflacja', 'Bezrobocie']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax4, fmt='.3f', cbar_kws={'shrink': 0.8})
    ax4.set_title('Macierz korelacji')
    
    plt.tight_layout()
    plt.savefig('analiza_korelacji_bezrobocie_inflacja.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Wykresy zapisane jako 'analiza_korelacji_bezrobocie_inflacja.png'")

def analiza_po_krajach(df):
    """
    Analizuje korelację dla poszczególnych krajów
    """
    print("\n6. ANALIZA PO KRAJACH:")
    
    # Oblicz korelację dla każdego kraju (jeśli ma więcej niż 3 obserwacje)
    korelacje_kraje = []
    
    for kraj in df['Kraj'].unique():
        dane_kraj = df[df['Kraj'] == kraj]
        
        if len(dane_kraj) >= 3:  # Minimum 3 obserwacje dla korelacji
            try:
                korelacja, p_value = stats.pearsonr(dane_kraj['Bezrobocie'], dane_kraj['Inflacja'])
                korelacje_kraje.append({
                    'Kraj': kraj,
                    'Korelacja': korelacja,
                    'P-value': p_value,
                    'Liczba_obserwacji': len(dane_kraj)
                })
            except:
                pass
    
    if korelacje_kraje:
        df_korelacje = pd.DataFrame(korelacje_kraje)
        df_korelacje = df_korelacje.sort_values('Korelacja', ascending=False)
        
        print(f"\nKorelacje dla poszczególnych krajów:")
        print(df_korelacje.to_string(index=False))
        
        # Wykres korelacji po krajach
        plt.figure(figsize=(12, 8))
        plt.barh(df_korelacje['Kraj'], df_korelacje['Korelacja'])
        plt.xlabel('Współczynnik korelacji')
        plt.title('Korelacja bezrobocie-inflacja po krajach')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('korelacja_po_krajach.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   Wykres zapisany jako 'korelacja_po_krajach.png'")
    else:
        print("   Brak wystarczających danych do analizy po krajach")

def analiza_czasowa(df):
    """
    Analizuje trendy czasowe
    """
    print("\n7. ANALIZA CZASOWA:")
    
    # Grupowanie po miesiącach
    df_czasowy = df.groupby('Data').agg({
        'Inflacja': 'mean',
        'Bezrobocie': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Podwójna oś Y
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    color = 'tab:red'
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Inflacja (%)', color=color)
    ax1.plot(df_czasowy['Data'], df_czasowy['Inflacja'], color=color, linewidth=2, label='Inflacja')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Bezrobocie (tys. osób)', color=color)
    ax2.plot(df_czasowy['Data'], df_czasowy['Bezrobocie'], color=color, linewidth=2, label='Bezrobocie')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Trendy czasowe: Inflacja i Bezrobocie')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trendy_czasowe.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Wykres zapisany jako 'trendy_czasowe.png'")

def main():
    """
    Główna funkcja programu
    """
    print("ANALIZA KORELACJI MIĘDZY BEZROBOCIEM A INFLACJĄ")
    print("=" * 60)
    
    try:
        # Wczytanie danych
        df_inflacja = wczytaj_dane_inflacja('inflation_small.csv')
        df_bezrobocie = wczytaj_dane_bezrobocie('unemployment_esmall.csv')
        
        # Połączenie danych
        df_polaczone = znajdz_wspolne_dane(df_inflacja, df_bezrobocie)
        
        if len(df_polaczone) == 0:
            print("BŁĄD: Brak wspólnych danych do analizy!")
            return
        
        # Analiza korelacji
        korelacja, p_value = analiza_korelacji(df_polaczone)
        
        # Wizualizacja
        wizualizacja_danych(df_polaczone)
        
        # Analiza po krajach
        analiza_po_krajach(df_polaczone)
        
        # Analiza czasowa
        analiza_czasowa(df_polaczone)
        
        # Podsumowanie
        print("\n" + "="*60)
        print("PODSUMOWANIE ANALIZY")
        print("="*60)
        print(f"Liczba obserwacji: {len(df_polaczone)}")
        print(f"Liczba krajów: {df_polaczone['Kraj'].nunique()}")
        print(f"Współczynnik korelacji: {korelacja:.4f}")
        print(f"Wartość p: {p_value:.4f}")
        
        # Interpretacja ekonomiczna
        print(f"\nINTERPRETACJA EKONOMICZNA:")
        if korelacja < 0:
            print("Wynik wskazuje na negatywną korelację między bezrobociem a inflacją,")
            print("co może sugerować odwrotną zależność zgodną z krzywą Phillipsa.")
        else:
            print("Wynik wskazuje na pozytywną korelację między bezrobociem a inflacją,")
            print("co może być związane z czynnikami strukturalnymi lub stagflacją.")
        
        print(f"\nPliki wygenerowane:")
        print("- analiza_korelacji_bezrobocie_inflacja.png")
        print("- korelacja_po_krajach.png")
        print("- trendy_czasowe.png")
        
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku: {e}")
        print("Upewnij się, że pliki 'inflation_small.csv' i 'unemployment_esmall.csv' są w tym samym folderze.")
    except Exception as e:
        print(f"BŁĄD: {e}")

if __name__ == "__main__":
    main() 