# 📊 Analiza Inflacji vs Bezrobocie - Krzywa Phillipsa

## 🎯 Opis projektu
Projekt analizuje zależność między inflacją a bezrobociem w krajach europejskich, weryfikując aktualność krzywej Phillipsa w współczesnej ekonomii. Wykorzystuje zaawansowane metody statystyczne, analizę klastryzacji oraz wizualizację danych.

## 📁 Pliki projektu
- **`analiza_inflacja_bezrobocie.py`** - główny kod analizy
- **Prezentacja** - wyniki i wnioski w formie graficznej
- **`dane_oczyszczone.csv`** - przetworzone dane (generowane automatycznie)

## 🔍 Główne wyniki
- **Brak klasycznej krzywej Phillipsa** - korelacja Pearsona ≈ 0
- **3 różne reżimy ekonomiczne** zamiast jednej prostej zależności
- **Klastryzacja krajów** według podobieństwa ekonomicznego
- **Krzywa Phillipsa to relikt XX wieku** - współczesne gospodarki wymagają bardziej złożonych modeli

## 📚 Literatura i źródła

### Opracowania naukowe:
1. [Krzywa Phillipsa w Polsce - analiza empiryczna](https://bazhum.muzhp.pl/media/texts/studia-i-prace-wydziau-nauk-ekonomicznych-i-zarzadzania/2017-tom-47-numer-3/studia_i_prace_wydzialu_nauk_ekonomicznych_i_zarzadzania-r2017-t47-n3-s171-182.pdf)
2. [Makroekonomia - Krzywa Phillipsa (OpenStax)](https://openstax.org/books/makroekonomia-podstawy/pages/7-3-krzywa-phillipsa)

## 🛠️ Technologie
- Python, pandas, numpy
- scikit-learn (K-means, PCA)
- matplotlib, seaborn, plotly
- scipy (testy statystyczne)

## 📊 Metodologia
1. Czyszczenie i preprocessing danych
2. Testy normalności i korelacji (Pearson, Spearman, Kendall)
3. Klastryzacja K-means
4. Analiza regresji liniowej
5. Wizualizacja wyników 