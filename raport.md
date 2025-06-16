
# Dokumentacja: Analiza zależności między inflacją a bezrobociem

## Cel projektu

Celem projektu była analiza zależności pomiędzy inflacją a bezrobociem w wybranych krajach na przestrzeni lat 1997–2025. Przeprowadzono zaawansowane czyszczenie danych, testy statystyczne, analizę regresji, klasteryzację oraz przygotowano wizualizacje i raport końcowy.


## Dane i metodyka

- **Źródła danych:**  
  - Inflacja: `hicp_full.csv`  
  - Bezrobocie: `full_un.csv`
- **Zakres czasowy:** 1997-01 do 2025-05
- **Liczba krajów po połączeniu:** 32
- **Liczba obserwacji po czyszczeniu:** 9,771

### Etapy analizy

1. **Czyszczenie danych:**  
   Usunięto wartości odstające (IQR, >3σ), null oraz nieskończone.
2. **Łączenie danych:**  
   Połączono dane po kraju i dacie, dodano zmienne opóźnione i znormalizowano wartości.
3. **Testy statystyczne:**  
   Przeprowadzono testy normalności, homoskedastyczności oraz obliczono korelacje (Pearson, Spearman, Kendall).
4. **Regresja liniowa:**  
   Zbudowano model regresji liniowej dla inflacji względem bezrobocia.
5. **Klasteryzacja (K-means):**  
   Wydzielono 3 klastry na podstawie znormalizowanych danych.
6. **Wizualizacje:**  
   Przygotowano wykresy ilustrujące zależności i trendy.
7. **Raport końcowy:**  
   Podsumowano wyniki i sformułowano wnioski ekonomiczne.

---

## Wyniki i interpretacja

### 1. Wykresy (`Figure_1.png`)

#### a) **Inflacja vs Bezrobocie (z klastrami)**
- Wykres rozrzutu pokazuje brak wyraźnej liniowej zależności między inflacją a bezrobociem.
- Kolory odpowiadają klastrom wyodrębnionym metodą K-means – nie widać jednoznacznych, odseparowanych grup.
- Linia regresji jest niemal pozioma, co potwierdza bardzo słabą korelację.

#### b) **Trendy inflacji i bezrobocia w czasie**
- Inflacja (czerwona linia) i bezrobocie (niebieska linia, prawa oś) wykazują niezależne trendy.
- Widać wyraźne piki inflacji w latach 2021–2022, bez wyraźnej reakcji bezrobocia.
- Brak widocznej odwrotnej zależności sugerowanej przez klasyczną krzywą Phillipsa.

#### c) **Korelacja inflacja–bezrobocie w krajach**
- Wartości współczynnika korelacji Pearsona dla poszczególnych krajów są bliskie zera lub lekko ujemne.
- W większości krajów nie obserwuje się silnej korelacji (ani dodatniej, ani ujemnej).

#### d) **Różnica w bezrobociu między płciami**
- W niektórych krajach (np. Turkiye, Japonia, UK) bezrobocie wśród mężczyzn jest wyraźnie wyższe niż wśród kobiet.
- W innych krajach różnice są minimalne lub nieznacznie na korzyść kobiet.

---

### 2. Wyniki testów statystycznych (output konsoli)

- **Testy normalności:**  
  Zarówno inflacja, jak i bezrobocie nie mają rozkładu normalnego (p-value ≈ 0).
- **Test homoskedastyczności:**  
  Wariancje nie są równe (p-value = 0.0032).
- **Korelacje:**
  - Pearson: r = -0.0047 (p = 0.6427) – bardzo słaba, nieistotna statystycznie korelacja.
  - Spearman: r = -0.0491 (p = 0.0000) – bardzo słaba, istotna statystycznie, ale praktycznie nieistotna ekonomicznie.
  - Kendall: r = -0.0327 (p = 0.0000) – jw.
- **Regresja liniowa:**  
  - R² = 0.0000 – model nie wyjaśnia wariancji inflacji na podstawie bezrobocia.
  - Współczynnik regresji ≈ 0 – brak liniowej zależności.

---

## Wnioski

1. **Brak prostej zależności:**  
   Analiza nie wykazała istotnej liniowej zależności między inflacją a bezrobociem w badanym okresie i krajach.
2. **Krzywa Phillipsa nie znajduje potwierdzenia:**  
   Klasyczna odwrotna zależność (niższe bezrobocie = wyższa inflacja) nie występuje w danych zagregowanych dla Europy w latach 1997–2025.
3. **Różnice krajowe:**  
   W poszczególnych krajach korelacje są zbliżone do zera, co sugeruje, że inne czynniki (np. polityka gospodarcza, struktura rynku pracy) mają większe znaczenie.
4. **Różnice płci:**  
   W niektórych krajach występują istotne różnice w bezrobociu między mężczyznami a kobietami, co może być przedmiotem osobnej analizy.
5. **Znaczenie innych czynników:**  
   Wyniki sugerują, że inflacja i bezrobocie są kształtowane przez złożone, wieloczynnikowe mechanizmy, a prosta analiza dwuwymiarowa nie oddaje pełnego obrazu.

---

## Pliki wyjściowe

- `wizualizacja_skoncentrowana.png` – główne wykresy analityczne
- `dane_oczyszczone.csv` – oczyszczone dane do dalszych analiz

---

## Rekomendacje

- Rozważyć analizę panelową z uwzględnieniem dodatkowych zmiennych (np. PKB, polityka monetarna).
- Przeprowadzić osobne analizy dla poszczególnych krajów lub okresów kryzysowych.
- Zbadać nieliniowe zależności lub opóźnienia czasowe (lagged effects).

---

**Autor:**  
*Imię i nazwisko / Zespół*  
*Data: [uzupełnij]*

---

**Załącznik:**  
![Wizualizacja wyników](Figure_1.png)

---

Jeśli chcesz, mogę rozwinąć dowolną sekcję lub dodać szczegółowe opisy do poszczególnych wykresów!