# Klasyfikacja gatunku książek na podstawie ich streszczenia
Rozpoznawanie gatunku książki na podstawie streszczenia wykonane w ramach przedmiotu Sztuczna Inteligencja i Inżynieria Wiedzy
-

Dane pobrano z [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html).

Plik z danymi ma format txt i zawiera streszczenia fabuły 16 559 książek. Dane dotyczące poszczególnych książek rozdzielane są znakiem nowej linii.

Każda linia zawiera następujące kolumny rozdzielone znakiem tabulacji:
- identyfikator artykułu Wikipedii, z którego pochodzi streszczenie fabuły,
- identyfikator z systemu Freebase,
- tytuł książki,
- autora książki,
- datę publikacji,
- listę gatunków przypisanych do danej książki w formie obiektu JSON, gdzie kluczem jest identyfikator gatunku z systemu Freebase, a wartością nazwa gatunku,
- streszczenie fabuły.

Aby zaklasyfikować książkę do konkretnego gatunku na podstawie jej streszczenia potrzebne są w zasadzie dwie informacje: streszczenie fabuły książki oraz lista gatunków, do których algorytm może zaklasyfikować daną książkę. Aby uzyskać tę listę gatunków należy z danych wejściowych przeczytać również gatunki dla każdej książki i odpowiednio je przetworzyć. Streszczenia pochodzą z Wikipedii, a więc każda książka ma przypisany identyfikator artykułu Wikipedii, z którego pochodzi streszczenie jej fabuły. Ten identyfikator może być np. identyfikatorem słownika przechowującego dane. Oprócz tego, choć nie jest to w żaden sposób istotne dla algorytmu, przechowywany będzie również tytuł książki, aby analizowane dane były zrozumiałe również dla człowieka obcującego z algorytmem. Wszystkie pozostałe dane nie będą w żaden sposób analizowane, a zatem mogą zostać odrzucone.

W związku z tym dane reprezentowane są jako słownik, w którym:
- kluczem jest identyfikator artykułu Wikipedii,
- wartością jest słownik:
  - o kluczach ‘title’, ‘genre’, ‘summary’,
  - w którym wartością jest dana typu string.

W pliku źródłowym specjalne znaki łacińskie są zapisane w postaci kodu UTF-16 np. \u00e0 oznacza à. Ponieważ CountVectorizer domyślnie wykorzystuje encoding UTF-8 dokonano konwersji zawartości pliku na UTF-8. Przy konwersji na UTF-8 znaleziono 9 rekordów, których nie dało się przekonwertować, ponieważ zawierały one znak \ nie poprzedzający kodu UTF-16. Te rekordy odrzucono.

W pierwszej kolejności odrzucono rekordy zawierające znak \ nie poprzedzający kodu UTF-16. Następnie odrzucono rekordy, w których identyfikator artykułu Wikipedii, tytuł, lista gatunków lub streszczenie książki były puste, a także rekordy zawierające podejrzane wartości – linki do innych stron, znaczniki i znaki specjalne html, komentarze xml itp. W ten sposób otrzymano 11410 rekordów.

Z listy gatunków dla każdej książki usunięto zbyt szerokie i zbyt wąskie gatunki. Następnie odrzucono wszystkie rekordy, dla których w wyniku poprzedniej operacji nie pozostał żaden gatunek. Otrzymano listę gatunków zaprezentowaną na Screenie 2. Usunięto również wszystkie rekordy, dla których przypisano dwa lub więcej gatunki z tej listy – zdecydowano się usuwać te rekordy zamiast wybierać losowo jeden z gatunków żeby nie zacierać kryteriów rozróżniających klasy.
  
