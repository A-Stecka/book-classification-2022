import io
import json
import csv
import pandas
import re
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# zeby pakiety z nltk dzialaly trzeba wpisac komendy:
# python -m nltk.downloader stopwords
# python -m nltk.downloader wordnet
# python -m nltk.downloader omw-1.4


# czytanie z utworzonego w metodzie process_and_import_into_csv pliku csv
def read_from_csv() -> dict[int, dict[str, str]]:
    # 0 - klucz, 1 - tytul, 2 - gatunek, 3 - streszczenie
    data_frame = pandas.read_csv('booksummaries/processed_data.csv', header=None)
    dataset = {}
    for line_index in data_frame.index:
        dataset[int(data_frame.loc[line_index, 0])] = {'title': str(data_frame.loc[line_index, 1]),
                                                       'genre': str(data_frame.loc[line_index, 2]),
                                                       'summary': str(data_frame.loc[line_index, 3])}
    return dataset


# preprocessing i zapis danych do pliku csv
def process_and_import_into_csv():
    dataset = data_cleanup()
    dataset_as_array = []
    for key in dataset:
        dataset_as_array.append([key, dataset[key]['title'], dataset[key]['genre'], dataset[key]['summary']])
    with open('booksummaries/processed_data.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(dataset_as_array)


# czyszczenie danych
def data_cleanup() -> dict[int, dict[str, str]]:
    raw_data = read_file()
    print("No of books in source file: " + str(len(raw_data)))
    print()
    data = []
    # usuwane zostaje 9 rekordow z \ ktory nie poprzedza znaku zakodowanego jako utf-16
    for index in range(len(raw_data)):
        try:
            data.append(utf_16_to_utf_8_conversion(raw_data[index]))
        except ValueError:
            pass
    print("No of books after initial cleanup: " + str(len(data)))
    print()
    temp_dict = insert_data_into_dict(data)
    print("No of books without missing data: " + str(len(temp_dict.keys())))
    print()
    temp_dict = remove_specific_and_broad_genres(temp_dict)
    print("No of books after removing too broad and too specific genres: " + str(len(temp_dict.keys())))
    print()
    temp_dict = remove_overlapping_books(temp_dict)
    print("No of books after removing books with more than one classifiable genre: " + str(len(temp_dict.keys())))
    print()
    # lista gatunkow
    counts = count_genres_occurrences(temp_dict)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("Genre list with book counts: ")
    for item in sorted_counts:
        print("\t" + item[0] + " " + str(item[1]))
    print()
    return insert_into_final_dataset(temp_dict)


# czytanie surowych danych z pliku zrodlowego
def read_file() -> list[str]:
    with io.open('booksummaries/booksummaries.txt', 'r', encoding='utf-8') as file:
        return file.readlines()


# https://reddit.fun/1604/how-print-strings-with-unicode-escape-characters-correctly?fbclid=IwAR2pX93Wcdntba-4u_BNcEmeD8YREvEaK5Peqtw0h9vN8nkvM6qw_WJUNdQ
# konwersja encodingu utf-16 do utf-8
# rzuca wyjatek ValueError exception - w pliku zrodlowym sa \ ktore nie poprzedzaja znakow utf-16
def utf_16_to_utf_8_conversion(data: str) -> str:
    data = list(data)
    for index, value in enumerate(data):
        if value == '\\':
            utf = ''.join([data[index + k + 2] for k in range(4)])
            for k in range(5):
                data.pop(index)
            data[index] = str(chr(int(utf, 16)))
    return ''.join(data)


# wstawienie danych do slownika, w ktorym kluczem jest identyfikator z wikipedii, a wartoscia jest slownik
# w zagniezdzonym slowniku sa 3 klucze: title, genres, summary
# identyfikator freebase, autor i data publikacji sa pomijane
def insert_data_into_dict(data: list[str]) -> dict[int, dict[str, list[str]]]:
    temp_dict = {}
    for line in data:
        # kolumny w pliku sa podzielone tabulatorami
        line = line.split("\t")
        # line[0] to identyfikator z wikipedii, line[2] to tytul, line[5] to json gatunkow, line[6] to streszczenie
        # spacje z poczatku i konca elementu sa usuwane
        line_stripped = [line[index].strip() for index in (0, 2, 5, 6)]
        # usuwane zostaja rekordy, w ktorych interesujace wartosci sa puste lub ktore zawieraja podejrzane wartosci
        valid = line_stripped[0] != '' and line_stripped[1] != '' and line_stripped[2] != '' \
            and line_stripped[3] != '' and line_stripped[3].count("To be added.") == 0 \
            and line_stripped[3].count("=") == 0 and line_stripped[3].count("Plot outline description") == 0 \
            and line_stripped[3].count("\\") == 0 and line_stripped[3].count("/") == 0 \
            and line_stripped[3].count("<") == 0 and line_stripped[3].count("&ndash;") == 0 \
            and line_stripped[3].count("&mdash;") == 0 and line_stripped[3].count("&nbsp;") == 0 \
            and line_stripped[3].count("&#") == 0 and line_stripped[3].count("http") == 0
        if valid:
            # preprocessing
            # zamiana wszystkich liter na male litery
            line_stripped[3] = line_stripped[3].lower()
            # usuniecie slow stopu np the, to, i, can itd
            stop = stopwords.words('english')
            line_stripped[3] = " ".join([word for word in line_stripped[3].split() if word not in stop])
            # usuwanie koncowek gramatycznych ze slow i pozostawienie tylko podstawy slowotworczej
            lemmatizer = WordNetLemmatizer()
            line_stripped[3] = " ".join([lemmatizer.lemmatize(word) for word in line_stripped[3].split()])
            # usuwanie koncowek gramatycznych ze slow (po prostu usuwajac koncowke)
            stemmer = PorterStemmer()
            line_stripped[3] = " ".join([stemmer.stem(word) for word in line_stripped[3].split()])
            # usuniecie znakow interpunkcyjnych
            line_stripped[3] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?",
                                      "", line_stripped[3])
            # zapis danych
            nested_dict = {'title': [line_stripped[1]],
                           'genres': [genre for genre in json.loads(line_stripped[2]).values()],
                           'summary': [line_stripped[3]]}
            temp_dict[int(line_stripped[0])] = nested_dict
    return temp_dict


# zliczanie wystapien kazdego z gatunkow
def count_genres_occurrences(temp_dict: dict[int, dict[str, list[str]]]) -> dict[str, int]:
    counts = {}
    for book in temp_dict.values():
        for genre in book['genres']:
            if genre in counts:
                counts[genre] += 1
            else:
                counts[genre] = 1
    return counts


# usuniecie gatunkow, ktore sa zbyt szerokie lub zbyt waskie
def remove_specific_and_broad_genres(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, list[str]]]:
    counts = count_genres_occurrences(temp_dict)
    for book in temp_dict.values():
        for genre in deepcopy(book['genres']):
            # if counts[genre] < 1000 or counts[genre] > 2000: -> 2
            # if counts[genre] < 1000 or counts[genre] > 3000: -> 5
            # if counts[genre] < 1000: -> 7
            # if counts[genre] < 500 or counts[genre] > 3000: -> 10
            # zbyt waskie gatunki maja mniej niz 500 wystapien, zbyt szerokie wiecej niz 3000
            if counts[genre] < 1000 or counts[genre] > 3000:
                book['genres'].remove(genre)
    helper = deepcopy(temp_dict)
    # usuniecie ksiazek, ktore w wyniku czyszczenia gatunkow nie maja juz przypisanego zadnego gatunku
    for key in helper:
        if len(temp_dict[key]['genres']) == 0:
            temp_dict.pop(key)
    return temp_dict


# usuniecie ksiazek, ktore maja przypisane kilka z gatunkow z listy
def remove_overlapping_books(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, list[str]]]:
    counts = count_genres_occurrences(temp_dict)
    helper = deepcopy(temp_dict)
    for key in helper:
        genre_count = 0
        for genre in helper[key]['genres']:
            if genre in counts:
                genre_count += 1
            if genre_count > 1:
                temp_dict.pop(key)
                break
    return temp_dict


# konwersja do ostatecznej reprezentacji danych
def insert_into_final_dataset(temp_dict: dict[int, dict[str, list[str]]]) -> dict[int, dict[str, str]]:
    dataset = {}
    for key in temp_dict:
        dataset[key] = {'title': temp_dict[key]['title'][0],
                        'genre': temp_dict[key]['genres'][0],
                        'summary': temp_dict[key]['summary'][0]}
    return dataset
