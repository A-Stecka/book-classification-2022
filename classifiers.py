import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


class Colors:
    GREEN = '\u001b[92m'
    END = '\033[0m'


# dostrajanie hiperparametrow
def tuning():
    tuning_mnb()
    tuning_svm()


# dostrajanie hiperparametrow dla klasyfikatora multinomial naive bayes
def tuning_mnb():
    max_df_params = [0.5, 0.7, 0.9]
    min_df_params = [0.001, 0.01, 0.1]
    max_f_params = [1000, 3000, 5000]
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    alpha_params = [0.001, 0.01, 0.1]
    best_mean = 0.0
    best_params = {'max_df': 0.0, 'min_df': 0.0, 'max_f': 0, 'ngram_range': (0, 0), 'alpha': 0.0}
    for max_df in max_df_params:
        for min_df in min_df_params:
            for max_f in max_f_params:
                for ngram_range in ngram_range_params:
                    for alpha in alpha_params:
                        new_mean = multinomial_naive_bayes_cross_val(False, max_df, min_df, max_f, ngram_range, True,
                                                                     alpha)
                        if new_mean > best_mean:
                            best_mean = new_mean
                            best_params['max_df'] = max_df
                            best_params['min_df'] = min_df
                            best_params['max_f'] = max_f
                            best_params['ngram_range'] = ngram_range
                            best_params['alpha'] = alpha
    print("------------------------------------------------- TUNING MULTINOMIAL NAIVE BAYES")
    print("Best mean: " + str(round(best_mean, 3)))
    print("Parameters for best mean: ")
    for key in best_params:
        print("\t" + key + ": " + str(best_params[key]))
    print()


# klasyfikator naive bayes
def multinomial_naive_bayes(train_size: float, max_df: float, min_df: float, max_f: int, ngram_range: (int, int),
                            alpha: float):
    # multinomial_naive_bayes_reg(True, train_size, max_df, min_df, max_f, ngram_range, True, alpha)
    multinomial_naive_bayes_cross_val(True, max_df, min_df, max_f, ngram_range, True, alpha)


def multinomial_naive_bayes_reg(verbose: bool, train_size: float, max_df: float, min_df: float, max_f: int,
                                ngram_range: (int, int), use_idf: bool, alpha: float) -> float:
    if verbose:
        print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES")
        print("------------------------------------------------------------ no cross validation")
        print()
    dataset = pandas.read_csv('booksummaries/processed_data.csv', names=['id', 'genre', 'summary'])
    # podzial zbioru danych na zbior trenujacy i testujacy / walidujacy
    x_train, x_validate, y_train, y_validate = train_test_split(dataset['summary'], dataset['genre'],
                                                                train_size=train_size)
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # CountVectorizer zamienia tablice dokumentow na macierz liczby wystapien tokenow (cech)
    # max_df, min_df - gorna i dolna granica ignorowania slow o zbyt czestym lub zbyt rzadkim wystepowaniu
    # max_features - maksymalna liczba wyznaczonych cech
    # ngram_range - liczba slow ktora moze byc traktowana jako cecha
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_f, ngram_range=ngram_range)
    x_train_counts = count_vectorizer.fit_transform(x_train)
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    # TfidfTransformer zamienia macierz liczby wystapien na znormalizowana reprezentacje tf lub tf-idf
    # tf = czestosc wystepowania slowa w dokumencie
    # tf-idf = czestosc wystepowania slowa w dokumencie w stosunku do rozmiaru dokumentu
    # korzystam z tf-idf bo streszczenia fabuly maja rozna dlugosc
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    # utworzenie i wyuczenie modelu uczenia maszynowego
    # model jest uczony na odpowiednio przygotowanym zbiorze danych wejsciowych i zbioru wyjsc dla kazdego wejscia
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    # alpha - parametr wygladzania krzywej
    ml_model = MultinomialNB(alpha=alpha)
    ml_model.fit(x_train_tfidf, y_train)
    # model wyznacza zbior gatunkow dla zbioru streszczen na podstawie tego czego sie nauczyl
    y_pred = ml_model.predict(count_vectorizer.transform(x_validate))
    pred_accuracy_mean = numpy.mean(y_pred == y_validate)
    if verbose:
        print("Prediction accuracy mean: ", end="")
        print(round(pred_accuracy_mean, 3))
        print()
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        # precision - zdolnosc modelu do nie przypisywania do streszczenia gatunku, jezeli ksiazka nie jest tego gatunku
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
        # recall - zdolnosc modelu do znalezienia wszystkich streszczen ksiazek danego gatunku
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # f1-score - srednia harmoniczna z precision i recall, gdzie 1 to najlepsza wartosc, a 0 najgorsza
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        # support = liczba wystapien kazdego z gatunkow w y_validate
        print("Classification report:")
        print(classification_report(y_validate, y_pred))
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # confusion matrix - macierz sluzaca do okreslenia prawdziwosci klasyfikacji
        # wartosc w komorce i,j reprezentuje liczbe streszczen, ktore byly gatunku i, a ktorym przypisano gatunek j
        # z tego powodu na przekatnej powinny byc najwieksze wartosci
        matrix = confusion_matrix(y_validate, y_pred)
        print("Confusion matrix:")
        for i in range(len(matrix)):
            print("\t", end="")
            for j in range(len(matrix[i])):
                if i == j:
                    print(f'{Colors.GREEN}{matrix[i][j]:5d}{Colors.END}', end=" ")
                else:
                    print(f'{matrix[i][j]:5d}', end=" ")
            print()
        print()
    return pred_accuracy_mean


def multinomial_naive_bayes_cross_val(verbose: bool, max_df: float, min_df: float, max_f: int, ngram_range: (int, int),
                                      use_idf: bool, alpha: float) -> (float, float, float):
    if verbose:
        print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES")
        print("--------------------------------------------------------------- cross validation")
        print()
    dataset = pandas.read_csv('booksummaries/processed_data.csv', names=['id', 'genre', 'summary'])
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
    # zgodnie z wymaganiami w liscie zadan dokonywana jest 10-krotna walidacja krzyzowa
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_f, ngram_range=ngram_range)
    x_train_counts_full = count_vectorizer.fit_transform(dataset['summary'])
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    x_train_tfidf_full = tfidf_transformer.fit_transform(x_train_counts_full)
    ml_model = MultinomialNB(alpha=alpha)
    cross_validation = cross_validate(ml_model, x_train_tfidf_full, dataset['genre'], cv=10)
    pred_accuracy_mean = numpy.mean(cross_validation['test_score'])
    fit_time_mean = numpy.mean(cross_validation['fit_time'])
    pred_time_mean = numpy.mean(cross_validation['score_time'])
    if verbose:
        print("Fit times for each iteration:")
        for element in cross_validation['fit_time']:
            print("\t" + str(round(element, 5)))
        print()
        print("Score times for each iteration:")
        for element in cross_validation['score_time']:
            print("\t" + str(round(element, 5)))
        print()
        print("Test scores for each iteration:")
        for element in cross_validation['test_score']:
            print("\t" + str(round(element, 3)))
        print()
        print("Prediction accuracy mean: ", end="")
        print(round(pred_accuracy_mean, 3))
        print()
    return pred_accuracy_mean, fit_time_mean, pred_time_mean


# dostrajanie hiperparametrow dla maszyny wektorow nosnych
def tuning_svm():
    max_df_params = [0.5, 0.7, 0.9]
    min_df_params = [0.001, 0.01, 0.1]
    max_f_params = [1000, 3000, 5000]
    ngram_range_params = [(1, 1), (1, 2), (1, 3)]
    c_params = [0.1, 1, 2]
    loss_params = ['hinge', 'squared_hinge']
    best_mean = 0.0
    best_params = {'max_df': 0.0, 'min_df': 0.0, 'max_f': 0, 'ngram_range': (0, 0), 'c': 0, 'loss': 'hinge'}
    for max_df in max_df_params:
        for min_df in min_df_params:
            for max_f in max_f_params:
                for ngram_range in ngram_range_params:
                    for c in c_params:
                        for loss in loss_params:
                            new_mean = support_vector_machine_cross_val(False, max_df, min_df, max_f, ngram_range,
                                                                        True, c, loss)
                            if new_mean > best_mean:
                                best_mean = new_mean
                                best_params['max_df'] = max_df
                                best_params['min_df'] = min_df
                                best_params['max_f'] = max_f
                                best_params['ngram_range'] = ngram_range
                                best_params['c'] = c
                                best_params['loss'] = loss
    print("------------------------------------------------- TUNING SUPPORT VECTOR MACHINES")
    print("------------------------------------------------- TUNING MULTINOMIAL NAIVE BAYES")
    print("Best mean: " + str(round(best_mean, 3)))
    print("Parameters for best mean: ")
    for key in best_params:
        print("\t" + key + ": " + str(best_params[key]))
    print()


# maszyna wektorow nosnych
def support_vector_machine(train_size: float, max_df: float, min_df: float, max_f: int, ngram_range: (int, int),
                           c: float, loss: str):
    # support_vector_machine_reg(True, train_size, max_df, min_df, max_f, ngram_range, True, c, loss)
    support_vector_machine_cross_val(True, max_df, min_df, max_f, ngram_range, True, c, loss)


def support_vector_machine_reg(verbose: bool, train_size: float, max_df: float, min_df: float, max_f: int,
                               ngram_range: (int, int), use_idf: bool, c: float, loss: str) -> float:
    if verbose:
        print("--------------------------------------------------------- SUPPORT VECTOR MACHINE")
        print("------------------------------------------------------------ no cross validation")
        print()
    dataset = pandas.read_csv('booksummaries/processed_data.csv', names=['id', 'genre', 'summary'])
    # podzial zbioru danych na zbior trenujacy i testujacy / walidujacy
    x_train, x_validate, y_train, y_validate = train_test_split(dataset['summary'], dataset['genre'],
                                                                train_size=train_size)
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # CountVectorizer zamienia tablice dokumentow na macierz liczby wystapien tokenow (cech)
    # max_df, min_df - gorna i dolna granicza ignorowania slow o zbyt czestym lub zbyt rzadkim wystepowaniu
    # max_features - maksymalna liczba wyznaczonych cech
    # ngram_range - liczba slow ktora moze byc traktowana jako cecha
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_f, ngram_range=ngram_range)
    x_train_counts = count_vectorizer.fit_transform(x_train)
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    # TfidfTransformer zamienia macierz liczby wystapien na znormalizowana reprezentacje tf lub tf-idf
    # tf = czestosc wystepowania slowa w dokumencie
    # tf-idf = czestosc wystepowania slowa w dokumencie w stosunku do rozmiaru dokumentu
    # korzystam z tf-idf bo streszczenia fabuly maja rozna dlugosc
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    # utworzenie i wyuczenie modelu uczenia maszynowego
    # model jest uczony na odpowiednio przygotowanym zbiorze danych wejsciowych i zbioru wyjsc dla kazdego wejscia
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    # class_weight - automatycznie dostraja wagi w zaleznosci od czestosci wystepowania w zbiorze treningowym
    # C - parametr regularyzacji
    # regularyzacja - zmniejsza blad przez odpowiednie dopasowanie funkcji i zapobieganie zjawisku overfitting
    # loss - wyznacza funkcje loss
    # funkcja loss - funkcja obliczajaca odleglosc miedzy aktualnym wyjsciem i oczekiwanym wyjsciem
    ml_model = LinearSVC(class_weight='balanced', C=c, loss=loss)
    ml_model.fit(x_train_tfidf, y_train)
    # model wyznacza zbior gatunkow dla zbioru streszczen na podstawie tego czego sie nauczyl
    y_pred = ml_model.predict(count_vectorizer.transform(x_validate))
    print("Prediction accuracy mean: ", end="")
    pred_accuracy_mean = numpy.mean(y_pred == y_validate)
    if verbose:
        print(round(pred_accuracy_mean, 3))
        print()
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
        # precision = zdolnosc modelu do nie przypisywania do streszczenia gatunku, jezeli ksiazka nie jest tego gatunku
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
        # recall = zdolnosc modelu do znalezienia wszystkich streszczen ksiazek danego gatunku
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # f1-score = srednia harmoniczna z precision i recall, gdzie 1 to najlepsza wartosc, a 0 najgorsza
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        # support = liczba wystapien kazdego z gatunkow w y_validate
        print("Classification report:")
        print(classification_report(y_validate, y_pred))
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # confusion matrix = macierz sluzaca do okreslenia prawdziwosci klasyfikacji
        # wartosc w komorce i,j reprezentuje liczbe streszczen, ktore byly gatunku i, a ktorym przypisano gatunek j
        # z tego powodu na przekatnej powinny byc najwieksze wartosci
        matrix = confusion_matrix(y_validate, y_pred)
        print("Confusion matrix:")
        for i in range(len(matrix)):
            print("\t", end="")
            for j in range(len(matrix[i])):
                if i == j:
                    print(f'{Colors.GREEN}{matrix[i][j]:5d}{Colors.END}', end=" ")
                else:
                    print(f'{matrix[i][j]:5d}', end=" ")
            print()
        print()
    return pred_accuracy_mean


def support_vector_machine_cross_val(verbose: bool, max_df: float, min_df: float, max_f: int, ngram_range: (int, int),
                                     use_idf: bool, c: float, loss: str) -> (float, float, float):
    if verbose:
        print("--------------------------------------------------------- SUPPORT VECTOR MACHINE")
        print("--------------------------------------------------------------- cross validation")
        print()
    dataset = pandas.read_csv('booksummaries/processed_data.csv', names=['id', 'genre', 'summary'])
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
    # zgodnie z wymaganiami w liscie zadan dokonywana jest 10-krotna walidacja krzyzowa
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_f, ngram_range=ngram_range)
    x_train_counts_full = count_vectorizer.fit_transform(dataset['summary'])
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    x_train_tfidf_full = tfidf_transformer.fit_transform(x_train_counts_full)
    ml_model = LinearSVC(class_weight='balanced', C=c, loss=loss)
    cross_validation = cross_validate(ml_model, x_train_tfidf_full, dataset['genre'], cv=10)
    pred_accuracy_mean = numpy.mean(cross_validation['test_score'])
    fit_time_mean = numpy.mean(cross_validation['fit_time'])
    pred_time_mean = numpy.mean(cross_validation['score_time'])
    if verbose:
        print("Fit times for each iteration:")
        for element in cross_validation['fit_time']:
            print("\t" + str(round(element, 5)))
        print()
        print("Score times for each iteration:")
        for element in cross_validation['score_time']:
            print("\t" + str(round(element, 5)))
        print()
        print("Test scores for each iteration:")
        for element in cross_validation['test_score']:
            print("\t" + str(round(element, 3)))
        print()
        print("Prediction accuracy mean: ", end="")
        print(round(pred_accuracy_mean, 3))
        print()
    return pred_accuracy_mean, fit_time_mean, pred_time_mean
