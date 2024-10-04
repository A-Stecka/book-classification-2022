from data_reader import process_and_import_into_csv
from data_reader import read_from_csv
from data_analyser import dataset_analysis
from classifiers import multinomial_naive_bayes, support_vector_machine, tuning
from experiments import test
import pandas


if __name__ == '__main__':
    # process_and_import_into_csv()
    # dataset_to_analyse = read_from_csv()
    # dataset_analysis(dataset_to_analyse)
    multinomial_naive_bayes(0.75, 0.5, 0.001, 5000, (1, 3), 0.1)
    support_vector_machine(0.75, 0.5, 0.001, 5000, (1, 3), 0.1, 'squared_hinge')
    # tuning()
    # test()
