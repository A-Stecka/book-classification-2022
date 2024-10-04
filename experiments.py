from classifiers import multinomial_naive_bayes_cross_val, support_vector_machine_cross_val


def test():
    print("-------------------------------------------------------------------------- ALPHA", end="\n\n")
    test_alpha()
    print("------------------------------------------------------------------------------ C", end="\n\n")
    test_c()
    print("--------------------------------------------------------------------------- LOSS", end="\n\n")
    test_loss()
    print("-------------------------------------------------------------------------- MAX_F", end="\n\n")
    test_max_f()
    print("-------------------------------------------------------------------- NGRAM_RANGE", end="\n\n")
    test_ngram_range()
    print("------------------------------------------------------------------------- MAX_DF", end="\n\n")
    test_max_df()
    print("------------------------------------------------------------------------- MIN_DF", end="\n\n")
    test_min_df()
    print("---------------------------------------------------------------------- TFID MODE", end="\n\n")
    test_tfid_mode()


def test_alpha():
    alpha_params = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for alpha in alpha_params:
        print("-------------------------------------------------------------------------- " + str(alpha))
        results = multinomial_naive_bayes_cross_val(False, 0.5, 0.001, 5000, (1, 3), True, alpha)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_c():
    c_params = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c in c_params:
        print("-------------------------------------------------------------------------- " + str(c))
        results = support_vector_machine_cross_val(False, 0.5, 0.001, 5000, (1, 3), True, c, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_loss():
    loss_params = ['hinge', 'squared_hinge']
    for loss in loss_params:
        print("-------------------------------------------------------------------------- " + loss)
        results = support_vector_machine_cross_val(False, 0.5, 0.001, 5000, (1, 3), True, 0.1, loss)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_max_f():
    print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES", end="\n\n")
    test_max_f_mnb()
    print("--------------------------------------------------------- SUPPORT VECTOR MACHINE", end="\n\n")
    test_max_f_svm()


def test_max_f_mnb():
    max_f_params = [1, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    for max_f in max_f_params:
        print("-------------------------------------------------------------------------- " + str(max_f))
        results = multinomial_naive_bayes_cross_val(False, 0.5, 0.001, max_f, (1, 3), True, 0.1)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_max_f_svm():
    max_f_params = [1, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    for max_f in max_f_params:
        print("-------------------------------------------------------------------------- " + str(max_f))
        results = support_vector_machine_cross_val(False, 0.5, 0.001, max_f, (1, 3), True, 0.1, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_ngram_range():
    print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES", end="\n\n")
    test_ngram_range_mnb()
    print("--------------------------------------------------------- SUPPORT VECTOR MACHINE", end="\n\n")
    test_ngram_range_svm()


def test_ngram_range_mnb():
    ngram_range_params = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
    for ngram_range in ngram_range_params:
        print("-------------------------------------------------------------------------- " + str(ngram_range))
        results = multinomial_naive_bayes_cross_val(False, 0.5, 0.001, 5000, ngram_range, True, 0.1)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_ngram_range_svm():
    ngram_range_params = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
    for ngram_range in ngram_range_params:
        print("-------------------------------------------------------------------------- " + str(ngram_range))
        results = support_vector_machine_cross_val(False, 0.5, 0.001, 5000, ngram_range, True, 0.1, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_max_df():
    print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES", end="\n\n")
    test_max_df_mnb()
    print("--------------------------------------------------------- SUPPORT VECTOR MACHINE", end="\n\n")
    test_max_df_svm()


def test_max_df_mnb():
    max_df_params = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for max_df in max_df_params:
        print("-------------------------------------------------------------------------- " + str(max_df))
        results = multinomial_naive_bayes_cross_val(False, max_df, 0.001, 5000, (1, 3), True, 0.1)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_max_df_svm():
    max_df_params = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for max_df in max_df_params:
        print("-------------------------------------------------------------------------- " + str(max_df))
        results = support_vector_machine_cross_val(False, max_df, 0.001, 5000, (1, 3), True, 0.1, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_min_df():
    print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES", end="\n\n")
    test_min_df_mnb()
    print("--------------------------------------------------------- SUPPORT VECTOR MACHINE", end="\n\n")
    test_min_df_svm()


def test_min_df_mnb():
    min_df_params = [0.00001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    for min_df in min_df_params:
        print("-------------------------------------------------------------------------- " + str(min_df))
        results = multinomial_naive_bayes_cross_val(False, 0.5, min_df, 5000, (1, 3), True, 0.1)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_min_df_svm():
    min_df_params = [0.00001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    for min_df in min_df_params:
        print("-------------------------------------------------------------------------- " + str(min_df))
        results = support_vector_machine_cross_val(False, 0.5, min_df, 5000, (1, 3), True, 0.1, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_tfid_mode():
    print("-------------------------------------------------------- MULTINOMIAL NAIVE BAYES", end="\n\n")
    test_tfid_mode_mnb()
    print("--------------------------------------------------------- SUPPORT VECTOR MACHINE", end="\n\n")
    test_tfid_mode_svm()


def test_tfid_mode_mnb():
    tfid_mode_params = [True, False]
    for tfid_mode in tfid_mode_params:
        print("-------------------------------------------------------------------------- " + str(tfid_mode))
        results = multinomial_naive_bayes_cross_val(False, 0.5, 0.001, 5000, (1, 3), tfid_mode, 0.1)
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))


def test_tfid_mode_svm():
    tfid_mode_params = [True, False]
    for tfid_mode in tfid_mode_params:
        print("-------------------------------------------------------------------------- " + str(tfid_mode))
        results = support_vector_machine_cross_val(False, 0.5, 0.001, 5000, (1, 3), tfid_mode, 0.1, 'squared_hinge')
        print("Prediction accuracy mean: ", end="")
        print(round(results[0], 3))
        print("Fit time mean: ", end="")
        print(round(results[1], 3))
        print("Prediction time mean: ", end="")
        print(round(results[2], 3))
