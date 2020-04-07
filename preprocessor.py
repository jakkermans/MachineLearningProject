#!/usr/bin/python3
"""
File name:  preprocessor.py
Course:     Machine Learning Project
Authors:    Martijn E.N.F.L. Schendstok (s2688174)
            Jannick J.C. Akkermans      (s3429075)
            Niels Westeneng             (s3469735)
Date:       March 2020
"""

from os import listdir  # to read files
from os.path import isfile, join  # to read files
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
import sys
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
import numpy
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score, confusion_matrix


def get_n_grams(tokens, n=2):
    '''
    Create n-grams list from token list
    :param tokens:  List of strings
    :param n:       Number of n-grams, set standard to bi-grams (2)
    :return:        List of strings containing n-grams
    '''

    if n < 1:
        print("Error pre_process_tokens: n must be 1 or more!", file=sys.stderr)

    n_grams = []
    for t in range(len(tokens)):
        n_gram = ""
        for x in range(n):
            if t + x < len(tokens):
                n_gram += tokens[t + x]
                n_grams.append(n_gram)
                n_gram += " "

    return n_grams


def replace_numbers(tokens):
    '''
    Replace numbers by string 'NUMBER'
    :param tokens:      List of strings
    :return: tokens:    List of strings, digits replace by "Number"
    '''

    for x in range(len(tokens)):
        if tokens[x].isdigit():
            tokens[x] = "#"

    return tokens


def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]  # Return a list of files in a certain folder


def read_files(categories, author_data, traits, n_grams=1):
    """
    This function reads in all the files from a folder. For each file, the function tokenizes the data and lowercases each token.
    The author of the file is derived from the dictionary of authors. For each of the OCEAN scores for this author, the function checks whether
    it exceeds 50. If it does, the personality trait is added to a list. This list, together with the tokenized data, is added to a list of read data
    :param categories:  list of folders to process
    :param author_data: Dictionary of authors and their corresponding data
    :param traits:
    :param n_grams:     Number of n-grams, set standard to uni-grams (1)
    :return:            List with read data, each entry in the form (tokenized_data, personality_traits)
    """

    feats = list()
    punct_list = ['.', ',', '?', ':', '(', ')', '!', "'", '`', '...', '``', "''", '"', "’", "”", "“", "’", "-", ";", "‘" "="]
    punct = ".,?:()!'```\"’”“’-‘"

    print("\n##### Reading files...")
    for category in categories:
        stemmer = SnowballStemmer("dutch")
        files = get_filenames_in_folder(category)
        num_files = 0
        for f in files:
            personality_traits = []
            try:
                data = open(category + '/' + f, 'r', encoding='UTF-8').read()
                id = f.split('_')[0]
                if id in author_data:
                    author = author_data[id]
                    if author[5] != '----':
                        tokens = word_tokenize(data)
                        lower_tokens = [token.lower() for token in tokens]
                        no_punct_tokens = [token.replace("-", " ").replace(".", "").replace(",", "").strip(punct).strip() for token in lower_tokens if token not in punct_list]
                        word_tokens = replace_numbers([t.strip() for token in no_punct_tokens for t in token.split()])
                        stem_tokens = [stemmer.stem(token) for token in word_tokens]
                        use_tokens = get_n_grams(stem_tokens, n=n_grams)
                        scores = author[5].split('-')
                        for i in range(len(scores)):
                            if int(scores[i]) >= 50:
                                personality_traits.append(traits[i])
                        feats.append((use_tokens, personality_traits))
                        # print len(tokens)
                        num_files += 1

            #if num_files>=50: # you may want to de-comment this and the next line if you're doing tests (it just loads N documents instead of the whole collection so it runs faster
            #	break
            except UnicodeDecodeError:
                print('Decode error')

        print("  Category %s, %i files read" % (category, num_files))

    print("  Total, %i files read\n" % (len(feats)))
    return feats


def read_authordata(authorfile):
    """
    This function reads in all the author data. Whenever the country of the author is The Netherlands or South Africa, it is changed to
    TheNetherlands or SouthAfrica so that it is considered as one token are not considered seperate tokens. Each author is added to a dictionary as key
    and the corresponding data of the author is added as value.
    :param authorfile: Text file with all the author data
    :return: Dictionary with authors and their data
    """

    authors = {}
    country_token = '[A-Z][a-z]+ [A-Z][a-z]+'
    with open(authorfile, 'r') as datafile:
        for line in datafile.readlines():
            token = re.findall(country_token, line)
            if token != []:
                for match in token:
                    if match == 'The Netherlands':
                        txt = re.sub(match, 'TheNetherlands', line)
                    else:
                        txt = re.sub(match, 'SouthAfrica', line)
                    txt = txt.strip().split()
                    authors[txt[0]] = txt[1:]
            else:
                line = line.strip().split()
                authors[line[0]] = line[1:]

    return authors


def high_information_words(files, score_fn=BigramAssocMeasures.chi_sq, min_score=50):
    word_dict = FreqDist()
    ocean_word_dict = ConditionalFreqDist()
    hiw_categories = []

    for file in files:
        # For each token, add 1 to the overall FreqDist and 1 to the ConditionalFreqDist under the current personality trait
        for token in file[0]:
            for trait in file[1]:
                ocean_word_dict[trait][token] += 1
            word_dict[token] += 1

    n_xx = ocean_word_dict.N()  # Get the total number of recordings in the ConditionalFreqDist
    high_info_words = set()

    for condition in ocean_word_dict.conditions():
        n_xi = ocean_word_dict[condition].N()  # Get the number of recordings for each personality trait
        word_scores = defaultdict(int)

        for word, n_ii in ocean_word_dict[condition].items():
            n_ix = word_dict[word]  # Get total number of recordings of a token
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        bw = list({k for k, v in sorted(word_scores.items(), key=lambda x: x[1], reverse=True)})
        high_info_words |= set(bestwords)
        hiw_categories.append((condition, bw[:10]))

    return high_info_words, hiw_categories


def get_fit(files, high_info_words):
    """
    This function creates the x and y lists used in the .fit function of the MultinomialNB classifier.
    :param files: all data of the files, created by read_file
    :return: Labels (0 or 1 per sentence) per personality type and all sentences in list feats
    """
    feats = []
    label_open = []
    label_extra = []
    label_con = []
    label_neu = []
    label_agree = []
    for file in files:
        tokens = file[0]
        used_tokens = [token for token in tokens if token in high_info_words]
        personalities = file[1]
        feats.append(used_tokens)
        if "Openness" in personalities:
            label_open.append(1)
        else:
            label_open.append(0)
        if "Concientiousness" in personalities:
            label_con.append(1)
        else:
            label_con.append(0)
        if "Extravertness" in personalities:
            label_extra.append(1)
        else:
            label_extra.append(0)
        if "Agreeableness" in personalities:
            label_agree.append(1)
        else:
            label_agree.append(0)
        if "Neuroticism" in personalities:
            label_neu.append(1)
        else:
            label_neu.append(0)
    return label_open, label_extra, label_con, label_neu, label_agree, feats


def get_classifier(label_open, label_extra, label_con, label_neu, label_agree, x_feats):
    """
    This function creates all classifier and fills them with data, all per personality type.
    :param files: x_feats and y labels created by get_fit
    :return: A classifier for every personality type
    """

    classifier_open = MultinomialNB()
    classifier_extra = MultinomialNB()
    classifier_con = MultinomialNB()
    classifier_neu = MultinomialNB()
    classifier_agree = MultinomialNB()
    classifier_open.fit(x_feats, label_open)
    classifier_extra.fit(x_feats, label_extra)
    classifier_con.fit(x_feats, label_con)
    classifier_neu.fit(x_feats, label_neu)
    classifier_agree.fit(x_feats, label_agree)
    return classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree


def evaluation(classifier_dict, x_feats, label_dict):
    """
    This function print the accuracy scores of the testing data.
    :param files: all classifiers created by classifier(), x_feats, and all y-labels created by get_fit
    :return: Prints the accuracies and returns them too
    """

    open_acc = classifier_dict["open"].score(x_feats, label_dict["open"])
    extra_acc = classifier_dict["extra"].score(x_feats, label_dict["extra"])
    conc_acc = classifier_dict["con"].score(x_feats, label_dict["con"])
    neuro_acc = classifier_dict["neu"].score(x_feats, label_dict["neu"])
    agree_acc = classifier_dict["agree"].score(x_feats, label_dict["agree"])
    av_acc = sum((open_acc, extra_acc, conc_acc, neuro_acc, agree_acc))/5

    print("Accuracy Openness:", open_acc)
    print("Accuracy Extravertness:", extra_acc)
    print("Accuracy Concientiousness:", conc_acc)
    print("Accuracy Neuroticism:", neuro_acc)
    print("Accuracy Agreeableness:", agree_acc)
    print("Average Accuracy:", round(av_acc, 2), "\n")

    return open_acc, extra_acc, conc_acc, neuro_acc, agree_acc, av_acc


def n_cross_validation(n, label_open, label_extra, label_con, label_neu, label_agree, feats, f_score=False):
    """
    This function does the cross validation. It prints the accuracies in the end
    :param n: amount of cross validations, all x and y values as labels or feats.
    :return: Prints total accuracies in the end
    """

    mlb = MultiLabelBinarizer()
    x_feats = mlb.fit_transform(feats)
    gap = int(len(x_feats) / n)
    tot_open_acc = tot_extra_acc = tot_conc_acc = tot_neuro_acc = tot_agree_acc = tot_av_acc = 0
    f_measures = defaultdict(int)
    tn = fp = fn = tp = 0

    for i in range(n):
        print("#### Cross Validation {}".format(i + 1))
        i1 = i * gap
        i2 = (i + 1) * gap
        if i == 0:
            x_train = x_feats[i2:]
            open_train = label_open[i2:]
            extra_train = label_extra[i2:]
            con_train = label_con[i2:]
            neu_train = label_neu[i2:]
            agree_train = label_agree[i2:]
            x_test = x_feats[:i2]
            open_test = label_open[:i2]
            extra_test = label_extra[:i2]
            con_test = label_con[:i2]
            neu_test = label_neu[:i2]
            agree_test = label_agree[:i2]
        elif i == (n - 1):
            x_train = x_feats[:i1]
            open_train = label_open[:i1]
            extra_train = label_extra[:i1]
            con_train = label_con[:i1]
            neu_train = label_neu[:i1]
            agree_train = label_agree[:i1]
            x_test = x_feats[i1:]
            open_test = label_open[i1:]
            extra_test = label_extra[i1:]
            con_test = label_con[i1:]
            neu_test = label_neu[i1:]
            agree_test = label_agree[i1:]
        else:
            x_train = numpy.concatenate((x_feats[:i1], x_feats[i2:]))
            open_train = label_open[:i1] + label_open[i2:]
            extra_train = label_extra[:i1] + label_extra[i2:]
            con_train = label_con[:i1] + label_con[i2:]
            neu_train = label_neu[:i1] + label_neu[i2:]
            agree_train = label_agree[:i1] + label_agree[i2:]
            x_test = x_feats[i1:i2]
            open_test = label_open[i1:i2]
            extra_test = label_extra[i1:i2]
            con_test = label_con[i1:i2]
            neu_test = label_neu[i1:i2]
            agree_test = label_agree[i1:i2]

        classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree = get_classifier(open_train,
                                                                                                             extra_train,
                                                                                                             con_train,
                                                                                                             neu_train,
                                                                                                             agree_train,
                                                                                                             x_train)
        classifier_dict = {"open": classifier_open,
                           "extra": classifier_extra,
                           "con": classifier_con,
                           "neu": classifier_neu,
                           "agree": classifier_agree}
        test_dict = {"open": open_test,
                     "extra": extra_test,
                     "con": con_test,
                     "neu": neu_test,
                     "agree": agree_test}

        open_acc, extra_acc, conc_acc, neuro_acc, agree_acc, av_acc = evaluation(classifier_dict,
                                                                                 x_test,
                                                                                 test_dict)
        tot_open_acc += open_acc
        tot_extra_acc += extra_acc
        tot_conc_acc += conc_acc
        tot_neuro_acc += neuro_acc
        tot_agree_acc += agree_acc
        tot_av_acc += av_acc

        # Get F-scores
        if f_score:
            for label in classifier_dict:
                y_true = test_dict[label]
                y_pred = classifier_dict[label].predict(x_test)
                f_measures[label] += f1_score(y_true,
                                              y_pred,
                                              average='macro')
                tn_x, fp_x, fn_x, tp_x = confusion_matrix(y_true, y_pred).ravel()
                tn += tn_x
                fp += fp_x
                fn += fn_x
                tp += tp_x

    print("#### Averages")
    print("Average Accuracy Openness:", round(tot_open_acc/n, 2))
    print("Average Accuracy Extravertness:", round(tot_extra_acc/n, 2))
    print("Average Accuracy Concientiousness:", round(tot_conc_acc/n, 2))
    print("Average Accuracy Neuroticism:", round(tot_neuro_acc/n, 2))
    print("Average Accuracy Agreeableness:", round(tot_agree_acc/n, 2))
    print("Average Average Accuracy:", round(tot_av_acc/n, 2), "\n")

    if f_score:
        print("#### F-score")
        print("Average F-score Openness:", round(f_measures["open"] / n, 2))
        print("Average F-score Extravertness:", round(f_measures["extra"] / n, 2))
        print("Average F-score Concientiousness:", round(f_measures["con"] / n, 2))
        print("Average F-score Neuroticism:", round(f_measures["neu"] / n, 2))
        print("Average F-score Agreeableness:", round(f_measures["agree"] / n, 2))
        tot_av_f1 = (f_measures["open"] + f_measures["extra"] + f_measures["con"] + f_measures["neu"] + f_measures["agree"])/5
        print("Average F-score Accuracy:", round(tot_av_f1 / n, 2), "\n")

        print("#### Confusion Matrix")
        print("{0:^15}{1:^23}".format("", ("-" * 23)))
        print("{0:^15}|{1:^21}|".format("", "Prediction classes"))
        print("{0:^15}{1:^23}".format("", ("-" * 23)))
        print("{0:^15}|{1:^10}|{2:^10}|".format("", "+", "-"))
        print("-" * 38)
        print("|{0:<8}|{1:^5}|{2:^10}|{3:^10}|".format("Actual", "+", tp, fn))
        print("|{0:<8}{1:^29}".format("", ("-" * 29)))
        print("|{0:<8}|{1:^5}|{2:^10}|{3:^10}|".format("Class", "-", fp, tn))
        print("-" * 38, "\n")

    return round(tot_av_acc/n, 2)


def get_high_information_words(hiw_categories):
    print("#### Best high information words per personality trait")
    print("# {0:<38} # {1:<38} # {2:<38} # {3:<38} # {4:<38}".format(hiw_categories[0][0],
                                                                     hiw_categories[1][0],
                                                                     hiw_categories[2][0],
                                                                     hiw_categories[3][0],
                                                                     hiw_categories[4][0]))
    for open_hiw, extra_hiw, conc_hiw, neuro_hiw, agree_hiw in zip(hiw_categories[0][1],
                                                                   hiw_categories[1][1],
                                                                   hiw_categories[2][1],
                                                                   hiw_categories[3][1],
                                                                   hiw_categories[4][1]):
        print("{0:<40} {1:<40} {2:<40} {3:<40} {4:<40}".format(open_hiw,
                                                               agree_hiw,
                                                               conc_hiw,
                                                               neuro_hiw,
                                                               agree_hiw))


def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    author_data = read_authordata(args[0])
    traits = ['Openness',
              'Concientiousness',
              'Extravertness',
              'Agreeableness',
              'Neuroticism']
    files = read_files(args[1:],
                       author_data,
                       traits,
                       n_grams=5)
    high_info, hiw_categories = high_information_words(files,
                                                       min_score=13)
    label_open, label_extra, label_con, label_neu, label_agree, feats = get_fit(files,
                                                                                high_info)

    n_cross_validation(10,
                       label_open,
                       label_extra,
                       label_con,
                       label_neu,
                       label_agree,
                       feats,
                       f_score=True)

    get_high_information_words(hiw_categories)


if __name__ == "__main__":
    main()
