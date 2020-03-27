#!/usr/bin/python3
"""
File name:  preprocessor.py
Course:     Machine Learning Project
Authors:    Martijn E.N.F.L. Schendstok (s2688174)
            Jannick J.C. Akkermans      (s3429075)
            Niels Westeneng             (s3469735)
Date:       March 2020
"""

from os import listdir # to read files
from os.path import isfile, join # to read files
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import sys
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict

def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))] #Return a list of files in a certain folder

def read_files(categories, author_data, traits):
    """
    This function reads in all the files from a folder. For each file, the function tokenizes the data and lowercases each token.
    The author of the file is derived from the dictionary of authors. For each of the OCEAN scores for this author, the function checks whether
    it exceeds 50. If it does, the personality trait is added to a list. This list, together with the tokenized data, is added to a list of read data
    :param categories: list of folders to process
    :param author_data: Dictionary of authors and their corresponding data
    :return: List with read data, each entry in the form (tokenized_data, personality_traits)
    """
    feats = list()
    print("\n##### Reading files...")
    for category in categories:
        files = get_filenames_in_folder(category)
        num_files = 0
        for f in files:
            personality_traits = []
            try:
                data = open(category + '/' + f, 'r', encoding='UTF-8').read()
                author = author_data[f[0:8]]
                if author[5] != '----':
                    tokens = word_tokenize(data)
                    lower_tokens = [token.lower() for token in tokens]
                    use_tokens = [token for token in lower_tokens if token not in ['.', ',', '?', ':', '(', ')', '!', '\'', '`', '...', '``', '\'\'', '\"']]
                    scores = author[5].split('-')
                    for i in range(len(scores)):
                        if int(scores[i]) >= 50:
                            personality_traits.append(traits[i])
                    feats.append((use_tokens, personality_traits))
                    # print len(tokens)
                    num_files += 1
                else:
                    pass
            # if num_files>=50: # you may want to de-comment this and the next line if you're doing tests (it just loads N documents instead of the whole collection so it runs faster
            #	break
            except UnicodeDecodeError:
                print('Decode error')

            print("  Category %s, %i files read" % (category, num_files))

    print("  Total, %i files read" % (len(feats)))
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


def high_information_words(files, score_fn=BigramAssocMeasures.chi_sq, min_score = 30):
    word_dict = FreqDist()
    ocean_word_dict = ConditionalFreqDist()

    for file in files:
        #For each token, add 1 to the overall FreqDist and 1 to the ConditionalFreqDist under the current personality trait
        for token in file[0]:
            for trait in file[1]:
                ocean_word_dict[trait][token] += 1
            word_dict[token] += 1

    n_xx = ocean_word_dict.N() #Get the total number of recordings in the ConditionalFreqDist
    high_info_words = set()

    for condition in ocean_word_dict.conditions():
        n_xi = ocean_word_dict[condition].N() #Get the number of recordings for each personality trait
        word_scores = defaultdict(int)

        for word, n_ii in ocean_word_dict[condition].items():
            n_ix = word_dict[word] #Get total number of recordings of a token
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)

        return high_info_words

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
    classifier_open.fit(x_feats[:450], label_open[:450])
    classifier_extra.fit(x_feats[:450], label_extra[:450])
    classifier_con.fit(x_feats[:450], label_con[:450])
    classifier_neu.fit(x_feats[:450], label_neu[:450])
    classifier_agree.fit(x_feats[:450], label_agree[:450])
    return classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree

def evaluation(classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree, x_feats, label_open, label_extra, label_con, label_neu, label_agree):
    """
    This function print the accuracy scores of the testing data.
    :param files: all classifiers created by classifier(), x_feats, and all y-labels created by get_fit
    :return: Prints the accuracies
    """

    open_acc = classifier_open.score(x_feats[450:], label_open[450:])
    extra_acc = classifier_extra.score(x_feats[450:], label_extra[450:])
    conc_acc = classifier_con.score(x_feats[450:], label_con[450:])
    neuro_acc = classifier_neu.score(x_feats[450:], label_neu[450:])
    agree_acc = classifier_agree.score(x_feats[450:], label_agree[450:])

    print("Accuracy Openness:", open_acc)
    print("Accuracy Extravertness:", extra_acc)
    print("Accuracy Concientiousness:", conc_acc)
    print("Accuracy Neuroticism:", neuro_acc)
    print("Accuracy Agreeableness:", agree_acc)
    print("Average Accuracy:", round(sum((open_acc, extra_acc, conc_acc, neuro_acc, agree_acc))/5,2))

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    author_data = read_authordata(args[0])
    traits = ['Openness', 'Concientiousness', 'Extravertness', 'Agreeableness', 'Neuroticism']
    files = read_files(args[1:], author_data, traits)
    high_info = high_information_words(files)
    label_open, label_extra, label_con, label_neu, label_agree, feats = get_fit(files, high_info)

    mlb = MultiLabelBinarizer()
    x_feats = mlb.fit_transform(feats)

    classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree = get_classifier(label_open, label_extra, label_con, label_neu, label_agree, x_feats)
    evaluation(classifier_open, classifier_extra, classifier_con, classifier_neu, classifier_agree, x_feats, label_open, label_extra, label_con, label_neu, label_agree)





if __name__ == "__main__":
    main()