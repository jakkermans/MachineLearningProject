#!/usr/bin/python3
"""
File name:  n-gram_plotter.py
Course:     Machine Learning Project
Authors:    Martijn E.N.F.L. Schendstok (s2688174)
            Jannick J.C. Akkermans      (s3429075)
            Niels Westeneng             (s3469735)
Date:       March 2020
"""

import matplotlib.pyplot as plt
from preprocessor import *


def main():
    args = []
    accs = []
    indices = []
    for arg in sys.argv[1:]:
        args.append(arg)

    author_data = read_authordata(args[0])
    traits = ['Openness',
              'Concientiousness',
              'Extravertness',
              'Agreeableness',
              'Neuroticism']

    for i in range(1, 20):
        print("\n")
        print("#### Test with {}-gram".format(i), end="")
        indices.append(i)

        files = read_files(args[1:],
                           author_data,
                           traits,
                           n_grams=i)

        high_info, hiw_categories = high_information_words(files,
                                                           min_score=15)
        label_open, label_extra, label_con, label_neu, label_agree, feats = get_fit(files,
                                                                                    high_info)

        avg_acc = n_cross_validation(10,
                                     label_open,
                                     label_extra,
                                     label_con,
                                     label_neu,
                                     label_agree,
                                     feats)
        accs.append(avg_acc)

    #get_high_information_words(hiw_categories)
    plt.plot(indices, accs)
    plt.xlabel('n-gram')
    plt.ylabel('Average accuracy')
    plt.title('Average accuracy for different n-grams')
    plt.show()
    plt.savefig('acc_n-gram.png')


if __name__ == "__main__":
    main()
