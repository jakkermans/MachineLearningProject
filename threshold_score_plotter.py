#!/usr/bin/python3
"""
File name:  threshold_score_plotter.py
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
    files = read_files(args[1:],
                       author_data,
                       traits)

    for i in range(5, 101):
        print("\n")
        print("#### Test with threshold score {}".format(i), end="")
        indices.append(i)
        high_info, hiw_categories = high_information_words(files,
                                                           min_score=i)
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
    plt.xlabel('Threshold score')
    plt.ylabel('Average accuracy')
    plt.title('Average accuracy for different threshold scores')
    plt.show()
    plt.savefig('acc_threshold.png')


if __name__ == "__main__":
    main()
