from os import listdir # to read files
from os.path import isfile, join # to read files
from nltk.tokenize import word_tokenize
import sys
import re

def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))] #Return a list of files in a certain folder

def read_files(categories, author_data):
    """
    This function reads in all the files from a folder. For each file, the function tokenizes the data and lowercases each token.
    The author of the file is derived from the dictionary of authors. For each of the OCEAN scores for this author, the function checks whether
    it exceeds 50. If it does, the personality trait is added to a list. This list, together with the tokenized data, is added to a list of read data
    :param categories: list of folders to process
    :param author_data: Dictionary of authors and their corresponding data
    :return: List with read data, each entry in the form (tokenized_data, personality_traits)
    """
    feats = list()
    traits = ['Openness', 'Concientiousness', 'Extravertness', 'Agreeableness', 'Neuroticism']
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

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    author_data = read_authordata(args[0])
    feats = read_files(args[1:], author_data)
    for feat in feats:
        print('{} {}'.format(feat[0], feat[1]))


if __name__ == "__main__":
    main()