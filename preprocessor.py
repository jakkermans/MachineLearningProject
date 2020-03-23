from os import listdir # to read files
from os.path import isfile, join # to read files
from nltk.tokenize import word_tokenize
import sys

def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def read_files(categories):
    feats = list()
    print("\n##### Reading files...")
    for category in categories:
        files = get_filenames_in_folder(category)
        num_files = 0
        for f in files:
            try:
                data = open(category + '/' + f, 'r', encoding='UTF-8').read()

                tokens = word_tokenize(data)
                lower_tokens = [token.lower() for token in tokens]
                feats.append((lower_tokens, category))
                # print len(tokens)
                num_files += 1
            # if num_files>=50: # you may want to de-comment this and the next line if you're doing tests (it just loads N documents instead of the whole collection so it runs faster
            #	break
            except UnicodeDecodeError:
                print('Decode error')

            print("  Category %s, %i files read" % (category, num_files))

    print("  Total, %i files read" % (len(feats)))
    return feats

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    feats = read_files(args)
    print(feats)


if __name__ == "__main__":
    main()