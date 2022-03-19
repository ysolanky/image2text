import sys
from PIL import Image, ImageDraw, ImageFont
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print(im.size)
    #print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def emission(alpha, beta):

    countm = 1
    countmNot = 1

    # we have assumed the probability of noise to be 0.4. So, whenever the pixels between the training image and the
    # observed image match, we multiply the probability of that observed image being that training image by 1-0.4.
    # Similarly, when we observe that the pixels between the training image and the observed image do not match, we
    # we multiply the probability of that observed image not being that training image by 0.4.

    m = 0.4
    mNot = 0.6

    for a in range(len(alpha)):
        for b in range(len(alpha[a])):
            if alpha[a][b] == beta[a][b]:
                countmNot *= mNot
            else:
                countm *= m
    # We return the product of the two probabilities.
    return countm * countmNot

# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

count_maxp = []
prediction = []

remove = ["ADJ", "NOUN", "DET", "VERB", "CONJ", "ADP", "ADV", "PRON", "NUM",
          "PRT", "--", "``", ":", "$", "&", ";", "*", "+", "[", "]", "/", "{", "}"]
# In the training data from part 1 there were these identifiers and some characters outside of the required 72 for part
# 3. So I remove them from the tokens to consider.
words = []
with open(train_txt_fname, "r") as f:
    for line in f:
        parsed = line.strip().split(' ')
        for i in parsed:
            if i not in remove:
                i = i.strip("/")
                i = i.strip("{")
                i = i.strip("}")
                i = i.strip(":")
                i = i.strip("*")
                i = i.strip("$")
                i = i.strip("%")
                words.append(i)

initial = {}

words = list(set(words))
length = len(words)

# Counting the occurrence of first alphabet/num/punctuation
for i in words:
    if i[0] in initial.keys():
        initial[i[0]] = initial[i[0]] + 1
    else:
        initial[i[0]] = 1

# Converting into initial probability
for value in initial:
    initial[value] = initial[value] / length

initial[" "] = 1 / length # A sentence would never really start with a space but to have all possible keys we assign a small value to it
initial['"'] = 1 / length # A word beginning with a quote was not observed in the training data so we assign it a small value to it.

# counting the transition from one alphabet/symbol to another to calc the transition probability for each letter/ symbol
transition = initial.copy()
for i in transition:
    transition[i] = {}
    for j in words:
        if i in j:
            temp = j.index(i)
            if j[j.index(i)] != j[-1]: # Cases when the alphabet/symbol is not the last in the word
                if j[temp + 1] in transition[i].keys():
                    transition[i][j[temp + 1]] = transition[i][j[temp + 1]] + 1
                else:
                    transition[i][j[temp + 1]] = 1
            else: # Cases when its the last symbol/alphabet in the word.
                if " " in transition[i].keys():
                    transition[i][" "] = transition[i][" "] + 1
                else:
                    transition[i][" "] = 1

transition[" "] = initial.copy() # After a space the probability of the next word is equal to the initial probability.
transition[" "][" "] = 1 / length # A space after space is not observed so we assign a small probability to it.

# Converting count into transition probability
for x, y in transition.items():
    temp_counts = sum(y.values())
    for p, q in y.items():
        transition[x][p] = transition[x][p] / temp_counts

# For the times when we do not get a probability of a particular alphabet/symbol followed by one of the remaining 72
# characters. So, we assign a very small number to such transition probabilities
for i, j in transition.items():
    for a in transition.keys():
        if a not in j.keys():
            transition[i][a] = 1 / length

states = list(initial.keys())

## Simple Bayes net
for letter in range(len(test_letters)):
    counts = []
    prediction = []

    m = 0.6
    mNot = 0.4

    for key, value in train_letters.items():
        countm = 1
        countmNot = 1
        for i in range(len(test_letters[letter])):
            for j in range(len(test_letters[letter][i])):
                if test_letters[letter][i][j] == train_letters[key][i][j]:
                    countm *= m
                else:
                    countmNot *= mNot

        prediction.append((countm * countmNot * initial[key], key)) #list of probability of observed image in comparison to all 72 characters

    count_maxp.append(max(map(lambda x: (x[0], x[1]), prediction)))

## Viterbi
# The structure of the viterbi algorithm was taken from the starter code provided during the viterbi in class activity
# I just made some changes to it to fit into this problem.
N = len(test_letters)
## https://stackoverflow.com/questions/2241891/how-to-initialize-a-dict-with-keys-from-a-list-and-empty-value-in-python
V_table = {i: N * [0] for i in states}
## End of code from stack overflow

for s in states:
    V_table[s][0] = math.log(initial[s]) + math.log(emission(train_letters[s], test_letters[0]))
P_table = {i: N * [0] for i in states}

for i in range(1, N):
    for s in states:
        (P_table[s][i], V_table[s][i]) = max([(s0, V_table[s0][i - 1] + math.log(transition[s0][s])) for s0 in states],
                                             key=lambda l: l[1])
        V_table[s][i] += math.log(emission(train_letters[s], test_letters[i]))
        # As the length of the string to pre predicted increased, so did the multiplications, leading to such a small
        # number that was being interpret as 0. So I had to use log addition instead of multiplication.
        # except KeyError:
        #     P_table[s][i], V_table[s][i] = 1/length,1/length
        #     V_table[s][i] *= emission(train_letters[s], test_letters[i])
viterbi_seq = [""] * N
a = []
for j in states:
    a.append((V_table[j][i], j))
maxi = max(a)
viterbi_seq[N - 1] = maxi[1]

for i in range(N - 2, -1, -1):
    viterbi_seq[i] = P_table[viterbi_seq[i + 1]][i + 1]

# Technique to print string from list taken from
# https://stackoverflow.com/questions/12453580/how-to-concatenate-items-in-a-list-to-a-single-string
result = ""
for i, j in count_maxp:
    result += j
print("Simple: " + result)


print("   HMM: " + "".join(viterbi_seq))
# End of code from stack overflow