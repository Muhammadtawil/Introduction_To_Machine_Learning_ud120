import string
from nltk.stem.snowball import SnowballStemmer

def parseOutText(f):
    '''
    Input: a file containing text
    
    Output: the stemmed words in the input text, all separated by a single space
    '''
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    
    # the stemmer
    stemmer = SnowballStemmer('english')
    
    # the string of words
    words = ""
    
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        for word in text_string.split():
            # stem the word and add it to words
            words += stemmer.stem(word) + ' '       
        
    return words[:-1]
    

ff = open("test_email.txt", "r")
text = parseOutText(ff)
print(text)