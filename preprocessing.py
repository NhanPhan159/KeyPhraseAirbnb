import re
from underthesea import word_tokenize
import urllib
import emoji
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

#Lowercase
def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenize(text):

    return word_tokenize(text, format="text")

# Download vietnamese stop words
data=urllib.request.urlopen('https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords-dash.txt').read()


# Read stop words
stopwords= [ x.decode('utf-8') for x in data.splitlines() ]

def remove_stopwords(text):

    return ' '.join([word for word in text.split() if word not in stopwords])

def remove_html_url(text):

    return re.sub(r'http\S+', '', text)

def remove_emoji(text):
    def get_emoji_regexp():
        # Sort emoji by length to make sure multi-character emojis are
    # matched first
        emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
        pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
        return re.compile(pattern)

    exp = get_emoji_regexp()
    return exp.sub(u'', text)

def preprocess_text(text):
    # '''Input : String, output : String'''
      text = lowercase(text)
      text = remove_punctuation(text)
      text = tokenize(text)
      text = remove_stopwords(text)
      text = remove_html_url(text)
      text = remove_emoji(text)
      return text

def predict(text):
    with open('tokenizer.pickle', 'rb') as handle:
        text_pre_token = [text.split()]
        tokenizer_load = pickle.load(handle)
        text_token = tokenizer_load.texts_to_sequences(text_pre_token)
        text_padded = pad_sequences(text_token,maxlen=50, truncating='post')
        reconstructed_model = load_model("my_model")
        predict = reconstructed_model.predict(text_padded)
        label_key_phrases = np.argmax(predict[0],axis=1, out=None)
        text_rever = tokenizer_load.sequences_to_texts(text_padded)
        text_rever_split = text_rever[0].split()
        key_phrases = [text_rever_split[i] for i in range(len(label_key_phrases)) if label_key_phrases[i] == 1]
        key_phrases = list(filter(lambda a: a != "<OOV>", key_phrases))
        key_phrases_nonToken = [i.replace("_"," ") for i in key_phrases]
        return key_phrases_nonToken