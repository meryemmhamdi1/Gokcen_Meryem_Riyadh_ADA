import re
import io
import nltk
from nltk.tokenize import RegexpTokenizer
import nltk.tag.stanford as st
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def replace_contractions(emails):
    f = io.open('IntermediateData/contractions.txt', 'r', encoding='utf8')
    text = f.read()
    contractions = eval(text)
    keys = list(contractions.keys())
    values = list(contractions.values())
    for i in range(0, len(contractions)):
        emails = emails.replace({keys[i]: values[i]}, regex=True)
    return emails

# Replace different versions of the same country in the text 
def term_normalization(emails):
    emails = emails.replace({"U.S.": "US"}, regex=True)
    emails_normalized = emails.replace({"DEPART.": "DEPARTMENT"}, regex=True)    
    return emails_normalized

# Tokenization of the emails based on column RawText
def bag_of_word_representation(emails):
    """
    Tokenization, UTF-8 decoding and Removal of white spaces
    :param emails:
    :return:
    """
    emails_bag_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    for email in emails['ExtractedBodyText']:
        # Tokenization and convert to lower case
        emails_filtered = [t.lower() for t in tokenizer.tokenize(str(email))]
        emails_bag_words.append(emails_filtered)
    return emails_bag_words

# Make flat list of words for emails 
def flat_bag_of_word_representation(emails_bag_words):
    email_bag_flat = [word for sublist in emails_bag_words for word in sublist]
    return email_bag_flat

# Make flat list of words for tagged emails 
def flat_tag_bag_of_word_representation(tagged_emails_bag_words):
    email_bag_flat = [word for sublist in tagged_emails_bag_words for (word,tag) in sublist]
    return email_bag_flat

def pos_tagging(emails_bag_words):
    """
    POS tagging of email bag of words using universal tagset
    :param email_bag_words:
    :return:
    """
    tagged_emails = []
    for i in range(0, len(emails_bag_words)):
        tagged_emails.append(nltk.pos_tag(emails_bag_words[i]))
    return tagged_emails

# Add More stopwords (junk words) to this list Later on
def punctuation_numbers_stopword_removal(tagged_emails_bag_words):
    tagged_emails_without = []
    stop_words = set(stopwords.words('english'))
    noisy_words = ['say','unclassified','call', 'know', 'would', 'get', 'time', 'work', 'like', 'today', 'see', 'morning', 'also', 'back', 'tomorrow', 'meeting', 'think', 'good', 'want', 'could', 'working', 'well', 'fyi','fw','make','go','case','doc','clintonemail','original','part','new','unclassified','no','f','date','state','to','from','sent','am','pm','subject','mailto','fw','send','message','call', 'know', 'would', 'get', 'time', 'work', 'like', 'today', 'see', 'morning', 'also','say','may','would','need','year','one', 'back', 'tomorrow', 'meeting', 'think', 'good', 'want', 'could', 'working', 'well']
    for i in range(0, len(tagged_emails_bag_words)):
        tagged_emails_without_sub = []
        for (word, tag) in tagged_emails_bag_words[i]:
            if word not in stop_words and word not in noisy_words and len(word) >= 2 and word.isdigit()==False:
                tagged_emails_without_sub.append((word, tag))
        tagged_emails_without.append(tagged_emails_without_sub)        
    return tagged_emails_without

def normalize_nava_tags(tagged_emails):
    """
    Normalized tags and keeps only nouns, verbs, adjectives and adverbs
    :param tagged_emails:
    :return: normalized_tagged_emails
    """
    normalized_tagged_emails = []
    for i in range(0, len(tagged_emails)):
        normalized_tagged_emails_sub = []
        for (word, tag) in tagged_emails[i]:
            if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS':
                normalized_tagged_emails_sub.append((word, 'n'))
            elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                normalized_tagged_emails_sub.append((word, 'v'))
            elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                normalized_tagged_emails_sub.append((word, 'adj'))
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                normalized_tagged_emails_sub.append((word, 'adv'))
        normalized_tagged_emails.append(list(normalized_tagged_emails_sub))
    return normalized_tagged_emails


def lemmatizer(normalized_tagged_emails):
    lemma_emails_whole = []
    lmtzr = WordNetLemmatizer()
    for i in range(0,len(normalized_tagged_emails)):
        lemma_emails_sub = []
        for (word,tag) in normalized_tagged_emails[i]:
            if tag=='v' or tag =='n':
                lemma_emails_sub.append((lmtzr.lemmatize(word,tag),tag))
            else: 
                lemma_emails_sub.append((word,tag))
        lemma_emails_whole.append(lemma_emails_sub)
    return lemma_emails_whole

# Remove POS tags from the list of emails
def untag_lemma_emails(tagged_emails_without_normalized_lemma):
    """
    Removal of POS tags from the list of emails
    :param tagged_emails_without_normalized_lemma:
    :return:
    """
    emails_lemma_untagged = []
    for email in tagged_emails_without_normalized_lemma:
        emails_lemma_untagged_sub = []
        for (word,tag) in email:
            emails_lemma_untagged_sub.append(word)
        emails_lemma_untagged.append(emails_lemma_untagged_sub)
    return emails_lemma_untagged
