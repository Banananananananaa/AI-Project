import abc
import time
import re
import spacy
import nltk
import pandas as pd
import numpy as np
from nltk import wordnet as wn
from collections import Counter
from nltk.corpus import wordnet as wn
import translate
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer

#For all classes, the cls is given as input in the function for the classmethod.
#The preprocessor takes the tweet strings as input, and returns a list of processed words.
class preprocessor():
    @classmethod
    def preprocess(cls, text):
        """removes noise from the tweets"""
   
#A class for translator functions to translate between languages.
#Takes as input a text to translate and outputs the translated text. 
#cls notifier for classmethod.
class Translator():
    @classmethod
    def translate(cls, text):
        """Translates from a language to the desired language if deemed necessary"""

#POS-tagger assigns tags to words.
#Takes list of words as input and gives the tagged words as output .
#cls notifier for classmethod.
class POStagger():
    @classmethod
    def tag(cls, text):
        """Assigns POS-tags to each word in the tweet"""

#Generates candidates from the text.
#Takes as input, a list of tagged words and gives a list of chosen words as output.
#cls notifier for classmethod.
class CandidateGenerator():
    @classmethod
    def generate_candidates(cls, tokenized_text):
        """Generates candidates from the chosen tokens"""

#Assigns ranks to the generated candidates.
#Takes a list of candidates and returns the the words together with a rank.
#cls notifier for classmethod.
class CandidateRanker():
    @classmethod
    def rank(cls, candidates):
        """Ranks the candidates"""

#Generates keywords from the ranked words.
#Takes a list of words with ranks as input and returns the selected keywords in a list as output.
#cls notifier for classmethod.
class KeyWordGenerator():
    @classmethod
    def getkeywords(cls, text):
        """Gets keywords from the ranked words"""

#Selects interests based on the keywords.
#Takes a list of keywords as input and returns a list of interests as output
#cls notifier for classmethod        
class InterestGenerator():
    @classmethod
    def select(cls, keywords):
        """Selects the interests from the keywords"""


#Subclass removes words with an @ at the beginning of the string.
#Takes as input a string of words in a tweet and returns the same.
#Substitutes the strings with an empty string, thus removing them.
#cls notifier for classmethod.
class atremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('@'+'\w+', '', text)
        return text

#Subclass removes words with an # at the beginning of the string.
#Takes as input a string of words in a tweet and returns the same.
#Substitutes the strings with an empty string, thus removing them.
#cls notifier for classmethod.
class hashtagremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('#'+'\w+','', text)
        return text

#Subclass removes words with http at the beginning of the string, designating hyperlinks.
#Takes as input a string of words in a tweet and returns the same.
#Substitutes the strings with an empty string, thus removing them.
#cls notifier for classmethod.
class linkremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('http'+'\S+', '', text)
        return text

#Subclass selects a hyperlink from a tweet and summerizes the location it leads to and returns the summary.
#Takes as input a string of words in a tweet and returns the same.
#cls notifier for classmethod.
class AutomaticSumerization(preprocessor):
    @classmethod
    def preprocess(cls, text):
		link = re.findall('http'+'\S+', text)
        LANGUAGE = "english"
        SENTENCES_COUNT = 10
        url = link
        parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        text = summarizer(parser.document, SENTENCES_COUNT) 
        return text


#Subclass to remove punctuation marks. 
#Takes a tweet as string as input and returns a tweet as a string.
#Finds all strings in the \w+ regular expression. Strings consisting one or more of [a-z,A-z,0-9]
#This removes . and , and other punctuation marks.
#cls notifier for classmethod.
class findall(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.findall('\w+', text)
        text = ' '.join(text)
        text = unicode(text, "utf-8")
        return text


#Subclass that uses the translate package to translate from Dutch to English.
#Takes a list of strings as input and returns a translate list of sttrings.
#Imports the translate package sets the language specifications and uses the package to translate.
#cls notifier for classmethod.
class Translate(Translator):
    @classmethod
    def translate(cls, text):
        from translate import Translator
        translator = Translator(from_lang="nl", to_lang="en")
        text = translator.translate(str(text))
        return text

		
#Subclass tagging words with the spaCy package.
#Takes a string of tweets as input and returns a list of tagged words.
#Sets the Spacy language for the POS-tagger as Dutch. 
#Uses the package to tagg the words and return them.
#cls notifier for classmethod.
class SpacyPOStagger(POStagger):
    @classmethod
    def tag(cls, text):
        nlp = spacy.load('nl')
        text = nlp(text)
        return text

#Subclass tagging words with the nltk package.
#Takes a string of tweets as input and returns a list of tagged words.
#Uses the package to tagg the words and return them.
#cls notifier for classmethod.
class NltkPOStagger(POStagger):
    @classmethod
    def tag(cls, text):
        text = nltk.pos_tag(text)
        return text



#Regular expression candidate generator subclass.
#Takes a list of tagged words as input and returns a list of candidates with the correct tags.
#Selects candidates based on the tag of the word and the next word or two words.
#Selects adjective or a noun followed by a noun.
#Selects verb followed by an adjective or a noun.
#Selects verb followed by an adjective or a noun once more followed by a noun.
#cls notifier for classmethod.
class RegexCandidateGenerator(CandidateGenerator):
    @classmethod
    def generate_candidates(cls, tokenized_text):
        candidates = []
        length = len(tokenized_text)
        for word in range(0, length - 1):
            if (word + 1):
                if tokenized_text[word].tag_ == (u'ADJ' or u'NOUN') and tokenized_text[word + 1].tag_ == u'NOUN':
                    candidates.append(str(tokenized_text[word]), str(tokenized_text[word+1]))
                elif tokenized_text[word].tag_ == u'VERB' and tokenized_text[word + 1].tag_ == (u'ADJ' or u'NOUN'):
                    candidates.append(str(tokenized_text[word]), str(tokenized_text[word+1]))
                elif tokenized_text[word].tag_ == u'VERB' and tokenized_text[word + 1].tag_ == (u'ADJ' or u'NOUN') and tokenized_text[word + 2].tag_ == u'NOUN':
                    candidates.append(str(tokenized_text[word]), str(tokenized_text[word+1]), str(tokenized_text[word+2]) )
        candidates = ' '.join(candidate for candidate in candidates)
        return candidates

#Regular expression candidate generator subclass.
#Takes a list of tagged words as input and returns a list of candidates with the correct tags.
#Selects candidates based on the tag of the word.
#Selects a word tagged with verb or an adjective or a noun.
#cls notifier for classmethod.
class TaggedwordCandidateGenerator(CandidateGenerator):
    @classmethod
    def generate_candidates(cls, tokenized_text):
        candidates = []
        for word in range(0, len(tokenized_text)):
            candidate = tokenized_text[word]
            if candidate.tag_ == (u'NOUN' or u'VERB' or u'ADJ' or u'ADV'):
                candidates.append(str(tokenized_text[word]))
        candidates = ' '.join(candidate for candidate in candidates)
        return candidates


#TF-IDF word ranker subclass
#Takes a list of candidate words as input and returns an array containing the TF-IDF scores.
#For more info visit http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 
class TfidfRanker(CandidateRanker):
    @classmethod
    def rank(cls, candidates):
        vectorizer = TfidfVectorizer(min_df=1)
        ranks = vectorizer.fit_transform(candidates)
        return ranks
    @classmethod
    def feature(cls, candidates):
        vectorizer = TfidfVectorizer(min_df=1)
        ranks = vectorizer.fit_transform(candidates)
        features = vectorizer.get_feature_names()
        return features

#Keyword generator subclass
#Takes an array of TF-IDF scores and returns a list containing the highest ranking word per tweet.
#getkeyword loops over the array to find the index of the highest ranking word in the tweet and returns the corresponding word
class RankedKeywordGenerator(KeyWordGenerator):
    @classmethod
    def getkeywords(cls, ranks, features):
        keywords = []
        for sentence in range(0, ranks.shape[0]):
            k = " "
            w = 0
            for word in range(0, len(features) - 1):
                if ranks[sentence, word] == ranks[sentence].max():
                    w = ranks[sentence, word]
                    k = features[word]
            keywords.append(str(k))
        return keywords

#Select interest subclass
#Takes a list of strings (keywords) and returns a list of strings (interests)
#As this is done with fixed labels, more labels can be inserted by inserting interest synset
#And inserting a new string at the end of list 'name', see also comment inside code 
class Select(InterestGenerator):
    @classmethod
    def select(cls, words):
        keywords = []
        for word in words:
            woord = wn.morphy(word, wn.NOUN)
            if (woord != None):
                keywords.append(woord) 
        labels = []
        labels.append(wn.synset('politics.n.01'))
        labels.append(wn.synset('sport.n.01'))
        labels.append(wn.synset('food.n.01'))
        labels.append(wn.synset('party.n.01'))
        labels.append(wn.synset('education.n.01'))
        labels.append(wn.synset('book.n.01'))
        labels.append(wn.synset('tv.n.01'))
        labels.append(wn.synset('holiday.n.01'))
        labels.append(wn.synset('computer.n.01'))
        labels.append(wn.synset('science.n.01'))
        names = ['politics', 'sports', 'food', 'party', 'education', 'book', 'tv', 'holiday', 'computer', 'science']
        #labels.append(wn.synset('animal.n.01'))
        #names = ['politics', 'sports', 'food', 'party', 'education', 'book', 'tv', 'holiday', 'computer', 'science', 'animal']
        length = len(labels)
        interest = np.zeros(length)
        for keyword in keywords:
            keyword = keyword + '.n.01'
            keyword = wn.synset(keyword)
            for label in labels:
                interest[labels.index(label)] =+ keyword.path_similarity(label)
        classification = []
        #By changing n, one can output their top N interest classes
        n = 3
        for i in range(0,n):
            element = max(xrange(len(interest)), key = lambda x: interest[x])
            classification.append(names[element])
            interest[element] = 0
        return classification


def main(tweets, translate):
    preprocessors = []
    preprocessors.append(atremoval)
    preprocessors.append(hashtagremoval)
    preprocessors.append(linkremoval)
    preprocessors.append(findall)
    candidate_generator = TaggedwordCandidateGenerator
    tokenizer = SpacyPOStagger
    ranker = TfidfRanker
    keywordgenerator = RankedKeywordGenerator
    translator = Translate
    interestgenerator = Select
    all_candidates = []
    for tweet in tweets:
        for preprocessor in preprocessors:
            tweet = preprocessor.preprocess(tweet)
        tags = tokenizer.tag(tweet)
        candidates = candidate_generator.generate_candidates(tags)
        all_candidates.append(candidates)
    print all_candidates
    rankings = ranker.rank(all_candidates)
    features = ranker.feature(all_candidates)
    keywords = keywordgenerator.getkeywords(rankings, features)
    translated_words = translator.translate(keywords)
    interest = interestgenerator.select(translated_words)
    print "Interests are:", interest