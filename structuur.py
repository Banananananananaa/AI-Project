import nltk
import abc
import re
import spacy
import pandas as pd
import re
import numpy as np
from nltk import wordnet as wn
from collections import Counter
from nltk.corpus import wordnet as wn
from translate import Translator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class preprocessor():
    @classmethod
    def preprocess(cls, text):
        """removes noise from the tweets"""
    
class Translation():
	@classmethod
	def translate(cls, language1, language2, text):
		"""Translates from a language to the desired language if deemed necessary"""

class POStagger():
	@classmethod
	def tag(cls, text):
		"""Assigns tags to each word in the tweet"""

class CandidateGenerator():
	@classmethod
	def generate_candidates(cls, tokenized_text):
		"""generates candidates from the tokens as specified"""


class CandidateRanker():
	@classmethod
	def rank(candidates):
		"""rankes the candidates"""

class KeyWordGenerator():
	@classmethod
	def getkeywords(cls, text):
		"""Gets keywords from the ranked words"""
		
		
		
class atremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('@'+'\w+', '', text)
        return text

class hashtagremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('#'+'\w+','', text)
        return text

class linkremoval(preprocessor):
    @classmethod
    def preprocess(cls, text):
        text = re.sub('http'+'\S+', '', text)
        return text			

class AutomaticSumerization(preprocessor):
	@classmethod
	def preprocess(cls, link):
		LANGUAGE = "english"
		SENTENCES_COUNT = 10
		url = link
		parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
		stemmer = Stemmer(LANGUAGE)
		summarizer = Summarizer(stemmer)
		summarizer.stop_words = get_stop_words(LANGUAGE)
		text = summarizer(parser.document, SENTENCES_COUNT) 
		return text
class Translator(translation):
	@classmethod
	def translate(cls, language1, language2, text):
		translator = Translator(from_lang=language1, to_lang=language2)
		return translator.translate(text)		

class SpacyPOStagger(POStagger):
	@classmethod
	def tag(cls, text):
		nlp = spacy.load('nl')
		text = nlp(text)

class NltkPOStagger(POStagger):
	@classmethod
	def tag(cls, text):
		text = nltk.pos_tag(text)
		return text

class RegexCandidateGenerator(CandidateGenerator):
	@classmethod
	def generate_candidates(cls, tokenized_text):
		#generate regex keywords
		
class TaggedwordCandidateGenerator(CandidateGenerator):
	@classmethod
	def generate_candidates(cls, tokenized_text):
		#generate keywords with one pos tag
	
class TfidfRanker(CandidateRanker):
	@classmethod
	def rank(cls, candidates):
		#tfidf

class RankedKeywordGenerator(KeyWordGenerator):
	@classmethod
	def getkeywords(cls, text):
		#code

class Select(keywords):
	@classmethod
	def select(cls, keywords):
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
        length = len(labels)
        interest = np.zeros(length)
        for keyword in keywords:
            keyword = keyword + '.n.01'
            keyword = wn.synset(keyword)
            for label in labels:
                print(word.path_similarity(label))
                interest[labels.index(label)] =+ keyword.path_similarity(label)
        classification = []
        for i in range(0,3):
            element = max(xrange(len(interest)), key = lambda x: interest[x])
            classification.append(names[element])
            interest[element] = 0
        return classification
		
		
def main(tweets, translate):
	preprocessors = []
    preprocessors.append(atremoval)
    preprocessors.append(hashtagremoval)
    preprocessors.append(linkremoval)
	candidate_generator = TaggedwordCandidateGenerator
	tokenizer = SpacyPOStagger
	ranker = TfidfRanker
	keywordgenerator = RankedKeywordGenerator
	all_candidates = set()
	translating = Translator
	
	if translate == TRUE:
		translating.translate(cls, language1, language2 tweets)
	for tweet in tweets:
		for preprocessor in preprocessors:
			tweet = preprocessor.preprocess(tweet)
		candidates = candidate_generator.generate_candidates(tokenizer.tag(tweet))
		all_candidates += candidates
		
	ranked_words = ranker.rank(all_candidates)
	keywords= keywordgenerator.getkeywords(ranked_words)
	interests = select.select(keywords)
	for i in range(len(interests)
		print interests[i]