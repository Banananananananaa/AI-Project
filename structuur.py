import abc
import re
import enchant
import spacy
import pandas as pd
import re
import nltk
from nltk import wordnet as wn
from collections import Counter
from nltk.corpus import wordnet as wn
from translate import Translator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class PreProcessor(metaclass=ABCMeta):

	@abstractmethod
	def preprocess(text, translate):
		language = ''
		if translate == True:
			text = enchant.Dict("en_US")
			language = 'en'
		else:
			language = 'nl'
		text = re.findall('\w+', text)
		text = ' '.join(text)
		text = unicode(text, "utf-8")
		return tuple(language, text)
		
	def atremoval(text):
		text = re.sub('@'+'\w+','', text)
		return text

	def hastagremoval(text):
		text = re.sub('#'+'\w+','', text)
		return text	
		
	def linkremoval(text):
		text = re.sub('http'+'\S+', '', text)
		return text
		
class POStagger(metaclass=ABCMeta):
	
	@abstractmethod
	def tag(text):
		pass

	@abstractmethod
	def get_language(text):
		pass

class CandidateGenerator(metaclass=ABCMeta):

	@abstractmethod
	def generate_candidates(tokenized_text):
		pass


class CandidateRanker(metaclass=ABCMeta):
	
	@abstractmethod
	def rank(candidates):
		pass

class SpacyPOStagger(POStagger):
	
	def __init__(self, spacy):
		self.spacy = spacy

	def tag(text):
		return self.spacy.tag(text)


class WordnetPOStagger(POStagger):
	
	def __init__(self, wordnet):
		self.wordnet = wordnet

	def tag(text):
		return self.spacy.tag(text)

class POStagCandidateGenerator(CandidateGenerator):
	def __init__(self, pos_tagger):
		do stuff..

class KeywordCandidateGenerator(CandidateGenerator):
	## simple keyword candidate generator

def main(tweets):
	preprocessors = [remoteGarbishPreprocessor(..), .. ,...]
	
	candidate_generator = ....
	tokenizer = ... twokenizer()
	all_candidates = set()
	for tweet in tweets:
		for preprocessor in preproccors:
			preprocessed_tweet = preprocessor.preprocess(tweet)
			candidates = candidate_generator.generate(tokenizer.tokenize(preprocessed_tweet))
			all_candidates += candidates
	
	ranker = ..
	
	keywords = ranker.rank(all_candidates)
	print keywords

#-----------------------------------------------------------------------------------------------

# takes hyperlink, outputs text
def automatic_summarizer(link):
    LANGUAGE = "english"
    SENTENCES_COUNT = 10
    url = link
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    text = summarizer(parser.document, SENTENCES_COUNT) 
    return text

# example usage
text = automatic_summarizer("https://en.wikipedia.org/wiki/Tf%E2%80%93idf")
for sentence in text:
    print(sentence)