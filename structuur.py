import abc
import re
import spacy
import nltk
import translator
	


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
		
class KeywordCandidateGenerator(CandidateGenerator):
	@classmethod
	def generate_candidates(cls, tokenized_text):
		#generate keywords with one pos tag
	
class TfidfRanker(CandidateRanker):
	@classmethod
	def rank(cls, candidates):
		#tfidf

def main(tweets, translate):
	preprocessors = []
    preprocessors.append(atremoval)
    preprocessors.append(hashtagremoval)
    preprocessors.append(linkremoval)
	candidate_generator = KeywordCandidateGenerator
	tokenizer = SpacyPOStagger
	ranker = TfidfRanker
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
	interest = select(ranked_words)
	print interest
