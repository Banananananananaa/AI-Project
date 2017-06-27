import abc
import re

class PreProcessor(metaclass=ABCMeta):

	@abstractmethod
	def preprocess(text):
		text = re.sub('@'+'\w+','', text)
		text = re.sub('#'+'\w', '', text)
		text = re.sub('http'+'\S+', '', text)
		text = re.findall('\w+', text)
		text = ' '.join(text)
		text = unicode(text, "utf-8")
		return text

class POStagger(metaclass=ABCMeta):
	
	@abstractmethod
	def tag(text):
		pass

	@abstractmethod
	def get_language():
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
