import abc
import re
import enchant

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
