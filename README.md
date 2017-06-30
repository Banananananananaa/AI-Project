# AI-Project
Interest Classifier is a Python implementation for retrieving user interests from Twitter messages. 
Needed by the program is a list of Twitter messages, the output given is the top three interest classes for given user input.


- __Preprocessor__

The preprocessor takes the Twitter messages as input in the form of a string and returns a list of processed words, without noise such as punctuation smileys etc. 
```
class Preprocessor()
	Remove all occurrences of '@' in a tweet using regular expression comparison
_	class atremoval(Preprocessor)

	Remove all occurrences of '#' in a tweet using regular expression comparison
	class hashtagremoval(Preprocessor)

	Remove all hyperlinks in a tweet using regular expression comparison
	class linkremoval(Preprocessor)

	If available, return a summerization of the hyperlink containing the most significant keywords of the hyperlink
	class AutomaticSumerization(Preprocessor)

	Remove all other punctuation marks and return a list containing only strings of words
	findall(Preprocessor)
```


- __Translator__

The translator takes as input a text and outputs the translated text.
```
class Translator()
	Use the translator provided in the translate package
	class Translate(Translator)
```


- __POS-Tagger__

The part of speech-tagger assigns tags to words.
Takes a list of words as input and gives the corresponding tagged words as output.
```
class POStagger()
	Use the spaCy natural language processor to tag the words (dutch)
	class SpacyPOStagger(POStagger)

	Use the NLTK natural language toolkit to tag the words (english)
	class NltkPOStagger(POStagger)
```



- __Candidate Generator__

The candidate generator generates candidates from text.
Takes as input a list of tagged words and gives a list of chosen words(nouns, verb, adjectives and adverbs) as output.
```
class CandidateGenerator()
	Get the words with the tag noun, verb, adjective or adverb from tagged words
	class TaggedwordCandidateGenerator(CandidateGenerator):

	Get the words with the tag noun, verb, adjective or adverb from tagged words and select according tag bi- and tri-grams of candidate words.
	class RegexCandidateGenerator(CandidateGenerator)
```


- __Candidate Ranker__

The candidate ranker assigns ranks to the generated candidates.
Takes a list of candidates and returns the words together with a rank.
```
class CandidateRanker()
	Calculate TF-IDF score for candidate words using the TfidfVectorizer provided by scikit-learn
	class TfidfRanker(CandidateRanker)
```


- __Keyword Generator__

The keyword generator generates keywords from the ranked words.
Takes a list of words with ranks as input and returns the selected keywords in a list as output.
```
class KeyWordGenerator()
	Select highest scoring word from tweet based on TF-IDF scores
	class RankedKeywordGenerator(KeyWordGenerator)
```


- __Interest Generator__

The interest generator selects interests based on the keywords.
Takes a list of keywords as input and returns a list of interests as output
```
class InterestGenerator()
	Based on word similarity between word and class, pick the class with highest word similarity
	class Select(InterestGenerator)
```


- __Dependencies__

For correct working of the program the following packages need to be installed/imported:

- abc
- re
- spaCy
- from NLTK WordNet
- Pandas
- Numpy
- Translate
- from scikit-learn TfidfVectorizer
