import itertools
import os
import sys
import dill as pickle
import csv
import json
import re
import string
import nltk
import spacy
import preprocessor as p
import emoji
from collections import Counter,defaultdict
from spacy_langdetect import LanguageDetector
from ftfy import fix_text
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict
from twitter_preprocessor import TwitterPreprocessor
p.set_options(p.OPT.EMOJI,p.OPT.RESERVED,p.OPT.SMILEY)
nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
nlp1 = spacy.load('en_core_web_sm')
MIN_TOKENS=10
MAX_TOKENS=70
MAX_SEQ = 500000
all_emoticons = ['(＠_＠;)', '~:>', '=p', ':‑.', '|-)', ':-]', '=P~', 'b-(', '😜', '(^○^)', '(v)o￥o(v)', 'o:-)', '🙆', '(^_-)-☆', ':-\\', '(Ｔ▽Ｔ)', '(*_*)', '(w)', ':o(', '<`ヘ´>', 'ಠ_ಠ', ';(', '(^・^)', '(✿◠‿◠)', ':)', '3:)', '😗', '😞', '(/◕ヮ◕)/', '😔', ':^)', 'x-p', '=-3', '🙊', '_(._.)_', '(^.^)/~~~', ":'-)", '8-0', ':((', '🏂', '=D>', '>:p', ':-"', '(?_?)', '🤓', '(so)', 'o_O', ':)>-', ':‑j', '😽', 't.t', '\\U0001f604', '^_^', ';‑]', '👶', '🙍', ',:-)', ':-L', ':-p', ':!!', '(^o^)／', '[-(', '😱', '(－‸ლ)', 'd=', '(+_+)', 'x-d', '=p~', '#:-S', ':‑|', ':‑þ', '😇', ';^)', '=-D', '😣', ':o3', '(~o~)', '(C)', '😿', "+/'\\", ':-j', '（´∀｀）', '%-(', '8-d', '0:‑3', '(´･ω･`)', '(-""-)', '$-)', '＼(◎o◎)／！', '😥', ';d', ':-þ', '<3', '<^!^>', '(;', '(Z)', '🙅', ':###..', 'Q_Q', '😒', '(._.)', '=:)', 'XP', '(6)', '=:o]', '(＾▽＾)', '=((', '(:_;)', ':‑)', '<(｀^´)>', ':o)', '(I)', '👽', '🎧', 'x-@', '(B)', '(^<^)', '(^^)', ':‑/', '@>‑‑>‑‑', "D-':", 'o/\\o', '=))', '😙', '(E)', '\\U0001f44b', '🍵', '☄️', ':\\', '>:s', '😄', ':-ss', '(>_<)>', ':-??', ';))', '(y)', ';-;', 'O:‑)', '😉', '(N)', '（’-’*)', '😫', ':]', '@}->--', ':っc', '<):)', '😮', '^o)', '(*^^)v', ':-#', ':^*', '>.<', '}:‑)', '(p)', 'X‑P', ':-(', '(゜_゜>)', ':-[', '8)', '(d)', '(o)', ':‑O', '(・・?', '°O°', ':p', '😟', '😹', '(^)o(^)', '%%-', 'V.v.V', 'x‑p', '(-_-メ)', "d-':", '(一一)', '(li)', ':-P', ':o', '(__)', '<`～´>', '|-O', '(*^。^*)', 'ヽ(^o^)丿', '😤', '\\u0001f3fb', '(ToT)', 'o_o', ':X', ':|', '(8)', '(^ム^)', ':-*', '=/', 'DX', '(-_-)zzz', '~X(', '\\U0001f1f3', 'XD', '8-X', '#-o', "(';')", '(=^・・^=)', '(ーー゛)', ';n;', ':}', '😀', 'O_o', '(&)', ':P', '😓', '🙌', ':‑J', ':っ)', '🙉', '3:-o', '[-o<', '(^_^メ)', ':‑X', 'xp', '\\U0001f64f', '🤔', '(sn)', '\\u0001f620', '🙀', '8‑d', 'd:', '(^_^)/~', '+o(', '|‑o', ':-))', '∩(・ω・)∩', ':-&', ':‑o', '(||)', ':{', '(*°∀°)=3', '>_>^', '\\:D/', '(~_~;)', '\\u0001f602', '8o|', '(})', ':<', '🐍', '(ー_ー)!!', '>:-)', '\\u0001f64f', '💯', '👋', 'l-)', '(~_~メ)', '\\u0001f3c2', '(um)', "@}‑;‑'‑‑‑", '(:', '°o°', 'm(__)m', '☺️', '(-.-)', '😳', ':‑#', '(-_-)', ':))', '8‑D', '~:‑\\', '😖', '><>', '3:‑)', '>:D<', '(^J^)', '🔘', '</3', ':-x', '(^o^)丿', '＼(-o-)／', '(゜.゜)', '(^0_0^)', '><(((*>', ':‑[', 'D8', '>:S', '~o)', '(co)', '😡', '//0-0\\\\', "('_')", '(mp)', '<:-|', '(=_=)', 'x_x', '}:-)', '(g)', '//0‑0\\\\', ';-D', '(^_^.)', '=3', ':-.', '(^^;)', '(‘a`)', ':-@', '😲', 'q.q', ')^o^(', '😦', '\\U0001f3c2', '8-)', '🐠', '(;_;)/~~~', '😯', '🦇', '(;_:)', '8-D', '^#(^', '(t_t)/~~~', '(;O;)', '-D', ':O', '😐', ':@', ':-0', '<(o0o)>', '((d[-_-]b))', '(%)', '*-)', '>:[', '(^.^)', ':-###..', '😪', '0:‑)', ':&', '☺', '(・_・;)', '(tot)', '(ip)', ':(', '😃', '(au)', 'o_0', '\\u0001f61c', '\\U0001f602', 'B^D', ':-Þ', '@};-', ':^(', '😠', '(^_^)v', ':d', '（⌒▽⌒）', '(゜レ゜)', ':‑P', '(ToT)/~~~', ':*', '😁', ':bz', '(bah)', 'o.O', ';-d', '%)', ":'-(", ':っC', '😈', 'o.o', '(>_<)', '(K)', '🇳', '🤑', '🐱', '~x(', '(b)', '🤦', '(・。・;)', '=]', '#-)', 'O:)', '({)', ':)]', '（・∀・）', '🦈', '~(_8^(I)', '😵', '(pi)', '>:‑)', '(T_T)', 'o->', '=\\', '(;o;)', 'o-o', '(*^0^*)', 'L-)', '=;', '☹️', '(O)', ':‑p', ';]', '<\\3', '🐟', ':!', '(x)', '🙈', ':#', '😴', '8d', ':>', '(t_t)', '（＾－＾）', '🇮', ':-q', '[:|]', '🖐', '(^)', ':->', '(P)', '>:\\', '(a)', ';D', '7:^]', 'v.v.v', '\\U0001f590', ':~)', "D‑':", '^/^', '(st)', '(-:', '(s)', 'b-)', '😨', '|;-)', 'i-)', ':c)', ':O)', '🤕', '(Y)', 'x-D', '😅', '😼', '%‑)', ':-3', ';-)', ':‑&', '\\U0001f620', '(T_T)/~~~', '😊', '(・ω・)', ':b', '（＾◇＾）', '\\U0001f518', ':-s', '(D)', '0:3', ':S', '(U)', '=_^=', ':l', '(o.o)', ":'‑(", 'I-)', '(^_-)', '*-:)', '\\U0001f4af', '\\o/', '😬', 'o=>', 'd:<', 'ヽ(^。^)ノ', '(#^.^#)', '($・・)/~~~', '((+_+))', ':-$', ':^o', '(M)', ':-?', 'o:)', '🙄', '@-)', 'qq', '=d>', '(^^ゞ', '8‑0', '.....φ(・∀・＊)', '=L', '>:{', ':[', ':/', '(W)', ':-,', '(^^)/~~~', '😺', ';P', '~O)', 'D:', '🙂', '\\u0001f60a', ':"(', '(c)', ':-S', '😌', '(f)', ':×', '(A)', '(z)', '(^。^)', '🐳', '^ω^', '😩', ':ar!', ':þ', '💤', '(@_@)', '(^_^)/', '>_>^^<_<', '\\u0001f4af', '😻', 'd8', '(X)', '<(__)>', '|‑O', '😾', ':@)', '(S)', '[..]', '>:(', ':-t', '😷', '8->', '(^^)v', '_(_^_)_', '(l)', '**==', ':,(', ':‑Þ', '(F)', '0;^)', 'xd', '@>-->--', ':-h', '*\\0/*', ':‑###..', '^<_<', '(:|', '(t)', ':L', '=P', '>^_^<', '（*^_^*）', '(*^▽^*)', '🙁', '😶', ':‑<', '\\:d/', '\\U0001f1ee', '#:-s', '(*)', ':-)', '(-_-;)', ';-]', '<*)))‑{', '*<|:-)', '(k)', '（￣□￣；）', 'q_q', '5:‑)', '>:/', '💣', '\\u0001f1f3', '(^j^)', '(T)', '(n)', ':-X', 'v.v', '0:-3', '(｀´）', '（*´▽｀*）', '(o|o)', '(゜_゜)', ':-O', '👾', '🐋', '🙃', '<m(__)m>', '<:-p', ':‑x', '\\u0001f590', '(@^^)/~~~', '=(', '0:)', ':‑c', '~(_8^(i)', '＼(~o~)／', 'xD', '😢', '(^_^;)', '^:)^', ',:‑)', '^_^;', ':x', 'd;', ':$', ':‑(', ':s', ';‑)', 'X_X', '\\u0001f518', 'x(', '[-x', 'x‑d', '^5', '(u)', '🙏', '(H)', '\\U0001f61c', '😝', ':\\*', '(h)', '😏', '8-x', 'D=', ':‑d', ':-l', '^^;', '|-o', 'B-)', ':Þ', '-d', '(G)', ':">', '😸', ';p', '🏻', ':-d', '|;‑)', '\\u0001f1ee', 'b^d', 'X-P', '<:o)', '😰', '\\U0001f31f', '=l', 'o‑o', '😆', '*<|:‑)', '(゜-゜)', '🤐', '\\u0001f31f', '*)', 'X-D', '0:-)', '>:)', 'dx', '\\u0001f604', '🙎', '☕️', ":'(", ':(:)', '😋', '☹', ':-<', '(‘A`)', '(;_;', "@}-;-'---", 'o-+', '>;)', ':‑,', ':-w', '}:)', '＼(^o^)／', ':-||', '=D', "d‑':", 'x‑D', '>:d<', ':‑b', '😘', '🤢', ';)', '(m)', '>:o', '(L)', '🙇', '(pl)', '🤒', '3:-)', '(；一_一)', 'QQ', '\\m/', '😭', '#‑)', '\\u0001f44b', '(e)', '=d', ':-b', ":')", '(ap)', '<:-P', '👼', ':(|)', '(;_;)', '=-d', '🚬', '~:-\\', '>:O', '🌟', ':-/', '§^。^§', '^m^', 'T.T', 'O-O', '(*^3^)/~☆', '3:-O', '😛', ':c', '=)', '\\U0001f60a', ':-c', '😚', '🙋', '(@)', '(~)', '8D', '(ｔ▽ｔ)', ';_;', '（￣ー￣）', '\\U0001f3fb', '😑', 'D;', '(・・;)', ':-bd', '>-)', '😂', '~@~', 'X‑D', '😍', '/:)', '>:', '>:P', '(^_^)', '(・へ・)', '<*)))-{', '(^O^)／', '😕', '[-X', '🐬', ':-SS', '（●＾o＾●）', '（＾ｕ＾）', 'o:‑)', '(@_@。', '(ーー;)', '💃', 'X(', ':-o', '(゜o゜)', '(*￣m￣)', ':-D', '(*_*;', ':D', '😎', '(^^)/', ":'‑)", 'O:-)', '（＾ｖ＾）', '🐡', '(≧∇≦)/', '🐙', '😧', ':3', '(p_-)', ':-B', '(-;', '(..)', ':-}', ';o)', '(+o+)', '(~~)', ':-|', '(V)o￥o(V)', '<:‑|', 'Q.Q', ';;)', '(^o^)', '5:-)', '8-}', '(mo)', '(i)', '(/_;)', 'x-(', '☻', ':‑D', 'O_O', '[-O<', '(－－〆)', '%-)', '!(^^)!', '(゜゜)', '(tot)/~~~', '(=^・^=)', ':-J', 'D:<', '✌️', '(^O^)', '8-|', '(~_~)']

# Takes tokenised tweet as input argument and return the list of emoticons
# present in the tweet.
def GetEmoticons(tweet):
	emoticons=[]
	for emoticon in all_emoticons:
		if emoticon in tweet:
			emoticons.append(emoticon)
	return emoticons

def give_emoji_free_text(text):
	return emoji.get_emoji_regexp().sub(u'', text)

def remove_emoticon(tweet):
	tweet = p.clean(tweet)
	'''delims = GetEmoticons(BaseTokeniser(tweet))
	for delim in delims:
		tweet = tweet.replace(delim,"")'''
	tweet = give_emoji_free_text(tweet)
	return tweet

def remove_tags(doc):
	doc = ' '.join(word for word in doc.split() if word[0]!='<')
	return doc

def remove_slang(sent):
	return [slangdict[w.strip()] if w.strip() in slangdict else w for w in sent]

# Takes tokenised tweet as input argument and return the list of hash tags
# present in the tweet.
def GetHashTags(tweet):
	hashtags = []
	for token in tweet:
		if token[0]=='#':
			hashtags.append(token)
	return hashtags

# Takes tokenised tweet as input argument and return the list of usernames
# present in the tweet.
def GetUserNames(tweet):
	user_names = []
	for token in tweet:
		if token[0]=='@':
			user_names.append(token)
	return user_names

# Takes tweet as input argument and return the list of URL's present
# in the tweet.
def GetURLs(tweet,lowercase=False):
	url_regex = [r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+']
	url_re = re.compile(r'('+'|'.join(url_regex)+')', re.VERBOSE | re.IGNORECASE)
	urls = url_re.findall(tweet)
	if lowercase:
		for x in range(len(urls)):
			urls[x] = urls[x].lower()
	return urls

# Remove punctuation marks from a preprocessed tweet.
def RemovePunctuations(processed_tweet):
	punctuations_marks = string.punctuation
	punctuations_marks = punctuations_marks+'¿'+"।"+"|"+'—'+'・'+'…'
	processed_tweet = processed_tweet.translate(str.maketrans('','',punctuations_marks))
	return processed_tweet

# Split the tweet into tokens.
def BaseTokeniser(tweet,lowercase=False):
	if lowercase:
		tweet = tweet.lower()
	tokenised_tweet = tweet.split(' ')
	tokenised_tweet = list(filter(None, tokenised_tweet))
	return tokenised_tweet


def ReplaceTwoOrMore(word):
	return re.sub(r'(.)\1+', r'\1\1', word)

def RemoveRepetitions(tweet):
	processed_tweet = tweet
	for i in range(len(processed_tweet)):
		word = processed_tweet[i]
		if processed_tweet[i] != ReplaceTwoOrMore(word):
			processed_tweet[i] = ReplaceTwoOrMore(word)
	return processed_tweet


def removeRepeat(string):
	return re.sub(r'(.)\1+', r'\1\1', string)

def RemoveNumbers(tweet):
	return re.sub(r'\d+', '',tweet)

def CustomPreprocessor(tweet,lowercase=False):
	tokenised_tweet = BaseTokeniser(tweet,lowercase)
	emoticons = GetEmoticons(tokenised_tweet)
	hashtags = GetHashTags(tokenised_tweet)
	usernames = GetUserNames(tokenised_tweet)
	urls = GetURLs(tweet,lowercase)
	delims=emoticons+hashtags+usernames+urls
	processed_tweet = ' '.join(tokenised_tweet)
	for delim in delims:
		processed_tweet = processed_tweet.replace(delim,"")
	processed_tweet = RemovePunctuations(processed_tweet)
	processed_tweet = RemoveNumbers(processed_tweet)
	processed_tweet_tokenised = BaseTokeniser(processed_tweet)
	processed_tweet= RemoveRepetitions(processed_tweet_tokenised)
	return ' '.join(processed_tweet).strip()


def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

def write_in_file(filename,data):
	print("Writing content in file....")
	fd = open(filename,"w",encoding="utf-8")
	fd.write("\n".join(data))
	fd.close()
	print("Done!!")


def is_english(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

def check_en_lang(text):
	doc = nlp(text)
	flag=True
	if doc._.language["language"]!="en":
		flag=False
	else:
		pass
		#for token in doc:
			#if token._.language["language"]!="en":
				#flag=False
				#break
	return flag

def get_sentences(data):
	##data :- list of text (paragraph)
	temp=[]
	for text in data:
		text_sentences = nlp1(text)
		for sentence in text_sentences.sents:
			temp.append(sentence.text)
	return temp

def TextPreprocessing(data,text_processor,sentence_segmentation=False,pre_lang_check=True,mode=0):
	global MIN_TOKENS
	sents = [fix_text(sent) for sent in data]
	if pre_lang_check:
		sents = [sent for sent in sents if is_english(sent)]
	if sentence_segmentation:
		sents = get_sentences(sents)
	sents = [remove_emoticon(" ".join(remove_slang(sent.split()))) for sent in sents]
	sents = [TwitterPreprocessor(" ".join(sent)).remove_blank_spaces().text for sent in list(text_processor.pre_process_docs(sents))]
	sents = [sent.replace("<censored>","").replace("<emphasis>","").replace("<elongated>","").replace("<repeated>","") for sent in sents]
	if mode:
		MIN_TOKENS*=2
	sents = []
	for sent in sents:
		str_=''
		sent = RemovePunctuations(sent)
		for word in sent.split():
			word = word.strip()
			if word:
				if len(list(word))==1:
					if word in ('a','e','i','o','u'):
						str_+=word+" "
				else:
					str_+=word+" "
		sent=str_.strip()
		if len(sent.split())>=MIN_TOKENS and len(sent.split())<=MAX_TOKENS:
			sents.append(sent)
	sents = [sent for sent in sents if check_en_lang(sent)]
	return sents

def get_mode(data_type):
	mode=0
	if data_type=="post":
		mode=1
	return mode

def get_preprocessed_data(raw_data,sentence_segmentation=False,pre_lang_check=True):
	text_processor = TextPreProcessor(
	omit=['url', 'email', 'user'],
	normalize=['url', 'email', 'user'],
	annotate={"elongated", "repeated",'emphasis', 'censored'},
	segmenter="twitter",
	corrector="twitter",
	unpack_hashtags=True,
	unpack_contractions=True,
	spell_correct_elong=False,
	spell_correction=False,
	tokenizer=SocialTokenizer(lowercase=True).tokenize)
	processed_data=[]
	for data_type in raw_data:
		if len(raw_data[data_type])>=MAX_SEQ:
			chunks = int(len(raw_data[data_type])/MAX_SEQ)
			for j in range(chunks):
				processed_data+=TextPreprocessing(raw_data[data_type][MAX_SEQ*j:MAX_SEQ*(j+1)],text_processor,sentence_segmentation,pre_lang_check,get_mode(data_type))
			if MAX_SEQ*chunks!=len(raw_data[data_type]):
				processed_data+=TextPreprocessing(raw_data[data_type][MAX_SEQ*chunks:],text_processor,sentence_segmentation,pre_lang_check,get_mode(data_type))
		else:
			processed_data+=TextPreprocessing(raw_data[data_type],text_processor,sentence_segmentation,pre_lang_check,get_mode(data_type))
	return processed_data

DATA_TYPE="tw"
if DATA_TYPE in ("tw","insta","fb"):
	DATA_PATH = "../data/raw_data/"+DATA_TYPE

data=defaultdict(list)
comments=False
for file in os.listdir(DATA_PATH):
	FILE_PATH=os.path.join(DATA_PATH,file)
	if DATA_TYPE=="tw":
		if file=="tweets.txt":
			for sent in open(FILE_PATH,"r").read().split("\n"):
				sent = sent.strip()
				if sent:
					temp  = ",".join(sent.split(",")[1:])
					if temp not in data['comment']:
						data['comment'].append(temp)
	elif DATA_TYPE=="fb":
		if file=="fb_posts" or file=="fb_comments":
			for sent in open(FILE_PATH,"r").read().split("--------------------"):
				sent = sent.strip()
				if sent:
					temp = ",".join(sent.strip().split(",")[1:]).replace("\n","")
					if 'post' in file:
						if temp not in data['post']:
							data['post'].append(temp)
					else:
						if temp not in data['comment']:
							data['comment'].append(temp)
	else:
		if file=="insta_captions" or file=="insta_comments":
			for sent in open(FILE_PATH,"r").read().split("--------------------"):
				sent=sent.strip()
				if sent:
					temp = ":".join(sent.strip().split(":")[1:]).replace("\n","")
					if 'caption' in file:
						if temp not in data['post']:
							data['post'].append(temp)
					else:
						if temp not in data['comment']:
							data['comment'].append(temp)

print("Total posts/captions in raw_data: ",len(data['post']))
print("Total comments in raw_data: ",len(data['comment']))
print("Total sequences in raw_data: ", len(data['post'])+len(data['comment']))
DATA_PATH = "../data/preprocessed_data/"+DATA_TYPE
if not os.path.exists(DATA_PATH):
	os.makedirs(DATA_PATH)

data = get_preprocessed_data(data)
preprocessed_filename=DATA_TYPE+"_data"
write_in_file(os.path.join(DATA_PATH,preprocessed_filename),data)
