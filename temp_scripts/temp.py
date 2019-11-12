from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.utils.nlp import polarity
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict
from ftfy import fix_text
import preprocessor as p
import emoji
import demoji
demoji.download_codes()
from twitter_preprocessor import TwitterPreprocessor
p.set_options(p.OPT.EMOJI,p.OPT.RESERVED,p.OPT.SMILEY)
all_emoticons = ['(＠_＠;)', '~:>', '=p', ':‑.', '|-)', ':-]', '=P~', 'b-(', '😜', '(^○^)', '(v)o￥o(v)', 'o:-)', '🙆', '(^_-)-☆', ':-\\', '(Ｔ▽Ｔ)', '(*_*)', '(w)', ':o(', '<`ヘ´>', 'ಠ_ಠ', ';(', '(^・^)', '(✿◠‿◠)', ':)', '3:)', '😗', '😞', '(/◕ヮ◕)/', '😔', ':^)', 'x-p', '=-3', '🙊', '_(._.)_', '(^.^)/~~~', ":'-)", '8-0', ':((', '🏂', '=D>', '>:p', ':-"', '(?_?)', '🤓', '(so)', 'o_O', ':)>-', ':‑j', '😽', 't.t', '\\U0001f604', '^_^', ';‑]', '👶', '🙍', ',:-)', ':-L', ':-p', ':!!', '(^o^)／', '[-(', '😱', '(－‸ლ)', 'd=', '(+_+)', 'x-d', '=p~', '#:-S', ':‑|', ':‑þ', '😇', ';^)', '=-D', '😣', ':o3', '(~o~)', '(C)', '😿', "+/'\\", ':-j', '（´∀｀）', '%-(', '8-d', '0:‑3', '(´･ω･`)', '(-""-)', '$-)', '＼(◎o◎)／！', '😥', ';d', ':-þ', '<3', '<^!^>', '(;', '(Z)', '🙅', ':###..', 'Q_Q', '😒', '(._.)', '=:)', 'XP', '(6)', '=:o]', '(＾▽＾)', '=((', '(:_;)', ':‑)', '<(｀^´)>', ':o)', '(I)', '👽', '🎧', 'x-@', '(B)', '(^<^)', '(^^)', ':‑/', '@>‑‑>‑‑', "D-':", 'o/\\o', '=))', '😙', '(E)', '\\U0001f44b', '🍵', '☄️', ':\\', '>:s', '😄', ':-ss', '(>_<)>', ':-??', ';))', '(y)', ';-;', 'O:‑)', '😉', '(N)', '（’-’*)', '😫', ':]', '@}->--', ':っc', '<):)', '😮', '^o)', '(*^^)v', ':-#', ':^*', '>.<', '}:‑)', '(p)', 'X‑P', ':-(', '(゜_゜>)', ':-[', '8)', '(d)', '(o)', ':‑O', '(・・?', '°O°', ':p', '😟', '😹', '(^)o(^)', '%%-', 'V.v.V', 'x‑p', '(-_-メ)', "d-':", '(一一)', '(li)', ':-P', ':o', '(__)', '<`～´>', '|-O', '(*^。^*)', 'ヽ(^o^)丿', '😤', '\\u0001f3fb', '(ToT)', 'o_o', ':X', ':|', '(8)', '(^ム^)', ':-*', '=/', 'DX', '(-_-)zzz', '~X(', '\\U0001f1f3', 'XD', '8-X', '#-o', "(';')", '(=^・・^=)', '(ーー゛)', ';n;', ':}', '😀', 'O_o', '(&)', ':P', '😓', '🙌', ':‑J', ':っ)', '🙉', '3:-o', '[-o<', '(^_^メ)', ':‑X', 'xp', '\\U0001f64f', '🤔', '(sn)', '\\u0001f620', '🙀', '8‑d', 'd:', '(^_^)/~', '+o(', '|‑o', ':-))', '∩(・ω・)∩', ':-&', ':‑o', '(||)', ':{', '(*°∀°)=3', '>_>^', '\\:D/', '(~_~;)', '\\u0001f602', '8o|', '(})', ':<', '🐍', '(ー_ー)!!', '>:-)', '\\u0001f64f', '💯', '👋', 'l-)', '(~_~メ)', '\\u0001f3c2', '(um)', "@}‑;‑'‑‑‑", '(:', '°o°', 'm(__)m', '☺️', '(-.-)', '😳', ':‑#', '(-_-)', ':))', '8‑D', '~:‑\\', '😖', '><>', '3:‑)', '>:D<', '(^J^)', '🔘', '</3', ':-x', '(^o^)丿', '＼(-o-)／', '(゜.゜)', '(^0_0^)', '><(((*>', ':‑[', 'D8', '>:S', '~o)', '(co)', '😡', '//0-0\\\\', "('_')", '(mp)', '<:-|', '(=_=)', 'x_x', '}:-)', '(g)', '//0‑0\\\\', ';-D', '(^_^.)', '=3', ':-.', '(^^;)', '(‘a`)', ':-@', '😲', 'q.q', ')^o^(', '😦', '\\U0001f3c2', '8-)', '🐠', '(;_;)/~~~', '😯', '🦇', '(;_:)', '8-D', '^#(^', '(t_t)/~~~', '(;O;)', '-D', ':O', '😐', ':@', ':-0', '<(o0o)>', '((d[-_-]b))', '(%)', '*-)', '>:[', '(^.^)', ':-###..', '😪', '0:‑)', ':&', '☺', '(・_・;)', '(tot)', '(ip)', ':(', '😃', '(au)', 'o_0', '\\u0001f61c', '\\U0001f602', 'B^D', ':-Þ', '@};-', ':^(', '😠', '(^_^)v', ':d', '（⌒▽⌒）', '(゜レ゜)', ':‑P', '(ToT)/~~~', ':*', '😁', ':bz', '(bah)', 'o.O', ';-d', '%)', ":'-(", ':っC', '😈', 'o.o', '(>_<)', '(K)', '🇳', '🤑', '🐱', '~x(', '(b)', '🤦', '(・。・;)', '=]', '#-)', 'O:)', '({)', ':)]', '（・∀・）', '🦈', '~(_8^(I)', '😵', '(pi)', '>:‑)', '(T_T)', 'o->', '=\\', '(;o;)', 'o-o', '(*^0^*)', 'L-)', '=;', '☹️', '(O)', ':‑p', ';]', '<\\3', '🐟', ':!', '(x)', '🙈', ':#', '😴', '8d', ':>', '(t_t)', '（＾－＾）', '🇮', ':-q', '[:|]', '🖐', '(^)', ':->', '(P)', '>:\\', '(a)', ';D', '7:^]', 'v.v.v', '\\U0001f590', ':~)', "D‑':", '^/^', '(st)', '(-:', '(s)', 'b-)', '😨', '|;-)', 'i-)', ':c)', ':O)', '🤕', '(Y)', 'x-D', '😅', '😼', '%‑)', ':-3', ';-)', ':‑&', '\\U0001f620', '(T_T)/~~~', '😊', '(・ω・)', ':b', '（＾◇＾）', '\\U0001f518', ':-s', '(D)', '0:3', ':S', '(U)', '=_^=', ':l', '(o.o)', ":'‑(", 'I-)', '(^_-)', '*-:)', '\\U0001f4af', '\\o/', '😬', 'o=>', 'd:<', 'ヽ(^。^)ノ', '(#^.^#)', '($・・)/~~~', '((+_+))', ':-$', ':^o', '(M)', ':-?', 'o:)', '🙄', '@-)', 'qq', '=d>', '(^^ゞ', '8‑0', '.....φ(・∀・＊)', '=L', '>:{', ':[', ':/', '(W)', ':-,', '(^^)/~~~', '😺', ';P', '~O)', 'D:', '🙂', '\\u0001f60a', ':"(', '(c)', ':-S', '😌', '(f)', ':×', '(A)', '(z)', '(^。^)', '🐳', '^ω^', '😩', ':ar!', ':þ', '💤', '(@_@)', '(^_^)/', '>_>^^<_<', '\\u0001f4af', '😻', 'd8', '(X)', '<(__)>', '|‑O', '😾', ':@)', '(S)', '[..]', '>:(', ':-t', '😷', '8->', '(^^)v', '_(_^_)_', '(l)', '**==', ':,(', ':‑Þ', '(F)', '0;^)', 'xd', '@>-->--', ':-h', '*\\0/*', ':‑###..', '^<_<', '(:|', '(t)', ':L', '=P', '>^_^<', '（*^_^*）', '(*^▽^*)', '🙁', '😶', ':‑<', '\\:d/', '\\U0001f1ee', '#:-s', '(*)', ':-)', '(-_-;)', ';-]', '<*)))‑{', '*<|:-)', '(k)', '（￣□￣；）', 'q_q', '5:‑)', '>:/', '💣', '\\u0001f1f3', '(^j^)', '(T)', '(n)', ':-X', 'v.v', '0:-3', '(｀´）', '（*´▽｀*）', '(o|o)', '(゜_゜)', ':-O', '👾', '🐋', '🙃', '<m(__)m>', '<:-p', ':‑x', '\\u0001f590', '(@^^)/~~~', '=(', '0:)', ':‑c', '~(_8^(i)', '＼(~o~)／', 'xD', '😢', '(^_^;)', '^:)^', ',:‑)', '^_^;', ':x', 'd;', ':$', ':‑(', ':s', ';‑)', 'X_X', '\\u0001f518', 'x(', '[-x', 'x‑d', '^5', '(u)', '🙏', '(H)', '\\U0001f61c', '😝', ':\\*', '(h)', '😏', '8-x', 'D=', ':‑d', ':-l', '^^;', '|-o', 'B-)', ':Þ', '-d', '(G)', ':">', '😸', ';p', '🏻', ':-d', '|;‑)', '\\u0001f1ee', 'b^d', 'X-P', '<:o)', '😰', '\\U0001f31f', '=l', 'o‑o', '😆', '*<|:‑)', '(゜-゜)', '🤐', '\\u0001f31f', '*)', 'X-D', '0:-)', '>:)', 'dx', '\\u0001f604', '🙎', '☕️', ":'(", ':(:)', '😋', '☹', ':-<', '(‘A`)', '(;_;', "@}-;-'---", 'o-+', '>;)', ':‑,', ':-w', '}:)', '＼(^o^)／', ':-||', '=D', "d‑':", 'x‑D', '>:d<', ':‑b', '😘', '🤢', ';)', '(m)', '>:o', '(L)', '🙇', '(pl)', '🤒', '3:-)', '(；一_一)', 'QQ', '\\m/', '😭', '#‑)', '\\u0001f44b', '(e)', '=d', ':-b', ":')", '(ap)', '<:-P', '👼', ':(|)', '(;_;)', '=-d', '🚬', '~:-\\', '>:O', '🌟', ':-/', '§^。^§', '^m^', 'T.T', 'O-O', '(*^3^)/~☆', '3:-O', '😛', ':c', '=)', '\\U0001f60a', ':-c', '😚', '🙋', '(@)', '(~)', '8D', '(ｔ▽ｔ)', ';_;', '（￣ー￣）', '\\U0001f3fb', '😑', 'D;', '(・・;)', ':-bd', '>-)', '😂', '~@~', 'X‑D', '😍', '/:)', '>:', '>:P', '(^_^)', '(・へ・)', '<*)))-{', '(^O^)／', '😕', '[-X', '🐬', ':-SS', '（●＾o＾●）', '（＾ｕ＾）', 'o:‑)', '(@_@。', '(ーー;)', '💃', 'X(', ':-o', '(゜o゜)', '(*￣m￣)', ':-D', '(*_*;', ':D', '😎', '(^^)/', ":'‑)", 'O:-)', '（＾ｖ＾）', '🐡', '(≧∇≦)/', '🐙', '😧', ':3', '(p_-)', ':-B', '(-;', '(..)', ':-}', ';o)', '(+o+)', '(~~)', ':-|', '(V)o￥o(V)', '<:‑|', 'Q.Q', ';;)', '(^o^)', '5:-)', '8-}', '(mo)', '(i)', '(/_;)', 'x-(', '☻', ':‑D', 'O_O', '[-O<', '(－－〆)', '%-)', '!(^^)!', '(゜゜)', '(tot)/~~~', '(=^・^=)', ':-J', 'D:<', '✌️', '(^O^)', '8-|', '(~_~)']
def GetEmoticons(tweet):
	emoticons=[]
	for emoticon in all_emoticons:
		if emoticon in tweet:
			emoticons.append(emoticon)
	return emoticons
def BaseTokeniser(tweet,lowercase=False):
	if lowercase:
		tweet = tweet.lower()
	tokenised_tweet = tweet.split(' ')
	tokenised_tweet = list(filter(None, tokenised_tweet))
	return tokenised_tweet

def give_emoji_free_text(text):
	return emoji.get_emoji_regexp().sub(u'', text)

def remove_emoticon(tweet):
	tweet = p.clean(tweet)
	'''delims = GetEmoticons(BaseTokeniser(tweet))
	for delim in delims:
		tweet = tweet.replace(delim,"")'''
	tweet = give_emoji_free_text(tweet)
	return tweet

sentences = [
	"_waga i can honestly say i wasn't playing this whole time. promise!",
	"So there is no way for me to plug it in here in the US unless I go by a converter.",
	"Thanks x https://t.co/ZXTcDLyDS9",
	"I saw the new #JOHNDOE movie AND IT SUCKS!!! WAISTED $10... #badmovies :/",
	"Good case, Excellent value. 100%",
	"Works great!",
	'The design is very odd, as the ear "clip" is not very comfortable at all.',
	"Needless to say, I wasted my money.",
	 "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／ yaaaay!!! #davidlynch #tvseries :)))",
	"I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies >3:/",
	"@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! >:-D http://sentimentsymposium.com/.",
	"@Calum5SOS You lil *poop* please follow @EmilyBain224 ☺️💕",
	"Words attendees would use to describe @prosper4africa's #ALN 2015! https://t.co/hmNm8AdwOh",
	 "@TheTideDrew Hi, Drew! I can't wait to see you!☺ Just letting you know that you'll always be my spidey, I love you!💕 Mind following me? x215",
	 "Welcome to http://t.co/3YNnqG23S9(SAMSUNG,Kodak,bp,Coca-cola,Holcim,Nestle) 26.11.2013( http://t.co/9Xp7K7g3UW MAILAN(KISSME) SINCE 1969",
	 "canbe foundathttp://www.osp.gatech.edu/rates/(http://www.osp.gatech.edu/rates/).",
	 "[https://link.springer.com/article/10.1007/s10940\\-016\\-9314\\-9](https://link.springer.com/article/10.1007/s10940-016-9314-9)",
	 "Call me on +917893774735 and 7893774735 is my number 888-222-111 10PM 2 pm 3 P.M f**k",
	 "<html>hello</html>, <body></body> , </br>",
	 "2getr"
]

text_processor = TextPreProcessor(
	# terms that will be normalized
	#omit=["percent","money","phone","time","date","number"],
	omit=['url', 'email', 'user'],
	normalize=['url', 'email', 'user'],

	#normalize=['url', 'email', 'user',"percent","money","phone","time","date","number"],
	# terms that will be annotated
	annotate={"elongated", "repeated",'emphasis', 'censored'},
	fix_bad_unicode = True,
	fix_text=True,
	fix_html=True,
	# corpus from which the word statistics are going to be used
	# for word segmentation
	segmenter="twitter",

	# corpus from which the word statistics are going to be used
	# for spell correction
	corrector="twitter",

	unpack_hashtags=True,  # perform word segmentation on hashtags
	unpack_contractions=True,  # Unpack contractions (can't -> can not)
	spell_correct_elong=False,  # spell correction for elongated words
	spell_correction=False,
	# select a tokenizer. You can use SocialTokenizer, or pass your own
	# the tokenizer, should take as input a string and return a list of tokens
	tokenizer=SocialTokenizer(lowercase=True).tokenize,

	# list of dictionaries, for replacing tokens extracted from the text,
	# with other expressions. You can pass more than one dictionaries.
)
def remove_tags(doc):
	"""
	Remove tags from sentence
	"""
	doc = ' '.join(word for word in doc.split() if word[0]!='<')
	return doc

def remove_slang(sent):
	return [slangdict[w.strip()] if w.strip() in slangdict else w for w in sent]

sents = [remove_emoticon(" ".join(remove_slang(sent.split()))) for sent in sentences]
tokenized_sentences = list(text_processor.pre_process_docs(sents))
for x in range(len(tokenized_sentences)):
	sent = " ".join(tokenized_sentences[x])
	sent = TwitterPreprocessor(sent).remove_blank_spaces().text
	print(sentences[x])
	print(sent,"\n\n")
