import os,sys,re,json
import dill as pickle
import csv
from collections import Counter,defaultdict
import random
import codecs
import shutil
cnt=0
def get_random_number(nums,min,max,step):
	while True:
		r = random.randrange(min,max+2,step)
		if r not in nums:
			num = r
			break
	return num

def utf16_to_utf8(path):
	for dir_ in os.listdir(path):
		if not os.path.isfile(os.path.join(path,dir_)):
			for file in os.listdir(os.path.join(path,dir_)):
				inpfile = os.path.join(os.path.join(path,dir_),file)
				with open(inpfile, 'rb') as source_file:
					with open(os.path.join(path,file), 'w+b') as dest_file:
						contents = source_file.read()
						dest_file.write(contents.decode('utf-16').encode('utf-8'))

DATA_TYPE="fb"
if DATA_TYPE in ("tw","insta","fb"):
	DATA_PATH = "../data/raw_data/"+DATA_TYPE
FB_PAGES = {
	'bbc': '228735667216',
	'FoxNews': '15704546335',
	'ABCNews': '86680728811',
	'NBCNews': '155869377766434',
	'CBSNews': '131459315949',
	'cnn': '5550296508',
	'msnbc': '273864989376427',
	'NPR': '10643211755',
	'politico': '62317591679',
	'Reuters': '114050161948682',
	'washingtonpost': '6250307292',
	'NYtimesworldnews': '5281959998',
	'TheEconomist': '6013004059',
	'financialtimes': '8860325749',
	'theguardian': '10513336322',
	'DailyMail': '164305410295882',
	'NYDailyNews': '268914272540',
	'Breitbart': '95475020353',
	'InfoWarCom': '80256732576',
	'Huffingtonnews': '18468761129',
	'dailykos': '43179984254',
	'salon': '120680396518',
	'SLUHillNews': '7533944086',
	'nationalreview': '15779440092',
	'usatoday': '13652355666',
	'wsj': '8304333127',
	'BuzzFeedNews': '618786471475708',
	'cnbc': '97212224368',
	'Newsweek': '18343191100',
	'APNews': '249655421622',
	'BloombergPolitics': '1481073582140028',
	'yahoonews': '338028696036',
	'chicagotribune': '5953023255',
	'latimes': '5863113009',
	'thedailybeast': '37763684202',
	'cyberDrudge': '1416139158459267',
	'TheBlaze': '140738092630206',
	'TheYoungCons': '147772245840',
	'DailyCaller': '182919686769',
	'newsmax': '85452072376',
	'WNDNews': '119984188013847',
	'theijr': '687156898054966',
	'time': '10606591490',
	'usnewsandworldreport': '5834919267',
	'businessinsider': '20446254070',
	'Slate': '21516776437',
	'Vox': '223649167822693',
	'thinkprogress': '200137333331078',
	'democratic_undergound': '455410617809655',
	'talkingpointsmemo': '98658495398',
	'TheNationMagazine': '7629206115',
	'motherjones': '7642602143',
	'TheRawStory': '20324257234',
	'propublica': '13320939444',
	'townhallcom': '41632789656',
	'WashingtonExaminer': '40656699159',
	'TheDailySignalNews': '300341323465160',
	'weeklystandard': '11643473298',
	'TheAtlantic': '29259828486',
	'newyorker': '9258148868',
	'MorningJoe': '90692553761',
	'vicenews': '235852889908002',
	'RTAmerica': '326683984410',
	'aljazeera': '7382473689',
	'OneAmericaNewsNetwork': '220198801458577',
	'ChristianScienceMonitor': '14660729657',
	'newshour': '6491828674',
	'miamiherald': '38925837299',
	'person_alex_jones': '6499393458',
	'andersoncooper': '60894670532',
	'therachelmaddowshow': '25987609066',
	'person_sean_hannity': '69813760388',
	'person_chris_matthews': '114114045339706',
	'MegynKelly': '1425464424382692',
	'TeamCavuto': '101988643193353',
	'chrishayesmsnbc': '153005644864469',
	'person_shepard_smith': '131010790489',
	'person_erin_burnett': '102938906478343',
	'person_joe_scarobourgh': '144128236879',
	'RushLimbaugh': '136264019722601',
	'Maher': '62507427296',
	'OfficialAnnCoulter': '695526053890545'
	}
files= [file.split("_output.txt")[0] for file in os.listdir(os.path.join(DATA_PATH,"scraped_posts"))]
matched_pages=[]
unmatched_new_pages=[]
unmatched_old_pages=[]
for filename in files:
	if filename not in FB_PAGES:
		unmatched_new_pages.append(filename)
	else:
		matched_pages.append(filename)
for pgname in FB_PAGES:
	if pgname not in matched_pages:
		unmatched_old_pages.append(pgname)

def strip_delims(s):
	return s.replace('"',"").replace("'","")


FB_PAGES_DATA_PATH = os.path.join(DATA_PATH,"facebook-news-master")
FB_PAGES_DATA=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
page_post_id_map=defaultdict(list)
with open(os.path.join(FB_PAGES_DATA_PATH,"fb_news_posts_20K.csv")) as f:
	rows = csv.reader(f, delimiter=',')
	l=0
	for row in rows:
		if l and row:
			try:
				x = strip_delims(row[5]).split("_")
				pageid = x[0]
				postid = x[1]
				post_text = row[3]
				if pageid.isdigit() and postid.isdigit() and postid not in page_post_id_map[pageid]:
					page_post_id_map[pageid].append(postid)
					if not post_text.isdigit() and post_text.strip():
						FB_PAGES_DATA[pageid]['post'][postid].append(post_text.strip())
						cnt+=1
					#else:
						#print("lol",post_text,[post_text])
				#else:
					#print("lol1")
			except:
				pass
		l+=1
print(cnt)
cnt=0
#path=os.path.join(DATA_PATH,"martinchek-2012-2016-facebook-posts")
#utf16_to_utf8(path)
for file in os.listdir(os.path.join(DATA_PATH,"martinchek-2012-2016-facebook-posts/utf8-converted")):
	if os.path.isfile(os.path.join(DATA_PATH,"martinchek-2012-2016-facebook-posts/utf8-converted/"+file)):
		with open(os.path.join(DATA_PATH,"martinchek-2012-2016-facebook-posts/utf8-converted/"+file)) as f:
			rows = csv.reader(f,delimiter=',')
			l=0
			for row in rows:
				if l and row:
					try:
						row[0] = row[0].encode('utf-8').decode('utf-8-sig')
						x = strip_delims(row[0]).split("_")
						pageid = x[0]
						postid = x[1]
						post_text = row[4].encode('utf-8').decode('utf-8-sig')
						if pageid.isdigit() and postid.isdigit() and postid not in page_post_id_map[pageid]:
							page_post_id_map[pageid].append(postid)
							if not post_text.isdigit() and post_text.strip() and post_text!='NULL':
								#print("ok")
								FB_PAGES_DATA[pageid]['post'][postid].append(post_text.strip())
								cnt+=1
							#else:
								#print("lol",post_text)
						else:
							print("lol1",[pageid],[postid])
					except:
						pass
				l+=1
print(cnt)
cnt=0
with open(os.path.join(FB_PAGES_DATA_PATH,"fb_news_comments_1000K.csv")) as f:
	rows = csv.reader(f,delimiter=',')
	l=0
	for row in rows:
		if l and row:
			try:
				x = strip_delims(row[4]).split("_")
				pageid = x[0]
				postid = x[1]
				comment_text = row[3]
				if pageid.isdigit() and postid.isdigit() and pageid in page_post_id_map and postid in page_post_id_map[pageid] and not comment_text.isdigit() and comment_text.strip():
						FB_PAGES_DATA[pageid]['comment'][postid].append(comment_text.strip())
			except:
				pass
		l+=1

for filename in matched_pages+unmatched_new_pages:
	file = os.path.join(os.path.join(DATA_PATH,"scraped_posts"),filename+"_output.txt")
	content = [line.strip() for line in open(file,"r").read().split("\n") if line.strip()]
	if filename in FB_PAGES:
		pageid = FB_PAGES[filename]
	else:
		temp = sorted(list(map(int, list(page_post_id_map.keys()))))
		pageid = get_random_number(temp,temp[0],temp[-1],1)
		pageid = str(pageid)
		if pageid in page_post_id_map:
			print("lol")
			print(pageid,page_post_id_map[pageid])
	for post in content:
		post = json.loads(post)
		if 'Post' in post and post['Post'].strip():
			posts=[]
			posts.append(post['Post'].strip())
			if 'Comments' in post and post['Comments']:
				comments=[]
				for username in post['Comments']:
					if 	post['Comments'][username] and 'text' in post['Comments'][username]:
						for item in post['Comments'][username]:
							if item=="text" and post['Comments'][username]['text'].strip():
								comments.append(post['Comments'][username]['text'].strip())
		if posts:
			if pageid in page_post_id_map:
				temp = sorted(list(map(int, page_post_id_map[pageid])))
				postid = get_random_number(temp,temp[0],temp[-1],1)
				postid = str(postid)
				assert postid not in page_post_id_map[pageid]
				page_post_id_map[pageid].append(postid)
			else:
				postid="9999999999"
			FB_PAGES_DATA[pageid]['post'][postid].extend(posts)
			cnt+=1
			if comments:
				FB_PAGES_DATA[pageid]['comment'][postid].extend(comments)

		else:
			print("lol",post)
print(cnt)
temp=[]
f = open(os.path.join(DATA_PATH,"fb_posts"),"w")
content=[]
filter_=[]
for pageid in FB_PAGES_DATA:
	for postid in FB_PAGES_DATA[pageid]['post']:
		for post_text in list(set(FB_PAGES_DATA[pageid]['post'][postid])):
			if "_".join([pageid.strip(),postid.strip()]) not in temp:
				temp.append("_".join([pageid.strip(),postid.strip()]))
				content.append("_".join([pageid.strip(),postid.strip()])+","+post_text.strip().replace("\n",""))
			#else:
				#print("lol")
f.write("--------------------".join(content))
f.close()
print("Total FB_posts: ",len(content))
f = open(os.path.join(DATA_PATH,"fb_comments"),"w")
content=[]
for id_ in temp:
	pageid=id_.split("_")[0]
	postid = id_.split("_")[1]
	for comment_text in list(set(FB_PAGES_DATA[pageid]['comment'][postid])):
		if comment_text not in filter_:
			filter_.append(comment_text)
		content.append("_".join([pageid,postid])+","+comment_text.strip().replace("\n",""))
f.write("--------------------".join(content))
f.close()
print("Total FB_comments: ",len(filter_))


