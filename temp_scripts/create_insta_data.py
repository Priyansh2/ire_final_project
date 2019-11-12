import os,sys,re,json
import dill as pickle
import csv
from collections import Counter,defaultdict
import random
import codecs
import shutil
def get_random_number(nums,min,max,step):
	while True:
		r = random.randrange(min,max+2,step)
		if r not in nums:
			num = r
			break
	return num

DATA_TYPE="insta"
if DATA_TYPE in ("tw","insta","fb"):
	DATA_PATH = "../data/raw_data/"+DATA_TYPE
INSTA_DATA=defaultdict(lambda: defaultdict(list))
filter_=[]
idx=0
def extract_data(data):
	global idx
	for x in data:
		comments,captions=[],[]
		for key in x:
			if key=="comments":
				for comment in x["comments"]:
					if 'comment' in comment:
						cmnt = comment['comment'].strip()
						if cmnt and cmnt not in comments:
							comments.append(cmnt)
			elif key=="caption":
				captions.append(x['caption'])
		if captions and len(captions)==1:
			caption = captions[0].strip()
			if caption not in filter_:
				filter_.append(caption)
				INSTA_DATA[idx]['caption']=[caption]
				INSTA_DATA[idx]['comments']=comments
				idx+=1
def get_data():
	for filename in sorted(os.listdir(DATA_PATH+"/instagram_data")):
		if not os.path.isfile(os.path.join(DATA_PATH+"/instagram_data",filename)):
			continue
		filpath = os.path.join(DATA_PATH+"/instagram_data",filename)
		with open(filpath,'r') as f:
			data = json.load(f)
		f.close()
		extract_data(data)

def comments_stats():
	comments=[]
	for idx in INSTA_DATA:
		if INSTA_DATA[idx]['comments']:
			for comment in INSTA_DATA[idx]['comments']:
				if comment not in comments:
					comments.append(comment)
	return len(comments)
get_data()
print("Total posts/captions ",len(INSTA_DATA))
print("Total comments ",comments_stats())
f = open(os.path.join(DATA_PATH,"insta_captions"),"w")
content=[]
valid_ids=[]
for idx in INSTA_DATA:
	for caption in INSTA_DATA[idx]['caption']:
		caption = caption.strip()
		if caption:
			valid_ids.append(idx)
			content.append(str(idx)+":"+caption.replace("\n",""))
f.write("--------------------".join(content))
f.close()
f = open(os.path.join(DATA_PATH,"insta_comments"),"w")
content=[]
for idx in valid_ids:
	if idx in INSTA_DATA and INSTA_DATA[idx]['comments']:
		for comment in INSTA_DATA[idx]['comments']:
			content.append(str(idx)+":"+comment)
f.write("--------------------".join(content))
f.close()
