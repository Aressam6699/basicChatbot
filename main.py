import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow as tf
import tflearn
import random
import json
import pickle

#import sys	
#numpy.set_printoptions(threshold=sys.maxsize)

with open("intents.json") as file:
	data = json.load(file)
try:
	x
	with open("data.pickle","rb") as f:
		words, labels, training, output = pickle.load(f)

except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	print("Words : \n",words,"\n")
	print("Labels : \n",labels,"\n")
	print("doc_x :\n",docs_x)
	print("doc_y : \n",docs_y)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	#in training 
	output = numpy.array(output)


	print("len(words):",len(words))
	print("len(doc_x):",len(docs_x))
	print("training\n",len(training),len(training[0]),"\n",training)
	print("output\n",len(output),"\n",output)

	with open("data.pickle","wb") as f:
		pickle.dump((words, labels, training, output),f)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
# 	x
# 	model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if(w == se):
				bag[i] = 1

	return numpy.array(bag)

def chat():
	print("Start")
	while True:
		inp = input("You : ")
		if(inp.lower() == 'quit'):
			break

		results = model.predict([bag_of_words(inp,words)])
		#print(results)
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		#print(tag)
		for t in data["intents"]:
			if(t["tag"] == tag):
				responses = t["responses"]

		print("Bot : "+random.choice(responses))

chat()