import tensorflow as tf
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize

'''
Setting MODE to 0 trains the model
Setting MODE to 1 allows the model to run on user input 
Setting MODE to 2 creates final 'predictions.json' file
'''
MODE = 1
'''
Lets you toggle whether you want to used previously saved
weights from the last run or override any previously saved 
weights.
'''
USE_OLD = True

#Load wordVectors and wordVecDim from pickle file
wordVectors, wordVecDim = pickle.load(open('wv.pkl','rb'))
def getVector(w):
	if (w in wordVectors):
		return wordVectors[w]
	return np.zeros(wordVecDim)

def wordVectorize(s):
	return [getVector(w) for w in word_tokenize(s)]

if(MODE==0 or MODE==2):
	print("Loading data...")

	if(MODE==0):
		labeledReviews = json.load(open('trainingDataset.json','r'))
	if(MODE==2):
		testReviews = json.load(open('testDataset.json','r'))

	def extract(reviews, ratingsPresent):

		titles = [r['title'] for r in reviews]
		texts = [r['review_text'] for r in reviews]
		if(ratingsPresent):
			ratings = [float(r['rating']) > 3 for r in reviews]
		numReviews = len(reviews)

		titleVectors = [wordVectorize(t) for t in titles]
		textVectors = [wordVectorize(t) for t in texts]

		def validReview(i):
			return len(titleVectors[i])>0 and len(textVectors[i])>0

		def onlyValid(l):
			return [l[i] for i in range(numReviews) if validReview(i)]

		if(ratingsPresent):
			ratings, titleVectors, textVectors = onlyValid(ratings), onlyValid(titleVectors), onlyValid(textVectors)
			return ratings, titleVectors, textVectors
		else:
			titleVectors, textVectors = onlyValid(titleVectors), onlyValid(textVectors)
			return titleVectors, textVectors

	if(MODE==0):
		tvCutoff = int(len(labeledReviews)*0.8)
		trReviews, valReviews = labeledReviews[:tvCutoff], labeledReviews[tvCutoff:]


		def sentimentSplit(reviews):
			ratings, titles, texts = extract(reviews, ratingsPresent=True)
			numValidReviews = len(titles)

			positiveReviews = [(titles[i], texts[i], ratings[i]) for i in range(numValidReviews) if ratings[i]==1]

			negativeReviews = [(titles[i], texts[i], ratings[i]) for i in range(numValidReviews) if ratings[i]==0]
			
			return positiveReviews, negativeReviews

		positiveTrain, negativeTrain = sentimentSplit(trReviews)
		numPositive = len(positiveTrain)
		numNegative = len(negativeTrain)

		valRatings, valTitles, valTexts = extract(valReviews, ratingsPresent=True)
		numVal = len(valTitles)

		valSet = [(valTitles[i], valTexts[i], valRatings[i]) for i in range(numVal)]
	if(MODE==2):
		teTitles, teTexts = extract(testReviews, ratingsPresent=False)
		numTest = len(teTitles)

		testSet = [(teTitles[i], teTexts[i]) for i in range(numTest)]

print("Setting up graph...")

titlePlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, wordVecDim])
textPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, wordVecDim])
y = tf.placeholder(dtype=tf.float32)

cellSize = 20
titleLSTM = tf.nn.rnn_cell.LSTMCell(num_units=cellSize)
textLSTM = tf.nn.rnn_cell.LSTMCell(num_units=cellSize)

_,finalTitleState = tf.nn.dynamic_rnn(cell=titleLSTM,
								 inputs=tf.reshape(titlePlaceholder,[1,-1,wordVecDim]),
								 dtype=tf.float32,
								 scope="titleLSTM")
titleCellState = tf.reshape(finalTitleState[0],[cellSize, 1])

_,finalTextState = tf.nn.dynamic_rnn(cell=textLSTM,
								 inputs=tf.reshape(textPlaceholder,[1,-1,wordVecDim]),
								 dtype=tf.float32,
								 scope="textLSTM")
textCellState = tf.reshape(finalTextState[0],[cellSize, 1])

combinedState = tf.concat((titleCellState, textCellState), axis = 0)


def weight(s):
	return tf.Variable(tf.truncated_normal(s, stddev=0.1))

hiddenSize = 10

hiddenMap = weight([hiddenSize, 2*cellSize])
#hiddenMap = weight([hiddenSize, cellSize])
hiddenBias = tf.Variable(tf.zeros([hiddenSize, 1]))
hiddenLayer = tf.nn.relu(tf.matmul(hiddenMap, combinedState) + hiddenBias)
#hiddenLayer = tf.nn.relu(tf.matmul(hiddenMap, textCellState) + hiddenBias)

predictionMap = weight([hiddenSize, 1])
logit = tf.reduce_sum(tf.multiply(predictionMap, hiddenLayer))
prediction = tf.sigmoid(logit)

#loss = (y-prediction)**2
epsilon = 1e-2
loss = -(y*tf.log(prediction + epsilon) + (1-y)*tf.log(1 - prediction + epsilon))
correct = (y*prediction + (1-y)*(1-prediction)) > 0.5

LEARNING_RATE = 4e-4
trainStep = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

if(USE_OLD):
	print("Restoring weights...")
	saver.restore(sess,"saved/weights.ckpt")
else:
	sess.run(tf.global_variables_initializer())

def printPerformance(predictions):
	correctNegative = predictions[0][1]/(sum(predictions[0]))
	correctPositive = predictions[1][1]/(sum(predictions[1]))
	normalizedAccuracy = 0.5*(correctPositive+correctNegative)
	print(f"Accuracies")
	print(f"Positive Accuracy: {correctPositive} Negative Accuracy: {correctNegative}")
	print(f"Overall Accuracy: {normalizedAccuracy}")

if(MODE==0):
	print("Training model...")

	BATCHES = 50
	BATCH_SIZE = 1000

	p = 0
	n = 0

	for i in range(BATCHES):
		batchLoss = 0
		predictions = [[0,0],[0,0]]
		for r in range(BATCH_SIZE):
			print(f"Batch {i+1} training {r+1}/{BATCH_SIZE}", end='\r')
			if(r%2==0):
				title, text, rating = positiveTrain[p]
				p = (p+1)%numPositive
			else:
				title,  text,rating = negativeTrain[n]
				n = (n+1)%numNegative
			l, c, _ = sess.run([loss, correct, trainStep], 
							feed_dict={titlePlaceholder:title, textPlaceholder:text, y:rating})
			predictions[rating][c]+=1;
			batchLoss+=l
		print()
		print(f"Batch {i+1} Loss: {batchLoss/BATCH_SIZE}")
		printPerformance(predictions)
		print("---------")
		if(i%4==3):
			print("Validation Performance")
			valLoss = 0
			predictions = [[0,0],[0,0]]
			for v in valSet:
				title, text, rating = v
				l, c = sess.run([loss, correct], 
							feed_dict={titlePlaceholder:title, textPlaceholder:text, y:rating})
				predictions[rating][c]+=1;
				valLoss+=l
			print(f"LOSS: {valLoss/numVal}")
			print("Saving weights...")
			printPerformance(predictions)
			saver.save(sess, "saved/weights.ckpt")
		print("---------")

if(MODE==1):
	while(True):
		reviewTitle = input("Supply a Review Title: ")
		reviewText = input("Supply text for your Review: ")
		p = sess.run(prediction, feed_dict={titlePlaceholder:wordVectorize(reviewTitle), 
											textPlaceholder:wordVectorize(reviewText)})
		if(p > 0.5):
			print(f"POSITIVE with {(p*100):.2f}% confidence.")
		else:
			print(f"NEGATIVE with {((1-p)*100):.2f}% confidence.")
		print()

if(MODE==2):
	print("Making test predictions...")
	predictions = []
	for i in range(numTest):
		print(f"Predicting {i+1}/{numTest}",end='\r')
		title, text = testSet[i]
		pred = sess.run(prediction, feed_dict={titlePlaceholder:title, textPlaceholder:text})
		predictions.append(float(pred)>0.5)
	json.dump(predictions, open('predictions.json','w'))



