import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.cross_validation import KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
#from sklearn.utils.extmath import np.dot

warnings.simplefilter("error")

users = 6040
items = 3952

def readingFile(filename):
	f = open(filename,"r")
	data = []
	for row in f:
		r = row.split(',')
		e = [int(r[0]), int(r[1]), int(r[2])]
		data.append(e)
	return data

def similarity_user(data):
	print "Hello User"
	#f_i_d = open("sim_user_based.txt","w")
	user_similarity_cosine = np.zeros((users,users))
	user_similarity_jaccard = np.zeros((users,users))
	user_similarity_pearson = np.zeros((users,users))
	for user1 in range(users):
		print user1
		for user2 in range(users):
			if np.count_nonzero(data[user1]) and np.count_nonzero(data[user2]):
				user_similarity_cosine[user1][user2] = 1-scipy.spatial.distance.cosine(data[user1],data[user2])
				user_similarity_jaccard[user1][user2] = 1-scipy.spatial.distance.jaccard(data[user1],data[user2])
				try:
					if not math.isnan(scipy.stats.pearsonr(data[user1],data[user2])[0]):
						user_similarity_pearson[user1][user2] = scipy.stats.pearsonr(data[user1],data[user2])[0]
					else:
						user_similarity_pearson[user1][user2] = 0
				except:
					user_similarity_pearson[user1][user2] = 0

			#f_i_d.write(str(user1) + "," + str(user2) + "," + str(user_similarity_cosine[user1][user2]) + "," + str(user_similarity_jaccard[user1][user2]) + "," + str(user_similarity_pearson[user1][user2]) + "\n")
	#f_i_d.close()
	return user_similarity_cosine, user_similarity_jaccard, user_similarity_pearson

def crossValidation(data):
	k_fold = KFold(n=len(data), n_folds=10)

	Mat = np.zeros((users,items))
	for e in data:
		Mat[e[0]-1][e[1]-1] = e[2]

	sim_user_cosine, sim_user_jaccard, sim_user_pearson = similarity_user(Mat)
	#sim_user_cosine, sim_user_jaccard, sim_user_pearson = np.random.rand(users,users), np.random.rand(users,users), np.random.rand(users,users)

	'''sim_user_cosine = np.zeros((users,users))
	sim_user_jaccard = np.zeros((users,users))
	sim_user_pearson = np.zeros((users,users))

	f_sim = open("sim_user_based.txt", "r")
	for row in f_sim:
		r = row.strip().split(',')
		sim_user_cosine[int(r[0])][int(r[1])] = float(r[2])
		sim_user_jaccard[int(r[0])][int(r[1])] = float(r[3])
		sim_user_pearson[int(r[0])][int(r[1])] = float(r[4])
	f_sim.close()'''

	rmse_cosine = []
	rmse_jaccard = []
	rmse_pearson = []

	for train_indices, test_indices in k_fold:
		train = [data[i] for i in train_indices]
		test = [data[i] for i in test_indices]

		M = np.zeros((users,items))

		for e in train:
			M[e[0]-1][e[1]-1] = e[2]

		true_rate = []
		pred_rate_cosine = []
		pred_rate_jaccard = []
		pred_rate_pearson = []

		for e in test:
			user = e[0]
			item = e[1]
			true_rate.append(e[2])

			pred_cosine = 3.0
			pred_jaccard = 3.0
			pred_pearson = 3.0

			#user-based
			if np.count_nonzero(M[user-1]):
				sim_cosine = sim_user_cosine[user-1]
				sim_jaccard = sim_user_jaccard[user-1]
				sim_pearson = sim_user_pearson[user-1]
				ind = (M[:,item-1] > 0)
				#ind[user-1] = False
				normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
				normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
				normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
				if normal_cosine > 0:
					pred_cosine = np.dot(sim_cosine,M[:,item-1])/normal_cosine

				if normal_jaccard > 0:
					pred_jaccard = np.dot(sim_jaccard,M[:,item-1])/normal_jaccard

				if normal_pearson > 0:
					pred_pearson = np.dot(sim_pearson,M[:,item-1])/normal_pearson

			if pred_cosine < 0:
				pred_cosine = 0

			if pred_cosine > 5:
				pred_cosine = 5

			if pred_jaccard < 0:
				pred_jaccard = 0

			if pred_jaccard > 5:
				pred_jaccard = 5

			if pred_pearson < 0:
				pred_pearson = 0

			if pred_pearson > 5:
				pred_pearson = 5

			print str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine) + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson)
			pred_rate_cosine.append(pred_cosine)
			pred_rate_jaccard.append(pred_jaccard)
			pred_rate_pearson.append(pred_pearson)

		rmse_cosine.append(sqrt(mean_squared_error(true_rate, pred_rate_cosine)))
		rmse_jaccard.append(sqrt(mean_squared_error(true_rate, pred_rate_jaccard)))
		rmse_pearson.append(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))

		print str(sqrt(mean_squared_error(true_rate, pred_rate_cosine))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))
		#raw_input()

	#print sum(rms) / float(len(rms))
	rmse_cosine = sum(rmse_cosine) / float(len(rmse_cosine))
	rmse_pearson = sum(rmse_pearson) / float(len(rmse_pearson))
	rmse_jaccard = sum(rmse_jaccard) / float(len(rmse_jaccard))

	print str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson)

	f_rmse = open("rmse_user.txt","w")
	f_rmse.write(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

	rmse = [rmse_cosine, rmse_jaccard, rmse_pearson]
	req_sim = rmse.index(min(rmse))

	print req_sim
	f_rmse.write(str(req_sim))
	f_rmse.close()

	if req_sim == 0:
		sim_mat_user = sim_user_cosine

	if req_sim == 1:
		sim_mat_user = sim_user_jaccard

	if req_sim == 2:
		sim_mat_user = sim_user_pearson

	#predictRating(Mat, sim_mat_user)
	return Mat, sim_mat_user


def predictRating(recommend_data):

	M, sim_user = crossValidation(recommend_data)

	#f = open("toBeRated.csv","r")
	f = open(sys.argv[2],"r")
	toBeRated = {"user":[], "item":[]}
	for row in f:
		r = row.split(',')	
		toBeRated["item"].append(int(r[1]))
		toBeRated["user"].append(int(r[0]))

	f.close()

	pred_rate = []

	#fw = open('result1.csv','w')
	fw_w = open('result1.csv','w')

	l = len(toBeRated["user"])
	for e in range(l):
		user = toBeRated["user"][e]
		item = toBeRated["item"][e]

		pred = 3.0

		#user-based
		if np.count_nonzero(M[user-1]):
			sim = sim_user[user-1]
			ind = (M[:,item-1] > 0)
			#ind[user-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				pred = np.dot(sim,M[:,item-1])/normal

		if pred < 0:
			pred = 0

		if pred > 5:
			pred = 5

		pred_rate.append(pred)
		print str(user) + "," + str(item) + "," + str(pred)
		#fw.write(str(user) + "," + str(item) + "," + str(pred) + "\n")
		fw_w.write(str(pred) + "\n")

	#fw.close()
	fw_w.close()

#recommend_data = readingFile("ratings.csv")
recommend_data = readingFile(sys.argv[1])
#crossValidation(recommend_data)
predictRating(recommend_data)

