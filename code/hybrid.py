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

def userData():
	filename = sys.argv[3]
	f = open(filename,"r")
	data = np.zeros((users,3))
	for row in f:
		r = row.strip().split(',')
		if r[1] == "M" or r[1] == "m":
			data[int(r[0])-1] = [1,(int(r[2])/56.0),((int(r[3])+1.0)/21.0)]
		else:
			data[int(r[0])-1] = [0,(int(r[2])/56.0),((int(r[3])+1)/21.0)]

	return data


def itemData():
	filename = sys.argv[4]
	f = open(filename,"r")
	data = np.zeros((items,18))
	genre = {"Action":0, "Adventure":1, "Animation":2, "Children's":3, "Comedy":4, "Crime":5, "Documentary":6, "Drama":7, "Fantasy":8, "Film-Noir":9, "Horror":10, "Musical":11, "Mystery":12, "Romance":13, "Sci-Fi":14, "Thriller":15, "War":16, "Western":17 }
	for row in f:
		r = row.split(',')
		g = r[len(r)-1].split('|')
		for e in g:
			if e.strip() not in genre.keys():
				continue
			else:
				data[int(r[0])-1][genre[e.strip()]] = 1

	return data

def similarity_item(data):
	print "Hello Item"
	#f_i_d = open("sim_item_hybrid.txt","w")
	item_similarity_cosine = np.zeros((items,items))
	item_similarity_jaccard = np.zeros((items,items))
	item_similarity_pearson = np.zeros((items,items))
	for item1 in range(items):
		print item1
		for item2 in range(items):
			if np.count_nonzero(data[item1]) and np.count_nonzero(data[item2]):
				item_similarity_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[item1],data[item2])
				item_similarity_jaccard[item1][item2] = 1-scipy.spatial.distance.jaccard(data[item1],data[item2])
				try:
					if not math.isnan(scipy.stats.pearsonr(data[item1],data[item2])[0]):
						item_similarity_pearson[item1][item2] = scipy.stats.pearsonr(data[item1],data[item2])[0]
					else:
						item_similarity_pearson[item1][item2] = 0
				except:
					item_similarity_pearson[item1][item2] = 0

			#f_i_d.write(str(item1) + "," + str(item2) + "," + str(item_similarity_cosine[item1][item2]) + "," + str(item_similarity_jaccard[item1][item2]) + "," + str(item_similarity_pearson[item1][item2]) + "\n")
	#f_i_d.close()
	return item_similarity_cosine, item_similarity_jaccard, item_similarity_pearson


def similarity_user(data):
	print "Hello User"
	#f_i_d = open("sim_user_hybrid.txt","w")
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

def crossValidation(data, user_data, item_data):
	k_fold = KFold(n=len(data), n_folds=10)

	sim_user_cosine, sim_user_jaccard, sim_user_pearson = similarity_user(user_data)
	sim_item_cosine, sim_item_jaccard, sim_item_pearson = similarity_item(item_data)
	#sim_user_cosine, sim_user_jaccard, sim_user_pearson = np.random.rand(users,users), np.random.rand(users,users), np.random.rand(users,users)
	#sim_item_cosine, sim_item_jaccard, sim_item_pearson = np.random.rand(items,items), np.random.rand(items,items), np.random.rand(items,items) 

	'''sim_user_cosine = np.zeros((users,users))
	sim_user_jaccard = np.zeros((users,users))
	sim_user_pearson = np.zeros((users,users))

	f_sim = open("sim_user_hybrid.txt", "r")
	for row in f_sim:
		#print row
		r = row.strip().split(',')
		sim_user_cosine[int(r[0])][int(r[1])] = float(r[2])
		sim_user_jaccard[int(r[0])][int(r[1])] = float(r[3])
		sim_user_pearson[int(r[0])][int(r[1])] = float(r[4])
	f_sim.close()


	sim_item_cosine = np.zeros((items,items))
	sim_item_jaccard = np.zeros((items,items))
	sim_item_pearson = np.zeros((items,items))

	f_sim_i = open("sim_item_hybrid.txt", "r")
	for row in f_sim_i:
		#print row
		r = row.strip().split(',')
		sim_item_cosine[int(r[0])][int(r[1])] = float(r[2])
		sim_item_jaccard[int(r[0])][int(r[1])] = float(r[3])
		sim_item_pearson[int(r[0])][int(r[1])] = float(r[4])
	f_sim_i.close()'''

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

			user_pred_cosine = 3.0
			item_pred_cosine = 3.0

			user_pred_jaccard = 3.0
			item_pred_jaccard = 3.0

			user_pred_pearson = 3.0
			item_pred_pearson = 3.0

			#item-based
			if np.count_nonzero(M[:,item-1]):
				sim_cosine = sim_item_cosine[item-1]
				sim_jaccard = sim_item_jaccard[item-1]
				sim_pearson = sim_item_pearson[item-1]
				ind = (M[user-1] > 0)
				#ind[item-1] = False
				normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
				normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
				normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
				if normal_cosine > 0:
					item_pred_cosine = np.dot(sim_cosine,M[user-1])/normal_cosine

				if normal_jaccard > 0:
					item_pred_jaccard = np.dot(sim_jaccard,M[user-1])/normal_jaccard

				if normal_pearson > 0:
					item_pred_pearson = np.dot(sim_pearson,M[user-1])/normal_pearson

			if item_pred_cosine < 0:
				item_pred_cosine = 0

			if item_pred_cosine > 5:
				item_pred_cosine = 5

			if item_pred_jaccard < 0:
				item_pred_jaccard = 0

			if item_pred_jaccard > 5:
				item_pred_jaccard = 5

			if item_pred_pearson < 0:
				item_pred_pearson = 0

			if item_pred_pearson > 5:
				item_pred_pearson = 5

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
					user_pred_cosine = np.dot(sim_cosine,M[:,item-1])/normal_cosine

				if normal_jaccard > 0:
					user_pred_jaccard = np.dot(sim_jaccard,M[:,item-1])/normal_jaccard

				if normal_pearson > 0:
					user_pred_pearson = np.dot(sim_pearson,M[:,item-1])/normal_pearson

			if user_pred_cosine < 0:
				user_pred_cosine = 0

			if user_pred_cosine > 5:
				user_pred_cosine = 5

			if user_pred_jaccard < 0:
				user_pred_jaccard = 0

			if user_pred_jaccard > 5:
				user_pred_jaccard = 5

			if user_pred_pearson < 0:
				user_pred_pearson = 0

			if user_pred_pearson > 5:
				user_pred_pearson = 5

			if (user_pred_cosine != 0 and user_pred_cosine != 5) and (item_pred_cosine != 0 and item_pred_cosine != 5):
				pred_cosine = (user_pred_cosine + item_pred_cosine)/2
			else:
				if (user_pred_cosine == 0 or user_pred_cosine == 5):
					if (item_pred_cosine != 0 and item_pred_cosine != 5):
						pred_cosine = item_pred_cosine
					else:
						pred_cosine = 3.0
				else:
					if (user_pred_cosine != 0 and user_pred_cosine != 5):
						pred_cosine = user_pred_cosine
					else:
						pred_cosine = 3.0

			if (user_pred_jaccard != 0 and user_pred_jaccard != 5) and (item_pred_jaccard != 0 and item_pred_jaccard != 5):
				pred_jaccard = (user_pred_jaccard + item_pred_jaccard)/2
			else:
				if (user_pred_jaccard == 0 or user_pred_jaccard == 5):
					if (item_pred_jaccard != 0 and item_pred_jaccard != 5):
						pred_jaccard = item_pred_jaccard
					else:
						pred_jaccard = 3.0
				else:
					if (user_pred_jaccard != 0 and user_pred_jaccard != 5):
						pred_jaccard = user_pred_jaccard
					else:
						pred_jaccard = 3.0

			if (user_pred_pearson != 0 and user_pred_pearson != 5) and (item_pred_pearson != 0 and item_pred_pearson != 5):
				pred_pearson = (user_pred_pearson + item_pred_pearson)/2
			else:
				if (user_pred_pearson == 0 or user_pred_pearson == 5):
					if (item_pred_pearson != 0 and item_pred_pearson != 5):
						pred_pearson = item_pred_pearson
					else:
						pred_pearson = 3.0
				else:
					if (user_pred_pearson != 0 and user_pred_pearson != 5):
						pred_pearson = user_pred_pearson
					else:
						pred_pearson = 3.0
			
			#pred_cosine = (user_pred_cosine + item_pred_cosine)/2
			#pred_jaccard = (user_pred_jaccard + item_pred_jaccard)/2
			#pred_pearson = (user_pred_pearson + item_pred_pearson)/2
			print str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine) + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson)
			pred_rate_cosine.append(pred_cosine)
			pred_rate_jaccard.append(pred_jaccard)
			pred_rate_pearson.append(pred_pearson)

		#print len(true_rate)
		#print len(pred_rate_cosine)
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

	f_rmse = open("rmse_hybrid.txt","w")
	f_rmse.write(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

	rmse = [rmse_cosine, rmse_jaccard, rmse_pearson]
	req_sim = rmse.index(min(rmse))

	print req_sim
	f_rmse.write(str(req_sim))
	f_rmse.close()

	if req_sim == 0:
		sim_mat_user = sim_user_cosine
		sim_mat_item = sim_item_cosine

	if req_sim == 1:
		sim_mat_user = sim_user_jaccard
		sim_mat_item = sim_item_jaccard

	if req_sim == 2:
		sim_mat_user = sim_user_pearson
		sim_mat_item = sim_item_pearson

	#predictRating(data, sim_mat_user, sim_mat_item)
	return sim_mat_user, sim_mat_item


def predictRating(data, user_data, item_data):

	sim_user, sim_item = crossValidation(data, user_data, item_data)

	M = np.zeros((users,items))
	for e in data:
		M[e[0]-1][e[1]-1] = e[2]

	#f = open("toBeRated.csv","r")
	f = open(sys.argv[2],"r")	
	toBeRated = {"user":[], "item":[]}
	for row in f:
		r = row.split(',')	
		toBeRated["item"].append(int(r[1]))
		toBeRated["user"].append(int(r[0]))

	f.close()

	pred_rate = []

	#fw = open('result3.csv','w')
	fw_w = open('result3.csv','w')

	l = len(toBeRated["user"])
	for e in range(l):
		user = toBeRated["user"][e]
		item = toBeRated["item"][e]

		user_pred = 3.0
		item_pred = 3.0

		#item-based
		if np.count_nonzero(M[:,item-1]):
			sim = sim_item[item-1]
			ind = (M[user-1] > 0)
			#ind[item-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				item_pred = np.dot(sim,M[user-1])/normal

		if item_pred < 0:
			item_pred = 0

		if item_pred > 5:
			item_pred = 5

		#user-based
		if np.count_nonzero(M[user-1]):
			sim = sim_user[user-1]
			ind = (M[:,item-1] > 0)
			#ind[user-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				user_pred = np.dot(sim,M[:,item-1])/normal

		if user_pred < 0:
			user_pred = 0

		if user_pred > 5:
			user_pred = 5

		if (user_pred != 0 and user_pred != 5) and (item_pred != 0 and item_pred != 5):
				pred = (user_pred + item_pred)/2
		else:
			if (user_pred == 0 or user_pred == 5):
				if (item_pred != 0 and item_pred != 5):
					pred = item_pred
				else:
					pred = 3.0
			else:
				if (user_pred != 0 and user_pred != 5):
					pred = user_pred
				else:
					pred = 3.0

		#pred = (user_pred + item_pred)/2
		pred_rate.append(pred)
		print str(user) + "," + str(item) + "," + str(pred)
		#fw.write(str(user) + "," + str(item) + "," + str(pred) + "\n")
		fw_w.write(str(pred) + "\n")

	#fw.close()
	fw_w.close()

#recommend_data = readingFile("ratings.csv")
recommend_data = readingFile(sys.argv[1])
user_data = userData()
item_data = itemData()
predictRating(recommend_data, user_data, item_data)
#crossValidation(recommend_data, user_data, item_data)


