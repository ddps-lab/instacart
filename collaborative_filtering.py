#!/usr/bin/env python
# Implementation of collaborative filtering recommendation engine

from recommendation_data import dataset
from math import sqrt

def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 
 
	both_viewed = {}		# To get both rated items by person1 and person2
 
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1
 
		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0
 
		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	
 
		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
 
		return 1/(1+sqrt(sum_of_eclidean_distance))

print "Lisa Rose rating on Lady in the water: {}n".format(dataset['Lisa Rose']['Lady in the Water'])
print "Michael Phillips rating on Lady in the water: {}n".format(dataset['Michael Phillips']['Lady in the Water'])

print '**************Jack Matthews ratings**************'
print dataset['Jack Matthews']

print '************** Smililarity Score Euclidean Distance Socre Lisa and Jack **************'
print similarity_score('Lisa Rose','Jack Matthews')