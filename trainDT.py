import argparse
import numpy as np
from array import array

# Design tree node structure #
class Node:
	def __init__(self, impurity_method, node_level, purity, data, mfeature):
		self.left = None
		self.right = None
		self.data_idx = data
		self.impurity_method = impurity_method
		self.dfeature = None #This gets assigned when finding the split feature
		self.impurity = purity
		self.nlevels = node_level
		self.class_label = None #This gets assigned when finding the split feature
		self.mfeatures = mfeature

	# This is the root node creation
	def buildDT(root, args):
		DT = Node.splitNode(root, args[3], args[4], args[1])
		return DT

	def splitNode(self, nl, p, train_label_matrix):
		if self.nlevels < nl and self.impurity < p: #This does not make sense at ALL #
			maxGain = -1
			splitFeature = -1

			for feature_i in range(0, self.mfeatures):
				current_feature = self.data_idx[:,feature_i]

				# The following block of code is intended to split up the feature based on 0 and 1's
				leftArray = []
				rightArray = []

				for data in range (0, len(current_feature)):
					# Adding to array the specific index located
					if current_feature[data] == 0:
						leftArray.append(data)
					else:
						rightArray.append(data)
				# After splitting up the feature, calculate the impurity 
				Pleft = self.calculateIP(leftArray, current_feature) #all the features on that data point that is 0
				Pright = self.calculateIP(rightArray, current_feature) #all of the features on the data point that is 1

				# Calculate the impurity measure after splitting 
				# Compute impurtiy measure of each child node, Pleft and Pright
				# M is the weighted impurity of the children
				M = ((Pleft * len(leftArray)) + (Pright * len(rightArray))) / (len(leftArray) + len(rightArray))
				Gain = self.impurity - M

				# Here, choose the attribute test condition that produces the highest gain
				if Gain > maxGain:
					maxGain = Gain
					splitFeature = feature_i
					self.dfeature = splitFeature
					self.label = train_label_matrix[splitFeature] #Here, we are assigning the label given the data point
			# Update decision feature to be the feature split on, based upon gain and reassign data
			data_idx_left = [[]]
			data_idx_right = [[]]
			splitFeatureData = self.data_idx[splitFeature, :]
			print (splitFeatureData)

			for data in range (0, len(splitFeatureData)):
				self.mfeatures = len(splitFeatureData)
				# Adding to array the specific index located
				if splitFeatureData[data] == 0:
					currentRow = self.data_idx[:, data]
					print(currentRow)
					np.append(data_idx_left, currentRow, axis = 0)

					print("-----------------")
				else:
					print("in 1")
					current = np.vstack([self.data_idx[:, data]])
					data_idx_right.append(current)
			# Now, create the nodes from the children
			self.left = Node(self.impurity_method, self.nlevels+1, Pleft, data_idx_left, len(splitFeatureData)) # Impurity method is the method of the parent #
			self.right = Node(self.impurity_method, self.nlevels+1, Pright, data_idx_right, len(splitFeatureData))
			dtL = Node.splitNode(self.left, nl, p, train_label_matrix)

			#self.right.splitNode(nl, p, train_label_matrix)

			currentNode = [self.class_label, self.left.class_label, self.right.class_label]
			return currentNode

	# Method serves as router between data and impurity methods
	def calculateIP(self, data, current_feature):
		if self.impurity_method == "gini":
			P = self.calculateGINI(data, current_feature)

		else:
			P = calculateEntropy(data)
		return P

	# Gini index is solved as 1 - the sum of the relative frequency of class j at node t squared
	def calculateGINI(self, data, data_idk):
		ginicof = 0.7
		return ginicof

	def calculateEntropy(data):
		
		return entropy

	def classify():
		args = getArguments()
		pred_file = args[7]
def getArguments():
	parser = argparse.ArgumentParser()

	# Add all of the arguments needed from the command line #
	parser.add_argument('-train_data', action="store", required=True)
	parser.add_argument('-train_label', action="store", required=True)
	parser.add_argument('-test_data', action="store", required=True)
	parser.add_argument('-test_label', action="store") #Not required, only for precision
	parser.add_argument('-nlevels', action="store", required=True)
	parser.add_argument('-pthrd', action="store", required=True)
	parser.add_argument('-impurity', action="store", required=True)
	parser.add_argument('-pred_file', action="store", required=True)

	# Convert the arguments to strings for better manipulation# 
	args = parser.parse_args()
	train_label = str(args.train_label)
	train_data = str(args.train_data)
	test_data = str(args.test_data)
	test_label = str(args.test_label)
	nlevels = str(args.nlevels)
	pthrd = str(args.pthrd)
	impurity = str(args.impurity)
	pred_file = str(args.pred_file)

	# Create matrixs for the data for better access #
	train_label_matrix = np.genfromtxt(train_label, delimiter =' ')
	train_data_matrix = np.genfromtxt(train_data, delimiter = ' ')
	#test_label_matrix = np.genfromtxt(test_label, delimiter = ' ') #Not required, only for precision
	test_data_matrix = np.genfromtxt(test_data, delimiter = ' ')

	args = [train_data_matrix, train_label_matrix, test_data_matrix, nlevels, pthrd, impurity, pred_file]
	return args


if __name__ == '__main__':
	args = getArguments()
	rootNode = Node(args[5], 0, 4, args[0], args[0].shape[1]) #This is the root node
	DT = Node.buildDT(rootNode, args)








