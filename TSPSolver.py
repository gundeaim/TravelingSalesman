#!/usr/bin/python3
import TSPClasses
import time
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import Nodes



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	#TC: O(n^3) SC: O(n)
	def greedy(self, time_allowance=60.0):
		#set up initail variables
		results = {} #TC: O(1) SC: O(1)
		cities = self._scenario.getCities() #TC: O(1) SC: O(1)
		ncities = len(cities) #TC: O(1) SC: O(1)
		foundTour = False #TC: O(1) SC: O(1)
		count = 0 #TC: O(1) SC: O(1)
		start_time = time.time() #TC: O(1) SC: O(1)
		citiesVisited = [] #TC: O(1) SC: O(1)
		ultimateBssf = math.inf #TC: O(1) SC: O(1)
		citiesSolution = [] #TC: O(1) SC: O(1)
		ultimateCitiesSolution = [] #TC: O(1) SC: O(1)

		#outside for loop will make each city the starting node
		#outloop will run n times where n is the number of cities
		for x in range(ncities):
			citiesVisited.clear() #TC: O(1) SC: O(1)
			citiesSolution.clear() #TC: O(1) SC: O(1)

			#list of whether a city has been visited that can be searched by city index
			#TC: O(n) SC: O(n)
			for i in range(ncities):
				citiesVisited.append(math.inf) #TC: O(1) SC: O(1)

			#sets the current city and resets need variables.
			citiesVisited[x] = 0 #TC: O(1) SC: O(1)
			startCity = x #TC: O(1) SC: O(1)
			fromCity = startCity #TC: O(1) SC: O(1)
			ncitiesVisited = 1 #TC: O(1) SC: O(1)
			smallestCost = math.inf #TC: O(1) SC: O(1)
			smallestCity = startCity #TC: O(1) SC: O(1)
			possible = True #TC: O(1) SC: O(1)
			bssf = 0 #TC: O(1) SC: O(1)
			citiesSolution.append(cities[startCity]) #TC: O(1) SC: O(1)

			#will loop until we have either found a tour or a dead end
			#while loop will run at the most O(n - 1) times this gives TC: O(n^2) SC: O(n)
			while not foundTour and possible:
				#goes through the cities to find shortest path to connected city
				#TC: O(n) SC: O(n) we will be saving the cities in a list for a solution
				for i in range(ncities):
					if i != fromCity: #TC: O(1) SC: O(1)
						#checks to ensure we don't revisit a city
						if citiesVisited[i] == math.inf: #TC: O(1) SC: O(1)
							cost = TSPClasses.City.costTo(cities[fromCity], cities[i]) #TC: O(1) SC: O(1)

							#finds smallest cost to next possible city
							if smallestCost > cost: #TC: O(1) SC: O(1)
								smallestCost = cost #TC: O(1) SC: O(1)
								smallestCity = i #TC: O(1) SC: O(1)

				#checks to see if we can make it to another city
				if smallestCost == math.inf: #TC: O(1) SC: O(1)
					possible = False #TC: O(1) SC: O(1)
				else:
					citiesVisited[smallestCity] = 0 #TC: O(1) SC: O(1)
					bssf = bssf + smallestCost #TC: O(1) SC: O(1)
					fromCity = smallestCity #TC: O(1) SC: O(1)
					citiesSolution.append(cities[smallestCity]) #TC: O(1) SC: O(1)
					ncitiesVisited = ncitiesVisited + 1 #TC: O(1) SC: O(1)
					smallestCost = math.inf #TC: O(1) SC: O(1)

				#checks to see if we are at the last city
				if ncitiesVisited == len(cities): #TC: O(1) SC: O(1)
					cost = TSPClasses.City.costTo(cities[fromCity], cities[startCity]) #TC: O(1) SC: O(1)

					#checks to see if we can get back to the first city
					if cost != math.inf: #TC: O(1) SC: O(1)
						bssf = bssf + cost #TC: O(1) SC: O(1)
						foundTour = True #TC: O(1) SC: O(1)

			#checks to see if the solution given made it to all cities
			if bssf != 0: #TC: O(1) SC: O(1)
				if len(citiesSolution) == len(cities): #TC: O(1) SC: O(1)
					count = count + 1 #TC: O(1) SC: O(1)

					#checks to see if the curr solution is better than previous if so it will update result variables
					if bssf < ultimateBssf: #TC: O(1) SC: O(1)
						ultimateBssf = bssf #TC: O(1) SC: O(1)
						ultimateCitiesSolution.clear() #TC: O(1) SC: O(1)
						for y in range(len(citiesSolution)): #TC: O(n) SC: O(n)
							ultimateCitiesSolution.append(citiesSolution[y])

		#set up results dictionary
		end_time = time.time() #TC: O(1) SC: O(1)
		results['cost'] = ultimateBssf #TC: O(1) SC: O(1)
		results['time'] = end_time - start_time #TC: O(1) SC: O(1)
		results['count'] = count #TC: O(1) SC: O(1)
		results['soln'] = TSPSolution(ultimateCitiesSolution) #TC: O(1) SC: O(1)
		results['max'] = None #TC: O(1) SC: O(1)
		results['total'] = None #TC: O(1) SC: O(1)
		results['pruned'] = None #TC: O(1) SC: O(1)

		return results
	
	
	

	#Total TC: O(n^2) SC: O(n^2)
	def createReduxMatrix(self):
		#creates the original root nodes Redux matrix.
		citiesList = TSPClasses.Scenario.getCities(self._scenario) #TC: O(1) SC: O(1)
		matrix = np.zeros((len(citiesList), len(citiesList))) #TC: O(1) SC: O(n^2) where n is num of cities

		#fills new root matrix with the cost from city to city
		# two for loops will be TC: O(n^2) SC: O(1) since it will run through all of the matrix
		for i in range(len(citiesList)):
			for j in range(len(citiesList)):
				matrix[i][j] = TSPClasses.City.costTo(citiesList[i], citiesList[j]) #TC: O(1) SC: O(1)
		return matrix

	#TC: O(n^2) SC: O(1) Note: Space complexity will not increase in the reduce Matrix call
	#due to the matrix already existing. The function is only editing existing addresses
	def reduceMatrix(self, matrix, bound = 0):
		lowerBound = bound #TC: O(1) SC: O(1)

		#goes through rows one by one looking for smallest number in row and then subtracts it once found
		#TC: O(n^2) SC: O(1) to run both loops
		for i in range(matrix.shape[0]):
			smallestNum = math.inf #TC: O(1) SC: O(1)

			#Goes through all columns of a given row to find smallest path length
			for j in range(matrix.shape[1]):
				if smallestNum > matrix[i][j]: #TC: O(1) SC: O(1)
					smallestNum = matrix[i][j] #TC: O(1) SC: O(1)
			#goes through all columns again to subtract smallest path length found for one row
			if smallestNum != math.inf: #TC: O(1) SC: O(1)
				for x in range(matrix.shape[1]): #TC: O(1) SC: O(1)
					matrix[i][x] = matrix[i][x] - smallestNum #TC: O(1) SC: O(1)
				lowerBound = lowerBound + smallestNum #TC: O(1) SC: O(1)

		#goes through all columns to make sure all zeros
		# TC: O(n^2) SC: O(1) to run both loops
		for j in range(matrix.shape[1]):
			smallestNum = math.inf #TC: O(1) SC: O(1)
			#goes through all the rows for a paticular col. for smallest number
			for i in range(matrix.shape[0]):
				if smallestNum > matrix[i][j]: #TC: O(1) SC: O(1)
					smallestNum = matrix[i][j] #TC: O(1) SC: O(1)
			#will delete the smallest number in the column from every entry in that column
			if smallestNum != math.inf: #TC: O(1) SC: O(1)
				for x in range(matrix.shape[0]): #TC: O(1) SC: O(1)
					matrix[x][j] = matrix[x][j] - smallestNum #TC: O(1) SC: O(1)
				lowerBound = lowerBound + smallestNum #TC: O(1) SC: O(1)

		return matrix, lowerBound

	#given a parent node makeChildren will create a child node for every other city
	#TC: O(n^2) SC: O(n^2)
	def makeChildren(self, parentName):
		#set up initial variables
		numCities = len(TSPClasses.Scenario.getCities(self._scenario)) #TC: O(1) SC: O(1)
		childrenList = []#TC: O(1) SC: O(1)

		if len(parentName.citiesVisited) == numCities - 1:
			parentName.haveBeenToCity[0] = False


		#this for loop will make a Node for each city except the parentNode city
		#TC: O(n^2) SC: O(n^2)
		for i in range(numCities):
			if i != parentName.name:
				if parentName.haveBeenToCity[i] == False:
					haveBeenToCity = []	#TC: O(1) SC: O(1)
					citiesVisited = [] #TC: O(1) SC: O(1)
					bound = parentName.bound #TC: O(1) SC: O(1)
					name = i #TC: O(1) SC: O(1)
					#uses parent cities visited and add itself to the visited list
					for x in range(len(parentName.citiesVisited)): #TC: O(n) SC: O(n)
						citiesVisited.append(parentName.citiesVisited[x]) #TC: O(1) SC: O(1)
					citiesVisited.append(TSPClasses.Scenario.getCities(self._scenario)[i])

					for x in range(len(parentName.haveBeenToCity)):
						haveBeenToCity.append(parentName.haveBeenToCity[x])
					haveBeenToCity[i] = True#TC: O(1) SC: O(1)

					#creates child node matrix
					childMatrix = np.zeros((numCities, numCities)) #TC: O(1) SC: O(n^2)

					#for loops run through all of parents matrix to create the child matrix adding necessary inf.
					#TC: O(n^2) SC: O(n^2)
					for y in range(childMatrix.shape[0]):
						for z in range(childMatrix.shape[1]):
							childMatrix[y][z] = parentName.nodeMatrix[y][z] #TC: O(1) SC: O(1)
							if y == parentName.name: #TC: O(1) SC: O(1)
								childMatrix[y][z] = math.inf #TC: O(1) SC: O(1)
							if z == name: #TC: O(1) SC: O(1)
								childMatrix[y][z] = math.inf #TC: O(1) SC: O(1)
							if y == name and z == parentName.name: #TC: O(1) SC: O(1)
								childMatrix[y][z] = math.inf #TC: O(1) SC: O(1)

					#creates new bound for the node by taking parent node and reducing the current
					# matrix, adding the two bounds together
					bound = bound + parentName.nodeMatrix[parentName.name][name] #TC: O(1) SC: O(1)
					nodeMatrix, lowerBound = self.reduceMatrix(childMatrix, bound) #TC: O(n^2) SC: O(1)

					#creates the node with the information created above
					newNode = Nodes.Nodes(name, lowerBound, citiesVisited, nodeMatrix, haveBeenToCity) #TC: O(1) SC: O(1)
					childrenList.append(newNode) #TC: O(1) SC: O(1)

		return childrenList

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	# TC: O(n^2 * b^n) SC: O(n^2 * b^n) where n is the number of cities and b the average
	def branchAndBound(self, time_allowance=60.0):
		start = time.time() #TC: O(1) SC: O(1)
		ns = 0 #TC: O(1) SC: O(1)

		#Get initial node information
		numCities = len(TSPClasses.Scenario.getCities(self._scenario)) #TC: O(1) SC: O(1)

		initialMatrix = self.createReduxMatrix() #TC: O(n^2) SC: O(n^2)
		initialMatrix, lowerBound = self.reduceMatrix(initialMatrix) #TC: O(n^2) SC: O(1)
		bssf = self.greedy().get('cost')
		haveBeenToCity = []#TC: O(n^3) SC: O(n)
		haveBeenToCity.append(True)
		for i in range(1, numCities):
			haveBeenToCity.append(False)

		#set up result variables
		numOfSolutions = 0 #TC: O(1) SC: O(1)
		maxQueueSize = 1 #TC: O(1) SC: O(1)
		statesCreated = 1 #TC: O(1) SC: O(1)
		statesPruned = 0 #TC: O(1) SC: O(1)
		bestSolution = [] #TC: O(1) SC: O(1)
		results = {} #TC: O(1) SC: O(1)

		#create root Node & push onto the PQ
		root = Nodes.Nodes(0, lowerBound, [TSPClasses.Scenario.getCities(self._scenario)[0]], initialMatrix, haveBeenToCity) #TC: O(1) SC: O(1)
		PQ = [(1, root)]

		#heapq.heapify(PQ)#TC: O(1) SC: O(1)
		#while will run if the Priority Queue is not empty
		#TC: O(n^2 * b^n) SC: O(n^2 * b^n) where n is the number of cities and b the average
		# number of cities that will be added to the queue.
		while (len(PQ) == 0) == False:

			#checks to see if we are in time constraint if not it will return best solution so far
			if ns > time_allowance: #TC: O(1) SC: O(1)
				print("time-out")
				if len(bestSolution) == 0:
					print("No Solution found", bssf, ns, numOfSolutions, maxQueueSize, statesCreated, statesPruned)
				results['cost'] = bssf #TC: O(1) SC: O(1)
				results['time'] = ns #TC: O(1) SC: O(1)
				results['count'] = numOfSolutions #TC: O(1) SC: O(1)
				results['soln'] = TSPSolution(bestSolution) #TC: O(1) SC: O(1)
				results['max'] = maxQueueSize #TC: O(1) SC: O(1)
				results['total'] = statesCreated #TC: O(1) SC: O(1)
				results['pruned'] = statesPruned #TC: O(1) SC: O(1)
				return results #TC: O(1) SC: O(1)
			#currNode will now be whatever Node from the PQ had the highest priority
			#heapq.heapify(PQ)
			#currListItem = heapq.heappop(PQ) #TC: O(1) SC: O(1)
			#currNode = currListItem[1]
			currNode = PQ.pop()[1]
			#check the Nodes bound to see if it is better than the BSSF
			if currNode.bound < bssf: #TC: O(1) SC: O(1)
				#make children will create children nodes
				# from the current node to all other cities with the needed info
				babyNodes = self.makeChildren(currNode) #TC: O(n^3) SC: O(n^3)

				#look through all the children that were just made to see if we are at a leaf and if not
				#check the bound to see if they should be placed on the PQ
				for i in range(len(babyNodes)):
					statesCreated = statesCreated + 1 #TC: O(1) SC: O(1)
					depth = len(babyNodes[i].citiesVisited) #TC: O(1) SC: O(1)

					#checks to see if we are at a leaf node and if the solution is the best so far
					if depth == numCities and babyNodes[i].bound < bssf: #TC: O(1) SC: O(1)
						bssf = babyNodes[i].bound #TC: O(1) SC: O(1)
						bestSolution.clear() #TC: O(1) SC: O(1)
						#TC: O(n) SC: O(n)
						for x in range(len(babyNodes[i].citiesVisited)):
							bestSolution.append(babyNodes[i].citiesVisited[x]) #TC: O(1) SC: O(1)
						numOfSolutions = numOfSolutions + 1 #TC: O(1) SC: O(1)

					#if node isn't a leaf node and it has a better bssf than we have found so far
					# it will place it on the PQ and update queue size if necessary
					elif babyNodes[i].bound < bssf: #TC: O(1) SC: O(1)
						PQ.append(((1000/babyNodes[i].bound + depth), babyNodes[i])) #TC: O(1) SC: O(1)
						if len(PQ) > maxQueueSize: #TC: O(1) SC: O(1)
							maxQueueSize = len(PQ) #TC: O(1) SC: O(1)
					#if the solution isn't better node is not added to queue (pruned)
					else:
						statesPruned = statesPruned + 1 #TC: O(1) SC: O(1)

			else:
				statesPruned = statesPruned + 1
			end = time.time() #TC: O(1) SC: O(1)
			ns = end - start #TC: O(1) SC: O(1)

		results['cost'] = bssf #TC: O(1) SC: O(1)
		results['time'] = ns #TC: O(1) SC: O(1)
		results['count'] = numOfSolutions #TC: O(1) SC: O(1)
		results['soln'] = TSPSolution(bestSolution) #TC: O(1) SC: O(1)
		results['max'] = maxQueueSize #TC: O(1) SC: O(1)
		results['total'] = statesCreated #TC: O(1) SC: O(1)
		results['pruned'] = statesPruned #TC: O(1) SC: O(1)
		return results

	def find_new_pt_position(self, solution_list, new_pt, previous_tour_cost, cost_matrix):
		tourCost = previous_tour_cost
		cost_to_pt = 0
		cost_from_pt = 0
		cost_broken_edge = 0
		for i in range(len(solution_list)):
			if i == len(solution_list):
				from_city_index = solution_list[len(solution_list) - 1]
				to_city_index = solution_list[0]
			else:
				from_city_index = solution_list[i]
				to_city_index = solution_list[i+1]

			#compute new Tour Cost





		''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		



