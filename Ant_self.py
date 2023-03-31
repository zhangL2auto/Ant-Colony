'''
Author: zhangL
Date: 2023-03-29 16:34:57
LastEditors: zhangL
LastEditTime: 2023-03-31 16:48:26
FilePath: /Hope/programm/Ant_Colony/Ant_self.py
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

class AntColony:
    def __init__(self, distance_graph, ant_num, iter_num = 100, alpha = 1, beta = 2, decay = 0.5):
        self.distance_graph = distance_graph
        self.pheromone_graph = np.ones(self.distance_graph.shape) / len(self.distance_graph)
        self.ant_num = ant_num
        self.city_num = len(self.distance_graph)
        self.iter_num = iter_num
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

    def __clean_data(self):
        self.total_distance = 0
        self.current_city = -1
        self.path = []
        self.open_list = [True for i in range(self.city_num)]
        self.move_count = 0

        # first time
        city_index = np.random.randint(0, len(self.distance_graph))
        self.current_city = city_index
        self.path.append(city_index)
        self.open_list[city_index] = False
        self.move_count = 1

# get the next city to go!
    def __choice_move(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(self.city_num)]
        total_prob = 0.0
 
        for i in range(self.city_num):
            if self.open_list[i]:
                try :
                    select_citys_prob[i] = pow(self.pheromone_graph[self.current_city][i], self.alpha) * pow((1.0/self.distance_graph[self.current_city][i]), self.beta)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print ('current city: {current}, target city: {target}'.format(current = self.current_city, target = i))
                    sys.exit(1)
        
        if total_prob > 0.0:
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.city_num):
                if self.open_list[i]:
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
 
        if (next_city == -1):
            next_city = random.randint(0, self.city_num - 1)
            while ((self.open_list[next_city]) == False):
                next_city = random.randint(0, self.city_num - 1)
 
        return next_city

# according to the next city, calculate the path and move_count
    def __get_path(self):
        prev = self.current_city
        for i in range(len(self.distance_graph) - 1):
            next_city = self.__choice_move()
            self.path.append(next_city)
            self.open_list[next_city] = False
            self.current_city = next_city
            self.move_count += 1
        self.path.append(prev)

# according to the path, calculate the total distance of path
    def __get_total_ditances(self):

        temp_distance = 0.0
        for i in range(1, self.city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += self.distance_graph[start][end]
        
        end = self.path[0]
        temp_distance += self.distance_graph[start][end]
        self.total_distance = temp_distance

# according to the paths of all ants, calculate the all paths
    def __get_all_paths_and_distances(self):
        all_paths = []
        all_total_distance = []
        for ant in range(self.ant_num):
            self.__clean_data()
            self.__get_path()
            self.__get_total_ditances()
            all_paths.append(self.path)
            all_total_distance.append(self.total_distance)
        
        return all_paths, all_total_distance

# according to the all paths, calculate the pheromone update    
    def __update_pheromone(self, all_paths, all_total_distance):
        temp_pheromone = [[0.0 for i in range(self.city_num)] for j in range(self.city_num)]
        for ant in range(self.ant_num):
            for i in range(1, self.city_num):
                start, end = all_paths[ant][i-1], all_paths[ant][i]
    
                temp_pheromone[start][end] += 100 / all_total_distance[ant]
                temp_pheromone[end][start] = temp_pheromone[start][end]
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone_graph[i][j] = self.pheromone_graph[i][j] * self.decay + temp_pheromone[i][j]

    def run(self):
        shortest_distance = np.inf

        for iter in range(self.iter_num):
            all_paths, all_total_distance = self.__get_all_paths_and_distances()
            self.__update_pheromone(all_paths, all_total_distance)
            best_ant_distance = min(all_total_distance)
            min_index = all_total_distance.index(best_ant_distance)
            print("iteration:{}, best:{}".format(iter, shortest_distance))
            if best_ant_distance < shortest_distance:
                shortest_distance = best_ant_distance
            best_path = all_paths[min_index]
        return best_path
    


if __name__ == '__main__':
    distances = np.array([[0, 3, 1, 2],
                      [3, 0, 5, 4],
                      [1, 5, 0, 2],
                      [2, 4, 2, 0]])
    distance_x = [
    178,272,176,171,650,499,267,703,408,437,491,74,532,
    416,626,42,271,359,163,508,229,576,147,560,35,714,
    757,517,64,314,675,690,391,628,87,240,705,699,258,
    428,614,36,360,482,666,597,209,201,492,294]
    distance_y = [
    170,395,198,151,242,556,57,401,305,421,267,105,525,
    381,244,330,395,169,141,380,153,442,528,329,232,48,
    498,265,343,120,165,50,433,63,491,275,348,222,288,
    490,213,524,244,114,104,552,70,425,227,331]
# data processer
    import math
    def cal_distance(x, y):
            distance = np.zeros((len(x), len(y)))
            for row in range(len(x)):
                for col in range(len(y)):
                    distance[row][col] = math.sqrt((x[row] - x[col]) ** 2 + (y[row] - y[col]) ** 2)
            return distance
    distance = cal_distance(distance_x, distance_y)



    ant_colony = AntColony(distance,10, 100)
    shortest_path = ant_colony.run()
    print ("shorted_path: {}".format(shortest_path))

