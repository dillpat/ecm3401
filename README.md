# ecm3401 

The main idea of this progam is to create a maze like game, which is used as the basis for the optimization framework. On this maze coins and enemies are placed as objectives which are to be optimized. These coins and enemies are used to set objectives for each type of play style used in the project. The optimization frameworks utilised are the Gaussian process and random forest. From scikit-optimize we use the gp minimize and forest minimize package. These operate on the objective functions.

Class:
Coin - This class is used to represent the coins which are placed on the maze, each coin has their own x and y cooridante which determine their positon on the maze. They also have the state collected, which when a player passes over a coin will change to set as True to stop the player agent from collecting the same coin over and over. 

Class:
Enemy - This class is used to represent the enemies which are placed on the maze, each enemy has their own x and y coordidante which determine their position on the maze. The enemies have has the state defeated. When a player travels to a enemy, they must combat the enemy, where a number is randomly selected from 0-9, and if it is higher then maze difficulty their state is set as defeated. However, if it loses then the player loses health, if this health reaches 0 then they have to travel back to the start cell to regain all their health. 

Class:
Quadrant - This class is used to represen the quadrant which divide the maze.

pyagame - This file contains the code which is used to create the maze. It creates customizable and random mazes using the create maze function. 

mazeBO - This file is the main executable file. On this file it contains the search pathign algorithms for the player agent. The objective functions for each player style as well as the optimization function calls. The methods for how a player can naviagate and operate across the maze can also be found here.

Output:
The mazeBO method is used to run the program. This will output graphs to help gather an understanding of the results. In order to progress to the next graph, the current graph will need to be closed. You save the graphs which have been produced.