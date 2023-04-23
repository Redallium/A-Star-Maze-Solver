# A-Star-Maze-Solver
Implementation of a graphical interface for solving mazes on surfaces of different topologies. Analysis of the effectiveness of various heuristics for the A* algorithm.
## Repository Structure

### `main.py`

GUI for modeling algorithms for finding paths in mazes on surfaces of different topologies using various heuristic functions.

### `MazeDataStructures.py`

Implementation of classes for storing maze nodes and correctly accessing them for various topologies

### `GeneticHeuristic.py`

Implementation of a genetic algorithm for finding the best heuristics.

### `EmpiricalCandidateRunner.py`

A program for calculating statistics for mazes of all topologies of predefined heuristics.

### `GraphAnalysis.py`

Code for counting the number of connectivity components from a given JSON file of vertical and horizontal walls in a maze.

### `PyPyVersion/`

A directory with similar scripts to run on the pypy interpreter.

## Interpreter Versions
### `PyPyVersion: `
Python 3.8.16 [PyPy 7.3.11]
### `Default Scripts: `
Python 3.11.0
