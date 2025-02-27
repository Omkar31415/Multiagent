# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        if len(foodList) == 0:
            return 999999

        foodDistances = [util.manhattanDistance(newPos, food) for food in foodList]
        closestFoodDist = min(foodDistances)
        secondClosestFoodDistances = [dist for dist in foodDistances if dist != closestFoodDist]
        if secondClosestFoodDistances:
            secondClosestFoodDist = min(secondClosestFoodDistances)
        else:
            secondClosestFoodDist = 9999999

        ghostDists = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if min(ghostDists) <= 1:
            return -999999 

        score = successorGameState.getScore()

        if action == Directions.STOP:
            score -= 10 

        if sum(newScaredTimes) > 0:
            score += 100 
            
        score += 10 / closestFoodDist    
        score += 1 / (secondClosestFoodDist + 1)
        
        #considering vertical movement if stuck in loop
        if closestFoodDist == secondClosestFoodDist:
            newX, newY = newPos
            oldX, oldY = currentGameState.getPacmanPosition()
            if newX != oldX: 
                score -= 1
            elif newY != oldY:
                score += 1
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman's turn
            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            else: #>=1 Ghost's turn
                return min_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex): #Pacman
            max_val = -9999999
            legal_actions = state.getLegalActions(agentIndex)

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % state.getNumAgents()
                max_val = max(max_val, minimax(successor, depth + 1 if next_agentIndex == 0 else depth, next_agentIndex))

            return max_val

        def min_value(state, depth, agentIndex): #Ghosts
            min_val = 99999999
            legal_actions = state.getLegalActions(agentIndex)

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % state.getNumAgents()
                min_val = min(min_val, minimax(successor, depth + 1 if next_agentIndex == 0 else depth, next_agentIndex))

            return min_val

        legal_actions = gameState.getLegalActions(0) #actions of pacman

        next_value = -99999999
        next_action = None

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)  # Depth 0, start with the first ghost (agentIndex 1)

            if value > next_value:
                next_value = value
                next_action = action

        return next_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState, self.depth, 0, -99999999, 9999999)[1]

    def alphabeta(self, state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None

        if agentIndex == 0:  # Pacman's turn
            return self.max_value(state, depth, agentIndex, alpha, beta)
        else:  # Ghost's turn
            return self.min_value(state, depth, agentIndex, alpha, beta)

    def max_value(self, state, depth, agentIndex, alpha, beta):
        max_val = -999999999
        next_action = None
        legal_actions = state.getLegalActions(agentIndex)
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            next_agentIndex = (agentIndex + 1) % state.getNumAgents()
            value, _ = self.alphabeta(successor, depth - 1 if next_agentIndex == 0 else depth, next_agentIndex, alpha, beta)
            if value > max_val:
                max_val = value
                next_action = action
            if max_val > beta:  # For autograder purpose
                return max_val, next_action
            alpha = max(alpha, max_val)
        return max_val, next_action

    def min_value(self, state, depth, agentIndex, alpha, beta):
        min_val = 99999999999
        legal_actions = state.getLegalActions(agentIndex)
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            next_agentIndex = (agentIndex + 1) % state.getNumAgents()
            value, _ = self.alphabeta(successor, depth - 1 if next_agentIndex == 0 else depth, next_agentIndex, alpha, beta)
            min_val = min(min_val, value)
            if min_val < alpha:  # For Autograder 
                return min_val, None
            beta = min(beta, min_val)
        return min_val, None
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, self.depth, 0)[1]

    def expectimax(self, state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None
        
        if agentIndex == 0:  # Pacman's turn
            return self.max_value(state, depth, agentIndex)
        else:  # Ghost's turn
            return self.exp_value(state, depth, agentIndex)

    def max_value(self, state, depth, agentIndex):
        max_val = -999999999
        next_action = None
        legal_actions = state.getLegalActions(agentIndex)
        
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            next_agentIndex = (agentIndex + 1) % state.getNumAgents()
            value, _ = self.expectimax(successor, depth - 1 if next_agentIndex == 0 else depth, next_agentIndex)
            if value > max_val:
                max_val = value
                next_action = action
        return max_val, next_action

    def exp_value(self, state, depth, agentIndex):
        exp_val = 0
        legal_actions = state.getLegalActions(agentIndex)
        
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            next_agentIndex = (agentIndex + 1) % state.getNumAgents()
            value, _ = self.expectimax(successor, depth - 1 if next_agentIndex == 0 else depth, next_agentIndex)
            exp_val += value * ( 1 / len(legal_actions))
        return exp_val, None
    
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I'm considering the following factors to bring the best out of the player (pacman)
    - Calculating the distance (Manhattan) to the every food pellet from all food pellets from current state and select the closest one (minimum manhattan distance)
    - Similarly calculating distance to closest super pellet which allows us to increase the chances of not getting caught by ghost.
    - Considering the nearest ghost to our current state which allows us to let player know to move away from ghost or else we penalize it.
    - Considering the scared timer of ghosts to determine whether to chase them or not.
    - We can add the final score to current score of gamestate.
    - We can calculate the final score by adding the current score with inverse of (min food distance+1), inverse of (min dist to superPellet+1) and total ghost scared timer. From all of this we need subtract the inverse of (minimum distance to ghost + 1) to get final score at last. by subtracting the minimum distance to ghost the player will know that going nearer to a ghost will result in failure rather than success.
    
    Here we are adding +1 to every variable because if we have no food left the minimum dist to food will be 0 and if we calculate 1/0 it becomes undefined and gets crashed. So, we add 1 to avoid such cases.
    """

    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList()
    currCapsule = currentGameState.getCapsules()
    currghostStates = currentGameState.getGhostStates()
    currScore = currentGameState.getScore()

    a=[]
    for food in currFood:
        a.append(util.manhattanDistance(currPos, food))
    minFoodDist = min(a) if a else 999999999
    
    b=[]
    for capsule in currCapsule:
        b.append(util.manhattanDistance(currPos, capsule))
    minCapsuleDist = min(b) if b else 99999999

    c=[]
    for i in currghostStates:
        c.append(i.scaredTimer)
    totalScaredTime = sum(c) if c else 99999999

    d=[]
    for j in currghostStates:
         if j.scaredTimer == 0:
                d.append(util.manhattanDistance(currPos, j.getPosition()))
    minGhostDist = min(d) if d else -999999999

    # Calculate the evaluation score
    newScore =  currScore + ( 1.0 / (minFoodDist + 1)) + ( 1.0 / (minCapsuleDist + 1)) + totalScaredTime - ( 1.0 / (minGhostDist+1))
    return newScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction