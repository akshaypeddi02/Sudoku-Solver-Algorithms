#linear programming
from pulp import *

#simulated annealing
import random
import numpy
import math 
from random import choice
import statistics 

#genetic
#import numpy
#import random
def linear_programming():
    #source: https://towardsdatascience.com/sudoku-solver-linear-programming-approach-using-pulp-c520cec2f8e8
    puzzleToSolve =  [[8, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 3, 6, 0, 0, 0, 0, 0],
                    [0, 7, 0, 0, 9, 0, 2, 0, 0],
                    [0, 5, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 0, 0, 4, 5, 7, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 3, 0],
                    [0, 0, 1, 0, 0, 0, 0, 6, 8],
                    [0, 0, 8, 5, 0, 0, 0, 1, 0],
                    [0, 9, 0, 0, 0, 0, 4, 0, 0]]

    # print sudoku problem
    print("Sudoku Problem")
    for r in range(len(puzzleToSolve)):
        if r == 0 or r == 3 or r == 6:
            print("+-------+-------+-------+")
        for c in range(len(puzzleToSolve[r])):
            if c == 0 or c == 3 or c ==6:
                print("| ", end = "")
            if puzzleToSolve[r][c] != 0:
                print(puzzleToSolve[r][c], end = " ")
            else:
                print(end = "  ")
            if c == 8:
                print("|")
    print("+-------+-------+-------+")

    # A list of strings from 1 to 9 is created
    Sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # The Vals, Rows and Cols sequences all follow this form
    Vals = Sequence
    Rows = Sequence
    Cols = Sequence

    # SquareBoxes list with the row and column index of each square
    squareBoxes =[]
    for i in range(3):
        for j in range(3):
            squareBoxes += [[(Rows[3*i+k],Cols[3*j+l]) for k in range(3) for l in range(3)]]
            
    # Define Problem       
    prob = LpProblem("Sudoku Problem",LpMinimize)

    # Creating a Set of Variables
    choices = LpVariable.dicts("Choice",(Vals,Rows,Cols),0,1,LpInteger)

    # Added arbitrary objective function
    prob += 0, "Arbitrary Objective Function"

    # Setting Constraints
    # 1. A constraint ensuring that only one value can be in each square is created
    for r in Rows:
        for c in Cols:
            prob += lpSum([choices[v][r][c] for v in Vals]) == 1, ""

    # 2, 3, 4. The row, column and square constraints are added for each value
    for v in Vals:
        for r in Rows:
            prob += lpSum([choices[v][r][c] for c in Cols]) == 1,""
            
        for c in Cols:
            prob += lpSum([choices[v][r][c] for r in Rows]) == 1,""

        for b in squareBoxes:
            prob += lpSum([choices[v][r][c] for (r,c) in b]) == 1,""
                            
    # 5. The starting numbers in sudoku problem are entered as constraints                
    for r in range(len(puzzleToSolve)):
        for c in range(len(puzzleToSolve[r])):
            value = puzzleToSolve[r][c]
            if value != 0:
                prob += choices[value][r + 1][c + 1] == 1,""
                
    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # print out sudoku solution
    print("\nSudoku Solution")
    for r in Rows:
        if r == 1 or r == 4 or r == 7:
            print("+-------+-------+-------+")
        for c in Cols:
            for v in Vals:
                if choices[v][r][c].varValue == 1:               
                    if c == 1 or c == 4 or c == 7:
                        print("| ", end = "")
                    print(v, end = " ")
                    
                    if c == 9:
                        print("|")
    print("+-------+-------+-------+")


def simulated_annealing():
    startingSudoku = """
                    024007000
                    600000000
                    003680415
                    431005000
                    500000032
                    790000060
                    209710800
                    040093000
                    310004750
                """

    sudoku = numpy.array([[int(i) for i in line] for line in startingSudoku.split()])

    def PrintSudoku(sudoku):
        print("\n")
        for i in range(len(sudoku)):
            line = ""
            if i == 3 or i == 6:
                print("---------------------")
            for j in range(len(sudoku[i])):
                if j == 3 or j == 6:
                    line += "| "
                line += str(sudoku[i,j])+" "
            print(line)

    def FixSudokuValues(fixed_sudoku):
        for i in range (0,9):
            for j in range (0,9):
                if fixed_sudoku[i,j] != 0:
                    fixed_sudoku[i,j] = 1
        
        return(fixed_sudoku)

    # Cost Function    
    def CalculateNumberOfErrors(sudoku):
        numberOfErrors = 0 
        for i in range (0,9):
            numberOfErrors += CalculateNumberOfErrorsRowColumn(i ,i ,sudoku)
        return(numberOfErrors)

    def CalculateNumberOfErrorsRowColumn(row, column, sudoku):
        numberOfErrors = (9 - len(numpy.unique(sudoku[:,column]))) + (9 - len(numpy.unique(sudoku[row,:])))
        return(numberOfErrors)


    def CreateList3x3Blocks ():
        finalListOfBlocks = []
        for r in range (0,9):
            tmpList = []
            block1 = [i + 3*((r)%3) for i in range(0,3)]
            block2 = [i + 3*math.trunc((r)/3) for i in range(0,3)]
            for x in block1:
                for y in block2:
                    tmpList.append([x,y])
            finalListOfBlocks.append(tmpList)
        return(finalListOfBlocks)

    def RandomlyFill3x3Blocks(sudoku, listOfBlocks):
        for block in listOfBlocks:
            for box in block:
                if sudoku[box[0],box[1]] == 0:
                    currentBlock = sudoku[block[0][0]:(block[-1][0]+1),block[0][1]:(block[-1][1]+1)]
                    sudoku[box[0],box[1]] = choice([i for i in range(1,10) if i not in currentBlock])
        return sudoku

    def SumOfOneBlock (sudoku, oneBlock):
        finalSum = 0
        for box in oneBlock:
            finalSum += sudoku[box[0], box[1]]
        return(finalSum)

    def TwoRandomBoxesWithinBlock(fixedSudoku, block):
        while (1):
            firstBox = random.choice(block)
            secondBox = choice([box for box in block if box is not firstBox ])

            if fixedSudoku[firstBox[0], firstBox[1]] != 1 and fixedSudoku[secondBox[0], secondBox[1]] != 1:
                return([firstBox, secondBox])

    def FlipBoxes(sudoku, boxesToFlip):
        proposedSudoku = numpy.copy(sudoku)
        placeHolder = proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]]
        proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]] = proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]]
        proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]] = placeHolder
        return (proposedSudoku)

    def ProposedState (sudoku, fixedSudoku, listOfBlocks):
        randomBlock = random.choice(listOfBlocks)

        if SumOfOneBlock(fixedSudoku, randomBlock) > 6:  
            return(sudoku, 1, 1)
        boxesToFlip = TwoRandomBoxesWithinBlock(fixedSudoku, randomBlock)
        proposedSudoku = FlipBoxes(sudoku,  boxesToFlip)
        return([proposedSudoku, boxesToFlip])

    def ChooseNewState (currentSudoku, fixedSudoku, listOfBlocks, sigma):
        proposal = ProposedState(currentSudoku, fixedSudoku, listOfBlocks)
        newSudoku = proposal[0]
        boxesToCheck = proposal[1]
        currentCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], currentSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], currentSudoku)
        newCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], newSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], newSudoku)
        # currentCost = CalculateNumberOfErrors(currentSudoku)
        # newCost = CalculateNumberOfErrors(newSudoku)
        costDifference = newCost - currentCost
        rho = math.exp(-costDifference/sigma)
        if(numpy.random.uniform(1,0,1) < rho):
            return([newSudoku, costDifference])
        return([currentSudoku, 0])


    def ChooseNumberOfItterations(fixed_sudoku):
        numberOfItterations = 0
        for i in range (0,9):
            for j in range (0,9):
                if fixed_sudoku[i,j] != 0:
                    numberOfItterations += 1
        return numberOfItterations

    def CalculateInitialSigma (sudoku, fixedSudoku, listOfBlocks):
        listOfDifferences = []
        tmpSudoku = sudoku
        for i in range(1,10):
            tmpSudoku = ProposedState(tmpSudoku, fixedSudoku, listOfBlocks)[0]
            listOfDifferences.append(CalculateNumberOfErrors(tmpSudoku))
        return (statistics.pstdev(listOfDifferences))


    def solveSudoku (sudoku):
        f = open("demofile2.txt", "a")
        solutionFound = 0
        while (solutionFound == 0):
            decreaseFactor = 0.99
            stuckCount = 0
            fixedSudoku = numpy.copy(sudoku)
            PrintSudoku(sudoku)
            FixSudokuValues(fixedSudoku)
            listOfBlocks = CreateList3x3Blocks()
            tmpSudoku = RandomlyFill3x3Blocks(sudoku, listOfBlocks)
            sigma = CalculateInitialSigma(sudoku, fixedSudoku, listOfBlocks)
            score = CalculateNumberOfErrors(tmpSudoku)
            itterations = ChooseNumberOfItterations(fixedSudoku)
            if score <= 0:
                solutionFound = 1

            while solutionFound == 0:
                previousScore = score
                for i in range (0, itterations):
                    newState = ChooseNewState(tmpSudoku, fixedSudoku, listOfBlocks, sigma)
                    tmpSudoku = newState[0]
                    scoreDiff = newState[1]
                    score += scoreDiff
                    #print(score)
                    f.write(str(score) + '\n')
                    if score <= 0:
                        solutionFound = 1
                        break

                sigma *= decreaseFactor
                if score <= 0:
                    solutionFound = 1
                    break
                if score >= previousScore:
                    stuckCount += 1
                else:
                    stuckCount = 0
                if (stuckCount > 80):
                    sigma += 2
                if(CalculateNumberOfErrors(tmpSudoku)==0):
                    PrintSudoku(tmpSudoku)
                    break
        f.close()
        return(tmpSudoku)

    solution = solveSudoku(sudoku)
    #print(CalculateNumberOfErrors(solution))
    print("\nSOLUTION:")
    PrintSudoku(solution)


if __name__ == "__main__":
    print("Begin Linear Programming: \n\n")
    #linear_programming()
    print("End Linear Programming: \n\n")
    
    print("begin simualted annealing\n")
    #simulated_annealing()
    print("\nend simulated annealing")