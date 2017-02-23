
import sys
import numpy

argList = sys.argv
matrixA = [map(int, line.split(',')) for line in open(argList[1])]
matrixB = [map(int, line.split(',')) for line in open(argList[2])]

matrixA = numpy.asarray(matrixA)
matrixB = numpy.asarray(matrixB)

ans = numpy.dot(matrixA, matrixB)
ansSorted = numpy.sort(ans, axis=None)

numpy.savetxt('ans_one.txt', ansSorted, fmt='%d', delimiter=',')
