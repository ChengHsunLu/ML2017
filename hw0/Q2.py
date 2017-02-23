import sys
from PIL import Image, ImageChops

argList = sys.argv
im1 = Image.open(argList[1])
im2 = Image.open(argList[2])

im1 = im1.convert('RGBA')
im2 = im2.convert('RGBA')

px1 = im1.load()
px2 = im2.load()


width, height = im1.size

for y in xrange(height):
	for x in xrange(width):
		if px1[x, y] == px2[x, y]:
			px2[x, y] = (0, 0, 0, 0)

im2.save("ans_two.png")

