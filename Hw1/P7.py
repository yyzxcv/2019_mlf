from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time as t
data = []
test = []

with open('hw1_7_train.dat', 'r') as f:
    lines = f.readlines()

for line in lines:
	l = line.split( )
	l.insert(0, 1.0)
	data.append(l);
	
with open('hw1_7_test.dat', 'r') as f:
    lines = f.readlines()

for line in lines:
	l = line.split( )
	l.insert(0, 1.0)
	test.append(l);

size = len(test)

def check(d, w):
	x = np.array(d[0:5], dtype=float)
	y = float(d[5])
	v = np.dot(x, w)
	if (v > 0):
		return 1.0
	else:
		return -1.0

n = []
def PLA_pocket(data):
	w = np.zeros(5)
	w_pocket = np.zeros(5)
	min_error = len(data)
	count = 0
	update = 1
	while update and count <= 100:
		for d in data:
			if check(d, w) != float(d[5]):
				count += 1
				w = w + 0.5 * float(d[5]) * np.array(d[0:5], dtype=float)
			
			#Greedy
			err = 0
			for d in data:
				if check(d, w) != float(d[5]):
					err += 1				
			if err < min_error:
				min_error = err
				w_pocket = w

		update = 0
		for d in data:
			if check(d, w) != float(d[5]):
				update = 1
				break

	error = 0.0
	for d in test:
		if check(d, w_pocket) != float(d[5]):
			error += 1.0

	n.append(float(error)/float(size))
	return(float(error)/float(size))
	
total_error_rate = 0.0
tStart = t.time()
for i in range(0, 1126):
	print(i)
	shuffle(data)
	total_error_rate += PLA_pocket(data)
tEnd = t.time()

avg_error_rate = total_error_rate / 1126.0
print('It cost %f sec' %(tEnd - tStart))
print('Average error rate: %f' %(avg_error_rate))
#0.11098223801065654

num = np.array(n)
arr = plt.hist(num, bins=20, facecolor='b', edgecolor='black')
plt.title('Average error rate: %f' %(avg_error_rate))
plt.xlabel('Error rate')
plt.ylabel('Frequency')
for i in range(20):
    plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
#plt.show()
plt.savefig(fname = "P7.png", dpi=100)