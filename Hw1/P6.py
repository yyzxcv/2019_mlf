from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time as t
data = []

with open('hw1_6_train.dat', 'r') as f:
    lines = f.readlines()

for line in lines:
	l = line.split( )
	l.insert(0, 1.0)
	data.append(l);

def check(d, w):
	x = np.array(d[0:5], dtype=float)
	y = float(d[5])
	v = np.dot(x, w)
	if (v > 0):
		return 1.0
	else:
		return -1.0

n = []
def PLA(data):
	w = np.zeros(5)
	count = 0
	update = 1
	while update:
		for d in data:
			if check(d, w) != float(d[5]):
				count += 1
				w = w + 0.5 * float(d[5]) * np.array(d[0:5], dtype=float)
		update = 0
		for d in data:
			if check(d, w) != float(d[5]):
				update = 1
				break
	n.append(count)
	return count

update_sum = 0
tStart = t.time()

for i in range(0, 1126):
	shuffle(data)
	count = PLA(data)
	update_sum += count

tEnd = t.time()
avg_update = update_sum / 1126.0

print('It cost %f sec' %(tEnd - tStart))
print('Average number of updates %f' %(avg_update))

num = np.array(n)
arr = plt.hist(num, bins=20, facecolor='b', edgecolor='black')
plt.title('Average number of updates: %f' %(avg_update))
plt.xlabel('Number of updates')
plt.ylabel('Frequency')
for i in range(20):
    plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
#plt.show()
plt.savefig(fname = "P6.png", dpi=100)