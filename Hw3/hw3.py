import numpy as np
import matplotlib.pyplot as plt
T = 2000
r1 = 0.01
r2 = 0.001

def import_data(fname):
	data = np.loadtxt(fname, dtype='f')
	X = data[:,:-1]
	Y = data[:,-1,np.newaxis]
	return X, Y

def gradient(w, x, y):
	e = np.exp(np.dot(x, w) * y)
	return ((1 / (1 + e) * -y * x)).mean(0)[:,np.newaxis]

def GD(x_train, y_train, x_test, y_test, sgd, eta):
	w = np.zeros((x_train.shape[-1], 1))
	ein, eout = [], []
	for t in range(T):
		if (sgd):
			g = gradient(w, x_train[t%1000][np.newaxis,:], y_train[t%1000][np.newaxis,:])
		else:
			g = gradient(w, x_train, y_train)
		w -= eta * g
		ein.append((np.sign(np.dot(x_train, w)) != y_train).mean())
		eout.append((np.sign(np.dot(x_test, w)) != y_test).mean())
	return ein, eout

def plot(name, gd, sgd, lr, E):
	plt.plot(gd, label='GD')
	plt.plot(sgd, label='SGD')
	plt.title(name + '  lr =' + str(lr))
	plt.xlabel('t')
	plt.ylabel(E)
	plt.legend()
	plt.savefig(fname = name+'_'+str(lr)+".png", dpi=200)
	plt.close('all')

X_train, Y_train = import_data("hw3_train.dat")
X_test, Y_test = import_data("hw3_test.dat")

sgd_in, sgd_out = GD(X_train, Y_train, X_test, Y_test, sgd=True, eta=r1)
gd_in, gd_out = GD(X_train, Y_train, X_test, Y_test, sgd=False, eta=r1)
plot('P7',gd_in, sgd_in, r1, 'Ein')
plot('P8', gd_out, sgd_out, r1, 'Eout')

sgd_in, sgd_out = GD(X_train, Y_train, X_test, Y_test, sgd=True, eta=r2)
gd_in, gd_out = GD(X_train, Y_train, X_test, Y_test, sgd=False, eta=r2)
plot('P7', gd_in, sgd_in, r2, 'Ein')
plot('P8', gd_out, sgd_out, r2, 'Eout')