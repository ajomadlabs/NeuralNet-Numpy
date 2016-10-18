import numpy as np


#Sigmoid Function
def sig_func(x,deriv=False):
	if deriv == True:
		return x*(1-x)

	return 1/(1+np.exp(-x))



#Input Dataset
X = np.array([[0,0,1],
	      [0,1,1],
	      [1,0,1],
	      [1,1,1]])

#Output Dataset
Y = np.array([[0],
	      [1],
	      [1],
	      [0]])


#Seed to make deterministic
np.random.seed(1)

#Synapsis
synap0 = 2 * np.random.random((3,4)) - 1
synap1 = 2 * np.random.random((4,1)) - 1

#Training step
for i in range(600000):
	
	#Prediction step
	layer0 = X
	layer1 = sig_func(np.dot(layer0, synap0))
	layer2 = sig_func(np.dot(layer1, synap1))
	
	#Error Rate
	layer2_error = Y - layer2
	
	#Printing Error Rate
	if i % 10000 == 0:
		print("Error : " + str(np.mean(np.abs(layer2))))

	layer2_delta = layer2_error * sig_func(layer2, deriv=True)
	layer1_error = layer2_delta.dot(synap1.T)
	layer1_delta = layer1_error * sig_func(layer1,deriv=True)

	synap1 = layer1.T.dot(layer2_delta)
	synap0 = layer0.T.dot(layer1_delta)

print("Output After Training Session")
print(layer2)	

	

