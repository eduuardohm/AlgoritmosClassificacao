import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

scaler = StandardScaler()

def selectDataset(id):
	if id == 1:
		dataset_name = 'Iris'
		nClasses = 3
		data, target = load_iris(return_X_y=True)		# Iris | OPENML: ID 61 | 4 features | 150 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Iris\n")

		return (data, target, nClasses, dataset_name)
	elif id == 2:
		dataset_name = 'Digits'
		nClasses = 10
		data, target = load_digits(return_X_y=True)		# Digits | OPENML: ID 28 | 64 features | 1797 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Digits\n")

		# Digits é um dataset de imagem

		return (data, target, nClasses, dataset_name)
	elif id == 3:
		dataset_name = 'Wine'
		nClasses = 3
		data, target = load_wine(return_X_y=True)		# Wine | OPENML: ID 187 | 13 features | 178 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Wine\n")

		return (data, target, nClasses, dataset_name)
	elif id == 4:
		dataset_name = 'Breast Cancer'
		nClasses = 2
		data, target = load_breast_cancer(return_X_y=True)	# Breast cancer | UCI: ID 17 | 30 features | 569 instances

		data = scaler.fit_transform(data)

		print("Dataset selecionado: Breast Cancer\n")

		return (data, target, nClasses, dataset_name)
	elif id == 5:
		dataset_name = 'Scene'
		data = arff.loadarff('datasets/scene.arff')		# Scene | OPENML: ID 312 | 294 features	| 2407 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)
		dataset = dataset.drop(dataset.columns[294:299], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2

		# Scene é um dataset de imagem
		
		print("Dataset selecionado: Scene\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 6:
		dataset_name = 'Madelon'
		data = arff.loadarff('datasets/madelon.arff')		# Madelon | OPENML: ID 1485 | 500 features | 4400 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2
		
		print("Dataset selecionado: Madelon\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 7:
		dataset_name = 'Hiva Agnostic'
		data = arff.loadarff('datasets/hiva_agnostic.arff')		# Hiva Agnostic | OPENML: ID 1039 | 1000 features | 4228 instances
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2
		
		print("Dataset selecionado: Hiva Agnostic\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 8:
		dataset_name = 'Musk Version 1'
		dataset = pd.read_csv('datasets/musk1.data', header=None)		# Musk (Version 1) | UCI Machine Learning Repository | 165 features | 476 Instances
		dataset = dataset.drop(dataset.columns[[0, 1]], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()
		dataset_ref = np.array(dataset_ref)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2

		print(dataset_ref)
		
		print("Dataset selecionado: Musk (Version 1)\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)
	elif id == 9:
		# (Dataset 1 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClasses = 3
		
		mu_11 = 5
		mu_12 = 0
		mu_21 = 15
		mu_22 = 5
		mu_31 = 18
		mu_32 = 14

		sigma_11 = 81
		sigma_12 = 9
		sigma_21 = 9
		sigma_22 = 100
		sigma_31 = 25
		sigma_32 = 36

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))

		synthetic = np.vstack((class1, class2, class3))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		
		data_ref = np.concatenate((refClass1, refClass2, refClass3))

		return [synthetic, data_ref, nClasses, "Pimentel2013 - Data 1"]
	elif id == 10:
		# (Dataset 2 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClasses = 3
		
		mu_11 = 0
		mu_12 = 0
		mu_21 = 30
		mu_22 = 0
		mu_31 = 10
		mu_32 = 25

		sigma_11 = 100
		sigma_12 = 100
		sigma_21 = 49
		sigma_22 = 49
		sigma_31 = 16
		sigma_32 = 16

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))

		synthetic = np.vstack((class1, class2, class3))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		
		data_ref = np.concatenate((refClass1, refClass2, refClass3))

		return [synthetic, data_ref, nClasses, "Pimentel2013 - Data 2"]
	elif id == 11:
		# (Dataset 3 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClasses = 3
		
		mu_11 = 0
		mu_12 = 0
		mu_21 = 15
		mu_22 = 3
		mu_31 = 15
		mu_32 = -3

		sigma_11 = 100
		sigma_12 = 4
		sigma_21 = 100
		sigma_22 = 4
		sigma_31 = 100
		sigma_32 = 4

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))

		synthetic = np.vstack((class1, class2, class3))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		
		data_ref = np.concatenate((refClass1, refClass2, refClass3))

		return [synthetic, data_ref, nClasses, "Pimentel2013 - Data 3"]
	elif id == 12:
		# (Dataset 4 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClasses = 3
		
		mu_11 = 0
		mu_12 = 0
		mu_21 = 15
		mu_22 = 0
		mu_31 = -15
		mu_32 = 0

		sigma_11 = 16
		sigma_12 = 16
		sigma_21 = 16
		sigma_22 = 16
		sigma_31 = 16
		sigma_32 = 16

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))

		synthetic = np.vstack((class1, class2, class3))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		
		data_ref = np.concatenate((refClass1, refClass2, refClass3))

		return [synthetic, data_ref, nClasses, "Pimentel2013 - Data 4"]
	elif id == 13:
		# Teste sintetico - Data 1
		n1 = 200
		n2 = 100
		
		n = n1 + n2
		
		nClasses = 2
		
		mu_11 = 5
		mu_12 = 0
		mu_21 = 15
		mu_22 = 5

		sigma_11 = 9
		sigma_12 = 3
		sigma_21 = 3
		sigma_22 = 10

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		data_ref = np.concatenate((refClass1, refClass2))

		return [synthetic, data_ref, nClasses, "Teste sintetico - Data 1"]
	elif id == 14:
		# Dataset Sintético 2 elipses

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClasses = 2
		
		mu_11 = 0
		mu_12 = 0

		mu_21 = 50
		mu_22 = 0
		
		sigma_11 = 4
		sigma_12 = 20
		
		sigma_21 = 4
		sigma_22 = 20
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClasses, "2 elipses VER."]
	elif id == 15:
		# Dataset Sintético 2 elipses 3D

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClasses = 2
		
		mu_11 = 0
		mu_12 = 0
		mu_13 = 0

		mu_21 = 20
		mu_22 = 0
		mu_23 = 50
		
		sigma_11 = 4
		sigma_12 = 20
		sigma_13 = 4
		
		sigma_21 = 4
		sigma_22 = 20
		sigma_23 = 4
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		z1 = np.random.normal(mu_13, sigma_13, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		z2 = np.random.normal(mu_23, sigma_23, n2)

		class1 = np.column_stack((x1, y1, z1))
		class2 = np.column_stack((x2, y2, z2))
		
		# class1 = np.column_stack((x1, z1))
		# class2 = np.column_stack((x2, z2))

		synthetic = np.vstack((class1, class2))
		synthetic = scaler.fit_transform(synthetic)

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClasses, "2 elipses 3D"]
	elif id == 16:
		n = 210 
		n_classes = 3 

		mu = np.array([
			[0, 0],
			[15, 0],
			[10, 25] 
		])
		sigma = np.array([
			[1, 10],
			[5, 1],
			[3, 5] 
			])
		sigma = [np.diag(sigma[i]) for i in range(3)]

		X_class1 = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/3))
		X_class2 = np.random.multivariate_normal(mu[1], sigma[1], size=int(n/3))
		X_class3 = np.random.multivariate_normal(mu[2], sigma[2], size=int(n/3))

		X = np.vstack((X_class1, X_class2, X_class3))

		y_class1 = np.ones(int(n/3)) * 1
		y_class2 = np.ones(int(n/3)) * 2
		y_class3 = np.ones(int(n/3)) * 3
		y = np.hstack((y_class1, y_class2, y_class3))

		synthetic = X
		data_ref = y

		synthetic = scaler.fit_transform(synthetic)

		return [synthetic, data_ref, n_classes, "Spherical Gaussian Distribution - 3 Classes"]
	elif id == 17:
		n = 210				# Dataset Sintético Relação linear
		nClasses = 3

		mu = np.array([
			[0, 0],
			[30, 0],
			[10, 25] 
			])
		sigma = np.array([
			[1, 10],
			[5, 1],
			[3, 5] 
			])
		sigma = [np.diag(sigma[i]) for i in range(3)]

		X_linear = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClasses))
		y_linear = np.repeat(1, int(n/nClasses))
		for i in range(1, nClasses):
			X_linear = np.vstack((X_linear, np.random.multivariate_normal(mu[i], sigma[i], size=int(n/nClasses))))
			y_linear = np.hstack((y_linear, np.repeat(i+1, int(n/nClasses))))

		synthetic = X_linear
		data_ref_ = y_linear

		synthetic = scaler.fit_transform(synthetic)

		return [synthetic, data_ref_, nClasses, "Relação linear"]
	elif id == 18:
		n = 200 
		n_classes = 3

		mu1 = np.array([0, 0])
		mu2 = np.array([30, 0])
		mu3 = np.array([10, 25])

		coef = 0.5
		base = 2

		X_class1 = mu1 + coef * np.random.rand(n // n_classes, 2)
		X_class2 = mu2 + coef * (base**np.random.rand(n // n_classes, 1)) * np.random.rand(n // n_classes, 2)
		X_class3 = mu3 + coef * (base**np.random.rand(n // n_classes, 1)) * np.random.rand(n // n_classes, 2)

		synthetic = np.vstack((X_class1, X_class2, X_class3))
		synthetic = scaler.fit_transform(synthetic)

		print(synthetic)

		data_ref = np.concatenate((np.repeat(1, n // n_classes), np.repeat(2, n // n_classes), np.repeat(3, n // n_classes)))

		return [synthetic, data_ref, n_classes, "Relação Exponencial"]
	
	elif id == 19:
		dataset_name = 'Ionosphere'
		dataset = pd.read_csv('datasets/ionosphere.data', header=None)		# Ionosphere | UCI Machine Learning Repository | 34 features | 351 instances
		
		dataset_ref = dataset.iloc[:, -1].map({'g': 1, 'b': 0}).to_numpy()
		dataset = dataset.drop(dataset.columns[-1], axis=1)
		dataset = scaler.fit_transform(dataset)
		nClasses = 2

		print(dataset_ref)
		
		print("Dataset selecionado: Ionosphere\n")
		
		return (dataset, dataset_ref, nClasses, dataset_name)

	elif id == 20:
		# Sonar
		dataset_name = 'Sonar'
		dataset = pd.read_csv("datasets/sonar.data", sep=",", header=None)
		dataset_ref = dataset.iloc[:, -1].map({'R': 1, 'M': 0}).to_numpy()
		
		dataset = dataset.drop(dataset.columns[-1], axis=1).to_numpy()
		dataset = scaler.fit_transform(dataset)

		nClasses = 2

		return [dataset, dataset_ref, nClasses, "Sonar Dataset"]
	
	elif id == 21:
		# Zoo
		dataset_name = 'Zoo'
		dataset = pd.read_csv("datasets/zoo.data", sep=",", header=None)
		
		dataset_ref = (dataset.iloc[:, -1] - 1).to_numpy()

		columns = dataset.columns[[0, -1]] # Remove animal name and class (0, -1 columns)
		
		dataset = dataset.drop(columns, axis=1).to_numpy()
		dataset = scaler.fit_transform(dataset)

		nClasses = 7

		return [dataset, dataset_ref, nClasses, "Zoo Dataset"]
	
	elif id == 22:
		# Heart Statlog
		dataset = pd.read_csv("datasets/heart-statlog.dat", sep=" ", header=None)
		dataset_ref = (dataset.iloc[:, -1] - 1).to_numpy()
		
		columns = [dataset.columns[-1]]

		dataset = dataset.drop(columns, axis=1).to_numpy()
		dataset = scaler.fit_transform(dataset)

		nClasses = 2
		

		return [dataset, dataset_ref, nClasses, "Heart Statlog Dataset"]