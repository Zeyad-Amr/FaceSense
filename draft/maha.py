class FaceRecognition:

    def _init_(self):
        
        self.TRAIN_IMG_FOLDER = './images/TestDetected/'
        self.train_imgs_names = os.listdir(self.TRAIN_IMG_FOLDER)
        self.label_dict = defaultdict(int)
        # amira 1/doha 2/maha 3/mariam 4/mayar 5
        for name in self.train_imgs_names:
            label = name.split('_')[0]
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict) + 1

        self.width = 128
        self.height = 128
        self.mean_face = np.zeros((1,self.height * self.width))
        self.Proj_Training_Matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.weights = None
        self.best_threshold = None


    def pca(self):

        Training_Matrix   = np.ndarray(shape=(len(self.train_imgs_names), self.height * self.width), dtype=np.float64)
        for i in range(len( self.train_imgs_names)):

            img = cv2.imread(self.TRAIN_IMG_FOLDER +  self.train_imgs_names[i], cv2.IMREAD_GRAYSCALE)
            Training_Matrix[i,:] = np.array(img, dtype='float64').flatten()


        for i in Training_Matrix:
            f.mean_face = np.add(self.mean_face,i)
        self.mean_face = np.divide(self.mean_face,float(Training_Matrix.shape[0])).flatten()

        Normalised_Training_Matrix = np.ndarray(shape=(Training_Matrix.shape[0], Training_Matrix.shape[1]))
        for i in range(len(self.train_imgs_names)):
            Normalised_Training_Matrix[i] = np.subtract(Training_Matrix[i],self.mean_face)
        cov_matrix=np.cov(Normalised_Training_Matrix)
        cov_matrix = np.divide(cov_matrix,float(len(self.train_imgs_names)))  
        self.eigenvalues, self.eigenvectors, = np.linalg.eig(cov_matrix)

        sorted_ind = sorted(range(self.eigenvalues.shape[0]), key=lambda k: self.eigenvalues[k], reverse=True) 
        self.eigenvalues = self.eigenvalues[sorted_ind]
        self.eigenvectors = self.eigenvectors[sorted_ind]  

        # var_comp_sum = np.cumsum(self.eigenvalues)/sum(self.eigenvalues)
        # for num_comp in range(1, len(self.eigenvalues) + 1):
        #     if var_comp_sum[num_comp - 1] >= 0.9:
        #         break

        # reduced_eigvectors = np.array(self.eigenvectors[:num_comp]).transpose()
        # self.Proj_Training_Matrix = np.dot(Training_Matrix.transpose(),reduced_eigvectors)

        self.Proj_Training_Matrix = np.dot(Training_Matrix.transpose(),self.eigenvectors)
        self.Proj_Training_Matrix =  self.Proj_Training_Matrix.transpose()

        self.weights = np.array([np.dot( self.Proj_Training_Matrix,i) for i in Normalised_Training_Matrix])


    def threshold_range(self):

        euclidean_values = []
        for img in self.test_imgs_names:

            unknown_face = cv2.imread(self.TEST_IMG_FOLDER+img,0)        
            unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
            normalised_uface_vector = np.subtract(unknown_face_vector,self.mean_face)

            unknown_weights = np.dot(self.Proj_Training_Matrix, normalised_uface_vector)
            diff  = self.weights - unknown_weights
            Euclidean_dist = np.linalg.norm(diff, axis=1)
            index = np.argmin(Euclidean_dist)
            euclidean_values.append( Euclidean_dist[index])
        return( min(euclidean_values) , max(euclidean_values) )
    


    
    def train(self):
        # Load and preprocess the training images
        X_train = []
        y_train = []
        for name in self.train_imgs_names:
            img = cv2.imread(self.TRAIN_IMG_FOLDER + name, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.width, self.height))
            X_train.append(img.flatten())
            label = self.label_dict[name.split('_')[0]]
            y_train.append(label)
        print(y_train)
        # Perform PCA on the training images
        self.pca()

        # Project the training images onto the PCA subspace
        X_train_proj = np.dot(self.Proj_Training_Matrix, np.array(X_train).T).T

        # Train an SVM classifier on the projected training images
        self.classifier = SVC(kernel='linear', C=1, gamma='auto')
        self.classifier.fit(X_train_proj, y_train)

    def recognize(self, img_path):
        # Load and preprocess the test image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img_vec = np.array(img, dtype='float64').flatten()
        img_vec_norm = np.subtract(img_vec, self.mean_face)

        # Project the test image onto the PCA subspace
        img_proj = np.dot(self.Proj_Training_Matrix, img_vec_norm)

        # Predict the label of the test image using the k-NN classifier
        label = self.classifier.predict([img_proj])[0]

        # Return the name of the predicted person
        for name, label_int in self.label_dict.items():
            if label_int == label:
                return name
        return 'Unknown'