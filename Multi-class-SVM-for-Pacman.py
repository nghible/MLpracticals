class MultiClassSVM:
    
    def __init__(self):
        
        print "Initialising"
        
    # General description of Multi-class Support Vector Machine:
    #
    # We train a Kx(D + 1) weights matrix, where K is the number of classes, and D is the dimensions
    # of the feature vector plus the bias dimension. This matrix will evaluate a feature vector 
    # and give out K scores that are associated with K classes. The class that has the highest 
    # score will be assigned to that feature vector. 
    #
    # The key thing here is how to train the weights matrix. We will use a formulation 
    # of the loss function stated here: http://cs231n.github.io/linear-classify/ or 
    # (Weston and Watkin 1999): https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf
    #
    # This formulation in the primal can be optimized using Gradient Descent.
    #
    # The loss, also called a hinge loss can be generally described as follows:
    #
    # The function add cost when the scores of the wrong class is close to score of the correct
    # class. When the score of the wrong class is differs from the correct class by more than
    # a defined margin, the function does not incur cost.
    #
    # Basically, the training process push to W matrix into assigning higher scores to the
    # correct class, and lower scores to the wrong class.
    #
    # The hinge loss require a hyperparameter which is the margin. The higher the margin the
    # more we want the scores of the wrong class differs from the correct class.
    #
    # First we code out the lost function, and then we optimize the loss using gradient descent
    
    # This is the loss function:
    
    def TotalLoss(self, X, y, W, margin = 1):
        
        # We calculate the score matrix:
        
        score_matrix = np.matmul(W, X.transpose()) # The score_matrix is 4x100
    
        # Every column of score_matrix is a score vector associated with each feature vector
        # and every score vector contains four scores, each associated with each class
    
        # Below is the vector of the scores of the correct class for each feature vector
    
        score_correct_class = score_matrix[y, np.arange(score_matrix.shape[1])]
    
        # Now we calculate the loss matrix:
    
        loss = np.maximum(0,(score_matrix - score_correct_class) + margin)
    
        # At this point, the loss matrix is 4x100 with each column correspond to each feature vector
        # and each row is the score correspond to each class
        
        # We now need to sum each column but drop the loss associated with the correct class
        # We drop those losses by assign their values to 0
    
        loss[y, np.arange(score_matrix.shape[1])] = 0
    
        # We average the sum of each column to get the data loss
    
        return np.mean(np.sum(loss, axis = 0))
        
        # This loss function is the data loss stated in the link: http://cs231n.github.io/linear-classify/
         
    
    #This is the training function:
    
    def TrainSVMClassifier(self, X, y, learning_rate = 0.001, iterations = 100, margin = 1):
        
        # Save useful information:
        
        number_of_classes = len(np.unique(y))
        number_of_dimensions = X.shape[1]
        number_of_samples = X.shape[0]
        
        # Initialize random weights matrix with mean 0 and 0.1 standard deviation
        
        self.W = np.random.normal(0, 0.1, (number_of_classes, number_of_dimensions))
        
        # This part to calculate inital loss of the random weights matrix is similar 
        # to the total loss function above:
        
        score_matrix = np.matmul(self.W, X.transpose())
        score_correct_class = score_matrix[y, np.arange(score_matrix.shape[1])]
        loss_matrix = np.maximum(0,(score_matrix - score_correct_class) + margin)
        loss_matrix[y, np.arange(score_matrix.shape[1])] = 0
        
        ##### The gradient of the loss function is as follows:
        #
        # 1. With respect to the weights of the correct class, we will count the total number of 
        # classes that does add to the loss function and scale the feature vector by that number.
        #
        # 2. With respect to the weights of the incorrect classes, if that incorrect class does contribute
        # to the loss then the gradient is just the feature vector, if that incorrect does not contribute
        # to the loss then the gradient is 0.
        #
        #####
        
        ## First, we calculate gradient with respect to the weights of the correct class:
        # This is to simulate the indicator function: any class that adds to the loss will be 
        # assign a value of 1:
        loss_matrix[loss_matrix > 0] = 1
        
        # Now we get the gradient matrix for the weights of the correct class:
        gradient_correct_class = - np.sum(loss_matrix, axis= 0) * X.transpose()
        
        # gradient_correct_class is 25x100 matrix of scaled feature vector, which is our gradient for correct class weights.
        
        ## Second, for the gradient of the incorrect class weights, the calculation is detailed
        # in the 'Update the incorrect class weights' part in the training 'for' loop below

        # Make a copy of the weight matrix in order to save the best weights matrix that minimize
        # the loss:
        self.W_best = np.copy(self.W)
        
        ##### The training 'for' loop:
        for a in np.arange(iterations):
    
            for i in np.arange(number_of_samples):
                
                #### Update the correct class weights:
                # Now to update the correct class weights with gradient_correct_class, we have to pick out for 
                # each vector the correct weights of the correct class and add them appropriately:
                self.W[y[i]] = self.W[y[i]] - learning_rate * gradient_correct_class[:,i]
                
                #### Update the incorrect class weights:
                # Notice count_add_loss's row index where the value is 1 tells us which weights vector 
                # to add the feature vector into. And each column index is associated with each feature vector
                #
                # So the plan is to create 100 matrix each corresponding to each feature vector, where
                # each matrix has the dimension of 4x25, with the feature vector at the appropriate row
                # to add in to weights that belongs to the incorrect class
                self.W = self.W - learning_rate * np.matmul(loss_matrix[:,[i]], X[[i]])
            
            # Save the best weight matrix
            if self.TotalLoss(X, y, self.W) <= self.TotalLoss(X, y, self.W_best):
                self.W_best = self.W
                
    
    def PredictMove(self, x):
        # This function accepts a KxD weight matrix, and a feature vector and return 
        # a number representing a move.
        return np.argmax(np.matmul(self.W_best, np.array(x).transpose()))
