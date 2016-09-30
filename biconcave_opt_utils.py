import numpy as np

#------------  compute objective function ---------------#
def compute_biconcave_obj(Conf, u, reg):
	'''
    Computes the biconcave objective function
    '''	
    # Number of classes
    N = Conf.shape[0]
    P_Conf = np.array([np.sum(Conf[n, :]) + np.sum(Conf[:, n]) for n in xrange(N)])
    return 2 * (u.dot(np.sqrt(np.diag(Conf)))) - (u**2).dot(P_Conf) - reg * np.linalg.norm(P_Conf)

#------------  compute performance for conf matrix  --------------#
def compute_u(C, eps): # will be updated to use f-score
    '''
    Optimizes u for a fixed confusion matrix C
    '''
    
    # Adding a small perturbation helps avoid loc optima
    C += eps
    # Number of classes
    N = C.shape[0]
    
    u = np.zeros(N)
    
    # Update u using closed form expression
    for n in xrange(N):
        u[n] = np.sqrt(C[n, n]) / (np.sum(C[n, :]) + np.sum(C[:, n]))
        
    return u

#------------  evaluate performance of classifier  --------------#
def eval_classifier(prob_estimates, G, X, y):
    '''
    Evaluate performance of classifier in terms of f-score   
    '''

    Conf = compute_conf(G, prob_estimates, y)
    perf = eval_conf(Conf)
    
    return perf, np.round(Conf, decimals=2), Conf * X.shape[0]
	
#------------  compute performance for conf matrix  --------------#
def eval_conf(C): # updated to use f-score
    '''
    Evaluate performance of confusion matrix in terms of f-score 
    '''       
    # Number of classes
    N = C.shape[0]
    perf = 0
    cl = 0
    
    for n in xrange(N):
        if (np.sum(C[n, :]) + np.sum(C[:, n])) > 0:
            perf = perf + (C[n, n] * 2.) / (np.sum(C[n, :]) + np.sum(C[:, n]))
            cl += 1
        #else:
            #print 'invalid confusion matrix at class', n
            #sys.exit('invalid confusion matrix')
    if cl == 0:
        return cl
    else:
        return perf / (1. * cl)
	
 
#------------  gradient of performance measure  -------------#
def compute_conf_grad(C, u, eps, reg): # updated to use f-score
    '''
    Finds gradient of f-score at confusion matrix C
    '''
    # Number of classes
    N = C.shape[0]
    C += eps

    grad = np.zeros(C.shape)
    
    for n in xrange(N):
        P = np.zeros(C.shape)
        P[n, :] = 1
        P[:, n] = 1
        P[n, n] = 2
        grad = grad - u[n] * u[n] * P
        grad[n, n] = grad[n, n] + u[n] / np.sqrt(2. * C[n, n])

    return grad - reg * 2 * C
	
#-----------------  predict labels -------------------#
def predict_labels(G, eta):
    '''
    Outputs prediction of classifier with gain matrix G \in \R^{n,n} when eta is known
    '''

    M = eta.shape[1]
    
    # optimal class labels
    labels = np.zeros(M)

    for m in xrange(M): # for each data point
        eta_m = eta[:, m] # get eta for the m^{th} point
        t = G.dot(eta_m) # get a row vector with (g_{y})'*\eta_{x}
        indx = np.argmax(t) # weighted argmax
        labels[m] = indx # label for m-th data point       

    return labels

#-----------------  compute confusion matrix -------------------#	
def compute_conf(G, eta, true_label):    
    '''
    Given a deterministic classifier (corresponding to the gain matrix G) 
    computes its confusion matrix C
    '''

    # Number of classes
    N = G.shape[0]
    # Number of instances
    M = len(true_label)
    
    # Initialize nxn Confision matrix
    C = np.zeros(G.shape)
    
    # Get prediction for the given classifier (gain matrix G)
    pred_label = predict_labels(G, eta)
    pred_label = pred_label.reshape((pred_label.shape[0], 1))

    # Update the entries of confusion matrix
    for i in xrange(N):
        for j in xrange(N):
            comp_label = pred_label[true_label == i]
            C[i, j] = len(comp_label[comp_label == j])

    C = C / (M * 1.)
    return C

def compute_rand_conf(classifiers, classifier_weights, eta, true_label):    
    '''
    Given a randomized classifier (corresponding to the gain matrices classifiers, 
    and weights classifier_weights) computes its confusion matrix C
    '''
    # Number of classifiers, classes
    n = classifiers.shape[0]
    N = classifiers.shape[1]
    
    # Expected conf matrix
    C = np.zeros((N,N))
    for i in range(n):
        conf = compute_conf(classifiers[i, :, :].reshape(N, N), eta, true_label)
        C =  C + classifier_weights[i] * conf
        
    return C
