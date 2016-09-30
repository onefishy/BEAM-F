import numpy as np
from biconcave_opt_utils import *

#-----------------  concave optimization -------------------#
def bi_concave_opt(N, weights, 
                   X, y, 
                   bi_concave_iters, conf_opt_iters, 
                   prob_estimates, 
                   eps, reg, thresh, out_put=True, seed=0):
    '''
    BayesCG Algorithm
    -----------------
    Input:          
    N: #classes
    X, y: training set to estimate weights
    bi_concave_iters, conf_opt_iters: inner, outter loop max iterations
    prob_estimates: class probability model (previously trained)
    eps, reg, thresh: perturbation constant, regularization, convergence threshhold
    seed: type of initial confusion matrix

    Output:
    classifiers: array of classifiers found
    classifier_weights: weights for classifiers found
    perf: macro f-score performance on data set
    '''
    
    # Initial Gain Matrix = I_{n} <=> classifier h = argmax \eta
    #--------------   weighted initialization
    if seed == 0:
        G_diag = [weights[n] for n in xrange(N)]
        G_0 = np.diag(G_diag)

    #--------------   balanced initialization
    elif seed == 1:
        G_0 = np.eye(N)
    
    #--------------   random initialization
    else:
        G_0 = 20 * np.random.random((N, N))
    
    # Get the initial confusion matrix form the classifer 
    # corresponding to initial gain matrix
    Conf_t = compute_conf(np.copy(G_0), 
                         np.copy(prob_estimates), 
                         np.copy(y))
        
    # Find performance of initial confusion matrix 
    perf = eval_conf(np.copy(Conf_t))

    if out_put == True:
        print 'initial:', perf, '\n'
    
    outer_obj = []
    
    
    classifiers = np.zeros((bi_concave_iters * conf_opt_iters + 1, N, N))
    classifier_weights = np.zeros(bi_concave_iters * conf_opt_iters + 1)
 
    classifiers[0, : , :] = G_0
    classifier_weights[0] = 1
    
    classifier_index = 1

    for s in xrange(bi_concave_iters):
        u_s = compute_u(np.copy(Conf_t), eps) # Optimize u for fixed conf mat
        inner_obj = []
        
        
        for t in xrange(conf_opt_iters): # Optimize conf mat for fixed u
            obj = compute_biconcave_obj(np.copy(Conf_t), u_s, reg)
            
            if out_put == True:
                print s, ',', t, ':', perf, ', ', obj
            
            if t == conf_opt_iters - 1:
                if len(outer_obj) >= 3:
                    outer_obj.pop(0)
                    outer_obj.append(obj)
                else:
                    outer_obj.append(obj)
            else:
                if len(inner_obj) >= 3:
                    inner_obj.pop(0)
                    inner_obj.append(obj)
                    if abs(inner_obj[0] - inner_obj[1]) <= thresh and \
                       abs(inner_obj[1] - inner_obj[2]) <= thresh:
                        if len(outer_obj) >= 3:
                            outer_obj.pop(0)
                            outer_obj.append(obj)
                        else:
                            outer_obj.append(obj)
                        break                                            
                else:
                    inner_obj.append(obj)
                    
            # Get the updated classifer (Update gain matrix G_t)
            G_t = compute_conf_grad(np.copy(Conf_t), np.copy(u_s), eps, reg) 

            # Find new confusion matrix
            Conf_new = compute_conf(np.copy(G_t), 
                                   np.copy(prob_estimates), 
                                   np.copy(y))
            
            # Update the confusion matrix 
            max_perf = -1
            Conf_old = Conf_t
            step_size = 0
            for i in xrange(100):
                l = i * 0.01
                Conf_temp = l * Conf_new + (1 - l) * Conf_old
                perf_temp = compute_biconcave_obj(Conf_temp, u_s, reg)
                if perf_temp > max_perf:
                    #print 'accept'
                    max_perf = perf_temp
                    Conf_t = Conf_temp
                    step_size = l                    
                #else:
                    #print 'reject'


            classifiers[classifier_index, :, :] = G_t
            classifier_weights[:classifier_index] = classifier_weights[:classifier_index] * (1 - step_size)
            classifier_weights[classifier_index] = step_size            
            classifier_index = classifier_index + 1
            
            # Find perf
            perf = eval_conf(np.copy(Conf_t))

        if len(outer_obj) >=3 and \
            abs(outer_obj[0] - outer_obj[1]) <= thresh and \
            abs(outer_obj[1] - outer_obj[2]) <= thresh:
            break 
      
    G_final = compute_conf_grad(np.copy(Conf_t), np.copy(u_s), eps, reg)            
    Conf_final = compute_conf(G_final, prob_estimates, y)

    classifiers = classifiers[:classifier_index, :, :]
    classifier_weights = classifier_weights[:classifier_index]
        

    if out_put == True:
        print 'final: ', str(perf), '\n'

    return classifiers, classifier_weights, perf
    

def seed_biconcave(N, weights, 
                   X, y, 
                   bi_concave_iters, conf_opt_iters, 
                   prob_estimates, 
                   eps, reg, thresh, out_put=None):
                                    
    best_classifier = 0
    best_weights = 0
    best_perf = -1
    
    for i in range(5):
        classifiers, classifier_weights, perf = bi_concave_opt(N, weights, 
                                                               X, y, 
                                                               bi_concave_iters, conf_opt_iters, 
                                                               prob_estimates, 
                                                               eps, reg, thresh, 
                                                               out_put=out_put, seed=i)
                                                                  
        if perf > best_perf:
            best_classifier = np.copy(classifiers)
            best_weights = np.copy(classifier_weights)
            best_perf = perf

    return best_classifier, best_weights