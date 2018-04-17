#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
        
        predictions is a list of predicted targets that come from your regression
        ages is the list of ages in the training set
        net_worths is the actual value of the net worths in the training set.
        
        
    """    
    
    ### your code goes here
    import numpy as np
    
    diff = np.abs(np.array(net_worths) - np.array(predictions))
    num = int(np.fix(len(diff)*(10/100)))
    
    index_sort_diff = diff.argsort(axis = 0)[-num:][::-1]    

    ages = np.delete(ages,index_sort_diff[:,0],axis=0)
    net_worths = np.delete(net_worths,index_sort_diff[:,0],axis=0)
    diff = np.delete(diff,index_sort_diff[:,0],axis=0)
    
    cleaned_data = []
    for printer in range(len(ages)):
        age = ages[printer,0]
        net_worth = net_worths[printer,0]
        adiff = diff[printer,0]
        acleaned_data = (age,net_worth,adiff)
        cleaned_data.append(acleaned_data)

    return cleaned_data

