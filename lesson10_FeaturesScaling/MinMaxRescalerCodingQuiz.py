""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):    
    if max(arr) != min(arr):
        return [float(x - min(arr)) / (max(arr) - min(arr)) for x in arr]
    else:
        return error

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print(featureScaling(data))
