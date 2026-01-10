import tiktoken
def bubble_sort(list): 
    n = len(list) 
  
    # Traverse through all array elements 
    for i in range(n): 
  
        # Last i elements are already sorted
        for j in range(0, n-i-1): 
  
            # traverse the array from 0 to n-i-i+1
            # Swap if the element found is greater than the next element
            if list[j] > list[j+1] : 
                list[j], list[j + 1] = list[j + 1], list[j]
    
    return list

print(bubble_sort([100,99,88,77,4,7]))