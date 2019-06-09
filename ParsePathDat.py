import sys  
import os

filepath = 'paths.dat'
if not os.path.isfile(filepath):
       print("File path {} does not exist. Exiting...".format(filepath))
       sys.exit()
      
        
dic = {}
with open(filepath) as fp:
       cnt = 0
       for line in fp:
           
           if cnt>=2:
               bag = line.split()
               
               if bag[3] == 'index,':
                   dic[bag[0]] = bag[-1]
           cnt += 1
                   
print(dic)