import json
c = {1.6:400}
b = {"a":c}
a = {"1":b}
print(a)
with open('Dictionary.json', 'w') as file:
    json.dump(a, file)

with open('Dictionary.json') as openFile:
    d = json.load(openFile)
    
print(d)