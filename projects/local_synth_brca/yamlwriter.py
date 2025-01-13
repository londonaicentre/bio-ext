import yaml

data = {
    "name": "tester",
    "projectid":"1234"
}

a = 1234

filename = "test"
with open(f"{filename}.yaml","w") as f:
    yaml.dump(data,f,sort_keys=False)
    print("written successfully")
    
    


def func1():
    return a