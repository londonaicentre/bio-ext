import yaml

with open("utils/config_dash.yaml", "r") as yamlfile:
    content = yaml.safe_load(yamlfile)

mylist = content["escapekwlist"]
mylist = sorted(mylist)

content["escapekwlist"] = mylist

with open("utils/config_dash.yaml", "w") as yamlfile:
    yamlfile.write(
        yaml.dump(
            content,
            default_flow_style=False,
        )
    )
