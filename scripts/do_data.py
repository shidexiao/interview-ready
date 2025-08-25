arr = [
    {"variableName": "submodel_v0_2_0", "variableValue": 23.6104},
    {"variableName": "submodel_v0_1_1", "variableValue": 43.2071},
    {"variableName": "submodel_v0_0_2", "variableValue": 45.5615},
    {"variableName": "submodel_v0_2_1", "variableValue": 21.8138},
    {"variableName": "submodel_v0_0_0", "variableValue": 37.5909},
    {"variableName": "submodel_v0_1_0", "variableValue": 52.1532},
    {"variableName": "submodel_v0_0_1", "variableValue": 42.9912},
]
n_d = {}
for a in arr:
    n_d[a["variableName"]] = a["variableValue"]
print(n_d)
