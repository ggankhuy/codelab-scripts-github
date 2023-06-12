#   Minimize tensile yaml script by reducing number for solution indexes.
#   YAML FORMAT:
#   [0]
#   [1]
#   [2]
#   [3]
#   [4]
#   [5] - kernel configuration parameters along with Solution Index.
#   [5][0]....[5][500+]
#   [6]
#   [7] - kernel performance data along with Solution index
#   [7][0]....[7][14000+]

import yaml
import time
import pprint

f=open("mv/arcturus_Cijk_Ailk_Bjlk_SB.yaml.bak")
f_out=open('arcturus_Cijk_Ailk_Bjlk_SB.yaml', 'w')
CONFIG_MAX_SOLUTION_INDEX=10
if not f:
    print("Failed to open...")
    quit(1)

content_in = list(yaml.load_all(f, Loader=yaml.Loader))
content_out = [[]]

for i in range(0, 8):
    if (i < 5): 
        content_out[0].append(content_in[0][i])

    if (i == 5):
        print("i==5") 
        for j in range(0, len(content_in[0])):
            if (content_in[0][i][j]['SolutionIndex'] < CONFIG_MAX_SOLUTION_INDEX):
                print("Updating [5]: SolutionIndex: ", content_in[0][i][j]['SolutionIndex'])
                content_out[0].append(content_in[0][i][j])
            
    if (i == 6): 
        content_out[0].append(content_in[0][i])

    if (i == 7):
        for j in range(0, len(content_in[0])):
            print("content_in[0][i][j][1][0]: ", content_in[0][i][j][1][0])
            if (content_in[0][i][j][1][0] < CONFIG_MAX_SOLUTION_INDEX):
                print("Updating [7]: SolutionIndex: ", content_in[0][i][j], ", iter=", j)
                content_out[0].append(content_in[0][i][j])
yaml.dump(content_out, f_out)
f_out.close()
f.close()

