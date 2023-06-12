import yaml
import time
import pprint

f=open("mv/arcturus_Cijk_Ailk_Bjlk_SB.yaml.bak")
f_out=open('arcturus_Cijk_Ailk_Bjlk_SB.yaml', 'w')

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
            if (content_in[0][i][j]['SolutionIndex'] < 10):
                print("Updating [5]: SolutionIndex: ", content_in[0][i][j]['SolutionIndex'])
                content_out[0].append(content_in[0][i][j])
            
    if (i == 6): 
        content_out[0].append(content_in[0][i])

    if (i == 7):
        #print("i==7") 
        for j in range(0, len(content_in[0])):
            print("content_in[0][i][j][1][0]: ", content_in[0][i][j][1][0])
            #time.sleep(1)
            if (content_in[0][i][j][1][0] < 10):
                print("Updating [7]: SolutionIndex: ", content_in[0][i][j], ", iter=", j)
                content_out[0].append(content_in[0][i][j])
yaml.dump(content_out, f_out)
f_out.close()
f.close()

