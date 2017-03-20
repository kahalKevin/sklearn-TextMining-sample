labeled = ""

label = []

with open('label.txt', 'r') as inFile:
        isi = inFile.readlines()
        for line in isi:
            line = line.split('\n')
            label.append(line[0])

with open('final.csv', 'r') as inFile:
        isi = inFile.readlines()
        i=-1
        for line in isi:
            line = line.split('\n')
            if i==-1:
                labeled += line[0] + ',label\n' 
            else:
                labeled += line[0] + ',' + label[i] + '\n'
            i=i+1

file = open("labeledall.csv","w") 
file.write(labeled)
file.close()