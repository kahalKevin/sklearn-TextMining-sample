from collections import Counter

counts = Counter()
template = {}
alltemplate = []
header = []

csv_ready = ""

with open('200sms.txt', 'r') as inFile:
        isi = inFile.readlines()
        hasil = []
        num = 1
        for line in isi:
            line = line.split()
            counts.update(hasil for hasil in line)
            words = [item for item in line]
            hasil += words
        for isi in hasil :
        	template[isi] = 0
        for fitur in counts :   
            header.append(fitur)
        counts.clear()


with open('200sms.txt', 'r') as inFile:
        isi = inFile.readlines()
        hasil = []
        num = 1
        for line in isi:
			copyTemplate = template.copy()
			line = line.split()
			for _word in line:
				copyTemplate[_word] = copyTemplate[_word] + 1
			alltemplate.append(copyTemplate)

# INSERTION PROCESS

i = 0
for fitur in header:
    if(i==len(header)-1):
        csv_ready += fitur + '\n'
    else:
        csv_ready += fitur + ','
        i=i+1


for filled in alltemplate:
    i = 0
    for fitur in header:
        if(i==len(header)-1):
            csv_ready += str(filled[fitur]) + '\n'
        else:
            csv_ready += str(filled[fitur]) + ','
            i=i+1

# print csv_ready

file = open("final.csv","w") 
file.write(csv_ready)
file.close()