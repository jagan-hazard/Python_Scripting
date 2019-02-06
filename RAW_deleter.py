import os
import re
x=os.popen('dir /b').read()
arr=x.split()
#print (arr)
jpg=[]
jpg_pre=[]
nef=[]
nef_pre=[]
for i in arr:
	#print (i)
	if (re.findall(".*.jpg",i)):
		jpg.append(i)
		x=i.split('.')
		jpg_pre.append(x[0])

	if (re.findall(".*.NEF",i)):
		nef.append(i)
		x=i.split('.')
		nef_pre.append(x[0])


def intersection(list1,list2):
	list3=[value for value in list1 if value not in list2]
	return list3

out=intersection(nef_pre,jpg_pre)
#print (out)
for j in out:
	file_name=j+".NEF"
	cmd='del '+file_name
	#print (cmd)
	os.system(cmd)
	print("deleting",file_name,"is done")
