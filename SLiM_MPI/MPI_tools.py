
def task_dis(size,task_list):

	task_num=len(task_list)
	task_dis_list=[]
	for i in range(size):
		task_dis_list.append([])
	
	#print(str(task_dis_list))

	for i in range(len(task_list)):
		for j in range(size):
			if i%size==j:
				task_dis_list[j].append(task_list[i])

	#for i in range(size):
	#	print("************"+str(i)+"************")
	#	print(str(task_dis_list[i]))

	return task_dis_list


#testing code

if 1==0:
	size=12
	task_list=[{0,2},{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},{8,2},{9,2},\
				{10,2},{11,2},{12,2},{13,2},{14,2},{15,2},{16,2},{17,2},{18,2},{19,2},
				{20,2},{21,2},{22,2},{23,2},{24,2},{25,2},{26,2},{27,2},{28,2},{29,2},
				{30,2},{31,2},{32,2},{33,2},{34,2},{35,2},{36,2},{37,2},{38,2},{39,2}]
	
	task_dis_list=task_dis(size,task_list)
	print(str(task_dis_list))
	
	for i in range(size):
		print("************"+str(i)+"************")
		print(str(task_dis_list[i]))
