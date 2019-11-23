import matplotlib.pyplot as plt
def saveCSV(rewards, figname='default'):
	import csv
	with open(figname + '.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',')
		for reward in rewards:
			csvwriter.writerow([reward])
def plotRewards(y, x=None, figname='default'):
	if(x):
		plt.plt(x,y)
	else:
		plt.plot(y)
	# plt.grid()
	# plt.title('average reward for ', figname)
	# plt.xlabel('time')
	# plt.ylabel('reward')
	plt.savefig(figname + '.png')
	plt.clf()
# def p():
# 	pass