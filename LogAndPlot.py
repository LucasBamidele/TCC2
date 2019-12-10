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
def main():
	from scipy.ndimage.filters import gaussian_filter
	import csv
	filename = 'saved_models/mymodel_ppo_1v1.csv'
	rewards = []
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			rewards.append(float(row[0]))
	sigma = 10
	saida_filtrada = gaussian_filter(rewards, sigma)
	# v1, v2 = 'Recompensas', 'Recompensas suavizadas'
	# plt.legend(handles=[v1, v2])

	# plt.title('Recom', figname)
	plt.xlabel('Episódios')
	plt.ylabel('Recompensa média')
	plt.plot(rewards, color = '#b3d1ff')
	plt.plot(saida_filtrada, color = 'C0')
	plt.savefig('saved_models/'+'testin2' + '.png')
	plt.clf()

if __name__ == '__main__':
	main()