#!/usr/bin/python
"""
Finds the number of blocks common between two files in ipfs
python ipfs_file_similarity.py '/Users/Kunal/Downloads/checkpoints/images' similarity_matrix.csv 262144
"""

import os, sys
import subprocess
import numpy as np

def add_file_ipfs(path, chunker_size):
	files_ipfs_path = {}
	for dirs in os.walk(path).next()[1]:
		fullpath = path + '/' + dirs
		print fullpath
		command = 'ipfs add --chunker=size-' + chunker_size + ' -rq ' + fullpath + ' | tail -n 1'
		p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
		ipfs_name, _ = p.communicate()
		files_ipfs_path[dirs] = ipfs_name.strip('\n')
	return files_ipfs_path

def get_file_hashes(files_ipfs_path):
	files_hashes = {}
	for keys in files_ipfs_path:
		print keys
		files_hashes[keys] = file_hash(files_ipfs_path[keys])
	# print files_hashes
	return files_hashes

def file_hash(ipfs_hash):
	hashes = []
	hashes_to_iterate = [ipfs_hash]
	while len(hashes_to_iterate) != 0:
		print len(hashes_to_iterate)
		command = 'ipfs object links ' + hashes_to_iterate[0]
		if hashes_to_iterate[0] not in hashes:
			hashes.append(hashes_to_iterate[0])
		hashes_to_iterate.remove(hashes_to_iterate[0])
		p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
		for line in p.stdout:
			new_hash = line.split(' ')[0]
			if new_hash not in hashes_to_iterate:
				hashes_to_iterate.append(new_hash)
	return hashes

def compare_files(files_hashes, file_names):
	w, h = len(files_hashes), len(files_hashes)
	similarity_matrix = [[0 for x in range(w)] for y in range(h)] 
	for i in range(1,17):
		key1_hashes = files_hashes[file_names[i-1]]
		for j in range(1,17):
			key2_hashes = files_hashes[file_names[j-1]]
			similarity_matrix[i-1][j-1] = (len(list(set(key1_hashes) & set(key2_hashes))) * 100)/len(key1_hashes)
	# print similarity_matrix
	return similarity_matrix

def save_similarity_matrix(similarity_matrix, file_names, output_file):
	np_matrix = np.asarray(similarity_matrix)
	header = ','.join(file_names)
	np.savetxt(output_file, np_matrix, delimiter=",", header=header)

def main():
	if len(sys.argv) != 4:
		print "USAGE: python ipfs_file_similarity.py <directory> <output.csv> <chunker size>"
		return
	path = sys.argv[1]
	output_file = sys.argv[2]
	chunker_size = sys.argv[3]
	files_ipfs_path = add_file_ipfs(path, chunker_size)
	# files_ipfs_path = {'images9': 'QmNxKCn4M8wA9noejhcurCQRkxMpkNpx8tfzoFVBt9Vxoy', 'images8': 'QmQnDdwmBYxta3gg54RiBewM8zFDKdpRcjkVsBoxTByHjn', 'images5': 'QmSdqFoKdAvj14K2woDSfPUMMmMcPi8xurczEeT1bLzUmB', 'images4': 'QmPLfTGEQb2XURbDt74EJCdrXopgN7qJYNFjZepUTztpNX', 'images7': 'QmNbDKjL2VfwX5bvCtoCs2Zb74KK26RfVUzqc65Z9SqeDK', 'images6': 'QmfZALohrBzyPQyRufdgFafndNBghW1c1PsrjFwrNJyUaa', 'images1': 'QmTAFJvt6Pk9nR1grQwu7f4vQh3BbhpLhXRebBpdqu1WwV', 'images3': 'QmXv3tQkASPx2C7FJ8jTrERgN7Dv79Rh8mkG4GasprE4bQ', 'images2': 'QmTuQftXTgb73X5yfiGLtP2twK1tp28e4fd9PHDboBzPYB', 'images11': 'QmUpeJVdAszZtbjuqt2X6d3tpvmWnCm2bvvDcQpWiUFLit', 'images10': 'QmWko1TvUBBJwZSWFuy3Y9ChzVBSoaboS6nsfPPVEZTiJw', 'images13': 'QmcARfzSLX6cGinMZ7t3qBZuspjggTL2qj8RBxDX2crEh9', 'images12': 'QmQdNPt1rNLq86vKkNQ34gPcUHBW3W3Ej7eZeDVP4YW6TN', 'images15': 'QmRMRoDo6jwffzSTUQ4fGdykhhtRrm8xkaA2DygxXYR1Fv', 'images14': 'QmT7hgr7ucXcTGzxvgAjbcKTMffNaJHF3om4einvsAYCvd', 'images16': 'QmZMUFo238MGpj7bRbQCSSb8UkCsjXQE95b1VDmLkDCvv3'}
	files_hashes = get_file_hashes(files_ipfs_path)
	file_names = ['images' + str(i) for i in range(1,17)]
	similarity_matrix = compare_files(files_hashes, file_names)
	# similarity_matrix = [[100, 53, 41, 38, 38, 37, 38, 39, 38, 37, 32, 31, 32, 24, 24, 24], [52, 100, 62, 47, 46, 46, 46, 46, 46, 46, 40, 34, 34, 24, 23, 23], [42, 64, 100, 58, 56, 55, 57, 53, 54, 53, 46, 40, 41, 27, 26, 27], [37, 46, 55, 100, 59, 57, 57, 52, 53, 51, 43, 39, 39, 27, 27, 27], [37, 45, 53, 59, 100, 66, 60, 56, 56, 53, 45, 42, 41, 28, 27, 26], [35, 44, 51, 55, 64, 100, 58, 54, 53, 51, 44, 40, 41, 26, 26, 25], [38, 46, 55, 57, 61, 61, 100, 62, 58, 56, 47, 41, 42, 26, 27, 26], [38, 46, 52, 52, 57, 57, 63, 100, 64, 60, 48, 42, 42, 28, 28, 26], [38, 48, 53, 55, 58, 57, 59, 66, 100, 71, 53, 46, 46, 30, 29, 28], [37, 47, 52, 52, 54, 54, 57, 60, 70, 100, 57, 46, 48, 31, 28, 28], [34, 43, 47, 47, 49, 49, 52, 52, 56, 61, 100, 56, 57, 40, 36, 36], [32, 35, 40, 41, 44, 43, 44, 44, 47, 48, 54, 100, 63, 40, 36, 36], [33, 35, 41, 41, 43, 44, 44, 44, 47, 49, 55, 62, 100, 52, 40, 37], [27, 27, 30, 31, 32, 30, 30, 31, 34, 34, 42, 44, 56, 100, 54, 49], [25, 24, 27, 28, 29, 28, 28, 30, 30, 30, 35, 37, 40, 50, 100, 59], [24, 23, 27, 28, 27, 27, 27, 27, 28, 28, 34, 36, 37, 44, 58, 100]]
	save_similarity_matrix(similarity_matrix, file_names, output_file)

if __name__ == '__main__':
	main()