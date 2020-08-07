# coding=utf-8
import torch
import time, os, json
from torch import nn

import argparse
import copy
import importlib
from itertools import combinations


def in_gpu(key, rank, model, stage_to_rank_map, module_to_stage_map):
	"""
	判断某个param是否在rank这个GPU里。
	"""
	global_index = 0
	#local_index = 0
	for (stage, inputs, outputs) in model:
		if rank not in stage_to_rank_map[str(module_to_stage_map[global_index])]:
			global_index += 1
			continue
		if key in stage.state_dict():
			return stage_to_rank_map[str(module_to_stage_map[global_index])]
		global_index += 1
		#local_index += 1
	return -1

def data_size(i, j):
	"""
	GPU i 与 GPU j之间的通信量。
	"""
	#send_list
	total_size = 0
	#print(i, j, each_mapping)
	if communication_dependencies[i][j] == 1:
		if use_gpu_num == src_num:
			index = i #多变少，需要send的param在第i个子模型上
			#print(stage_to_rank_map_src)
			index = rank_to_stage_map_src[i]
		else:
			index = rank_to_stage_map_src[i] #少变多，需要send的param在第each_mapping.index(i)个subm上
		statedict = model_src[index][0].state_dict()
		for eachparam in send_list[i]:
			if j in send_list[i][eachparam]:
				ans = 1
				for each in statedict[eachparam].size():
					ans *= each
				total_size += ans
	elif communication_dependencies[i][j] == -1:
		if use_gpu_num == src_num:
			index = rank_to_stage_map_src[j] #多变少，需要receive的param在第j个子模型上
		else:
			index = rank_to_stage_map_src[j] #少变多，需要receive的param在src(02->012)
		statedict = model_src[index][0].state_dict()
		for eachparam in receive_list[i]:
			if j in receive_list[i][eachparam]:
				ans = 1
				for each in statedict[eachparam].size():
					ans *= each
				total_size += ans
	return total_size * 4 #float32

start_time = time.time()
"""
#少变多
module_src = 'models.resnext101_32x16d.gpus=2'
module_dst = 'models.resnext101_32x16d.gpus=3'
config_src = './models/resnext101_32x16d/gpus=2/my_mp_config.json'
config_dst = './models/resnext101_32x16d/gpus=3/my_mp_config.json'

#多变少
module_src = 'models.resnext101_32x16d.gpus=3'
module_dst = 'models.resnext101_32x16d.gpus=2'
config_src = './models/resnext101_32x16d/gpus=3/my_mp_config.json'
config_dst = './models/resnext101_32x16d/gpus=2/my_mp_config.json'
"""
#少变多
module_src = 'image_classification.models.vgg16.gpus=2'
module_dst = 'image_classification.models.vgg16.gpus=4'
config_src = './image_classification/models/vgg16/gpus=2/mp_conf.json'
config_dst = './image_classification/models/vgg16/gpus=4/hybrid_conf_2-2.json'
"""
#多变少
module_src = 'models.vgg16.gpus=8'
module_dst = 'models.vgg16.gpus=4'
config_src = './models/vgg16/gpus=8/hybrid_conf.json'
config_dst = './models/vgg16/gpus=4/hybrid_conf.json'
"""
json_config_file_src = json.load(open(config_src, 'r'))
stage_to_rank_map_src = json_config_file_src.get("stage_to_rank_map", None)
module_to_stage_map_src = json_config_file_src.get("module_to_stage_map", None)
json_config_file_dst = json.load(open(config_dst, 'r'))
stage_to_rank_map_dst = json_config_file_dst.get("stage_to_rank_map", None)
module_to_stage_map_dst = json_config_file_dst.get("module_to_stage_map", None)

module = importlib.import_module(module_src)
full_model = module.full_model() # 2s!

loss_fn = nn.MSELoss()
module = importlib.import_module(module_src)
model_src = module.model(loss_fn) # 2s!
module = importlib.import_module(module_dst)
model_dst = module.model(loss_fn)


src_num = int(module_src.split("=")[1])
dst_num = int(module_dst.split("=")[1])

bandwidth = 100000000
#backward_time = [0.1 for i in range(src_num)]
backward_time = 0.01

use_gpu_num = max(src_num, dst_num)
device_map = [i for i in range(use_gpu_num)]

all_mappings = combinations(device_map, min(src_num, dst_num))
#少变多：index为old rank；多变少：index为new rank

best_mapping = None
current_min_time = None
all_send_lists = {}
all_receive_lists = {}
for each_mapping in all_mappings:
	#print("Mapping: ", each_mapping)
	tmp_stage_to_rank_map_src = copy.deepcopy(stage_to_rank_map_src)
	tmp_stage_to_rank_map_dst = copy.deepcopy(stage_to_rank_map_dst)
	rank_to_stage_map_src, rank_to_stage_map_dst = {}, {}
	if use_gpu_num == src_num:
		# 多变少
		for eachstage in stage_to_rank_map_dst:
			for i in range(len(stage_to_rank_map_dst[eachstage])):
				tmp_stage_to_rank_map_dst[eachstage][i] = each_mapping[
				stage_to_rank_map_dst[eachstage][i]]
				rank_to_stage_map_dst[each_mapping[
				stage_to_rank_map_dst[eachstage][i]]] = int(eachstage)
		for eachstage in stage_to_rank_map_src:
			for i in range(len(stage_to_rank_map_src[eachstage])):
				rank_to_stage_map_src[stage_to_rank_map_src[eachstage][i]] = int(eachstage)

	else:
		# 少变多
		for eachstage in stage_to_rank_map_src:
			for i in range(len(stage_to_rank_map_src[eachstage])):
				tmp_stage_to_rank_map_src[eachstage][i] = each_mapping[
				stage_to_rank_map_src[eachstage][i]]
				rank_to_stage_map_src[each_mapping[
				stage_to_rank_map_src[eachstage][i]]] = int(eachstage)
		for eachstage in stage_to_rank_map_dst:
			for i in range(len(stage_to_rank_map_dst[eachstage])):
				rank_to_stage_map_dst[stage_to_rank_map_dst[eachstage][i]] = int(eachstage)

	
	#找出每个GPU的send dict和receive dict
	#communication_dependencies[i][j]==-1代表i接收j的参数；1代表发送
	communication_dependency = [0 for i in range(use_gpu_num)]
	communication_dependencies = [copy.deepcopy(communication_dependency) for i in range(use_gpu_num)]

	send_list = []
	for rank in range(use_gpu_num):
		tmp_send_dict = {}
		new_index = 0
		send_cnt = 0
		for (stage, inputs, outputs) in model_dst:
			for each_parameter, v in stage.state_dict().items():
				# param即将出现在stage里，如果这个参数现在在rank上
				if in_gpu(each_parameter, rank, model_src, 
					tmp_stage_to_rank_map_src, module_to_stage_map_src) != -1:
					#即将出现在这些GPU上
					dst_list = tmp_stage_to_rank_map_dst[str(module_to_stage_map_dst[new_index])]
					#print(each_parameter, "now in ", rank, "will be in rank", dst_list)
					#那么这些GPU中的哪些现在没有这个参数呢？
					dist_send_list = []
					for eachdst in dst_list:
						if eachdst not in rank_to_stage_map_src or rank_to_stage_map_src[eachdst] != rank_to_stage_map_src[rank]:
							dist_send_list.append(eachdst)
					if len(dist_send_list) > 0:
						sent = False
						for each_send_dict in send_list:
							if each_parameter in each_send_dict:
								sent = True
								break
						if not sent:
							tmp_send_dict[each_parameter] = dist_send_list
							for eachdst in dist_send_list:
								communication_dependencies[rank][eachdst] = 1
								communication_dependencies[eachdst][rank] = -1
							send_cnt += 1
							#print(each_parameter, rank, "send to ", dist_send_list)
			new_index += 1
		reversed_send_dct = reversed(list(tmp_send_dict.keys()))
		tmp_send_dict = dict([(key, tmp_send_dict[key]) for key in reversed_send_dct])
		send_list.append(tmp_send_dict)
		print("rank", rank, "send", send_cnt)
	
	receive_list = []
	for rank in range(use_gpu_num):
		tmp_receive_dict = {}
		receive_list.append(tmp_receive_dict)
	idx = 0
	for each_send_dict in send_list:
		for each_param in each_send_dict:
			for eachdst in each_send_dict[each_param]:
				if each_param in receive_list[eachdst]:
					receive_list[eachdst][each_param].append(idx)
				else:
					receive_list[eachdst][each_param] = [idx]
		idx += 1
	all_send_lists[each_mapping] = send_list
	all_receive_lists[each_mapping] = receive_list


	#print("communication_dependencies:", communication_dependencies)

	#计算每个GPU的通信情况。少变多，空GPUready time是0；
	#ready time[i][j]: GPU i可以和GPU j通信了的时间，i ready时j可能还没ready
	gpu_ready_time = [0 for i in range(use_gpu_num)]
	ready_time = [copy.deepcopy(gpu_ready_time) for i in range(use_gpu_num)]
	if use_gpu_num == dst_num:
		# 少变多
		for i in reversed(range(use_gpu_num)):
			for j in range(use_gpu_num):
				# 如果i和j之间需要通信，且i==j+1，ready time = max(结束backward的时间,j开始back的时间)
				#i!=j+1则ready time=发送方
				if communication_dependencies[i][j] == 1:
					if j in each_mapping and i in each_mapping and each_mapping.index(i) == each_mapping.index(j) + 1:
						ready_time[i][j] = (src_num - i) * backward_time #i finishes backward
					else:
						ready_time[i][j] = (src_num - i - 1) * backward_time#i begins backward
				elif communication_dependencies[i][j] == -1:
					# i 接收完训练所需数据后就可以接受其他的参数了
					ready_time[i][j] = (src_num - i - 1) * backward_time#i begins backward
				if ready_time[i][j] < 0:
					ready_time[i][j] = 0
	else:
		# 多变少
		for i in reversed(range(use_gpu_num)):
			for j in range(use_gpu_num):#for j in range(dst_num):
				if communication_dependencies[i][j] == 1:
					if i == j + 1:
						ready_time[i][j] = (src_num - i) * backward_time#i finishes backward
					else:
						ready_time[i][j] = (src_num - i - 1) * backward_time#i begins backward
				elif communication_dependencies[i][j] == -1:
					ready_time[i][j] = (src_num - i - 1) * backward_time#i begins backward
	#print("ready_time", ready_time)
	# update
	# 从后向前，看每一块GPU和前面的GPU的全部通信。顺便更新涉及到的GPU的ready time。同时更新curtime。
	cur_time = 0
	stage_num = len(model_src)
	endtime = backward_time * stage_num
	last_gpu_time = 0
	for i in reversed(range(use_gpu_num)):
		if i in rank_to_stage_map_src:
			cur_gpu_time = (stage_num - rank_to_stage_map_src[i] - 1) * backward_time
			last_gpu_time = cur_gpu_time
		else:
			cur_gpu_time = last_gpu_time
		for j in reversed(range(i)):
			if communication_dependencies[i][j] == 0:
				#cur_gpu_time = (stage_num - i) * backward_time
				continue
			#print("compute data size: ", i, j, communication_dependencies[i][j])
			comm_ready_time = max(ready_time[i][j], ready_time[j][i], cur_gpu_time)
			comm_time = data_size(i, j) / bandwidth
			cur_gpu_time = comm_ready_time + comm_time
		if i in rank_to_stage_map_src:
			cur_gpu_time = max(cur_gpu_time, (stage_num - rank_to_stage_map_src[i]) * backward_time)
		endtime = max(endtime, cur_gpu_time)
		# 此处可以剪枝
		#print("i: ", i, " endtime", endtime)

	#calculate time
	if current_min_time is None:
		current_min_time = endtime
		best_mapping = each_mapping
	else:
		if endtime < current_min_time:
			current_min_time = endtime
			best_mapping = each_mapping
	print("\n")

print("best mapping: ", best_mapping)
# parameter_list.json # after runtime is ready
# 同一个param不会接收两次！
send_receive_list = []
for i in range(use_gpu_num):
	comm_dict = {}
	for eachparam in all_send_lists[best_mapping][i]:
		comm_dict[eachparam] = [i, all_send_lists[best_mapping][i][eachparam]]
	for eachparam in all_receive_lists[best_mapping][i]:
		comm_dict[eachparam] = [all_receive_lists[best_mapping][i][eachparam], i]
	send_receive_list.append([(k,comm_dict[k]) for k in reversed(sorted(comm_dict.keys()))])

send_receive_list = json.dumps(send_receive_list)
try:
	with open("./image_classification/models/vgg16/gpus=4/send_receive_list.json", 
		"w", encoding='utf-8') as f:
		f.write(send_receive_list + "\n")
		print("^_^ write success")
except Exception as e:
	print("write error==>", e)

finish_time = time.time()
print("mapping time: ", finish_time - start_time)
