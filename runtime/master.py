# coding=utf-8
from socket import *
from time import ctime
import threading
import time
import argparse

config = 1
old_config = 1
machines = []
epoch_in_machines = []
epoch_in_progress = 0
scaled = []
finishes = []


parser = argparse.ArgumentParser(description='PyTorch Model Parallel Master')
parser.add_argument('--world_size', type=int, help='world size')
args = parser.parse_args()

for i in range(args.world_size):
    epoch_in_machines.append(None)
    scaled.append(True)
    finishes.append(False)

clients = []

def read(socket_fuwu, epoch_lock):
    global epoch_in_machines
    global machines, scaled
    global epoch_in_progress
    while True:
        message_length = socket_fuwu.recv(1).decode('utf-8')
        if not message_length:
            continue
        message_length = int(message_length)
        recv_data = socket_fuwu.recv(message_length)
        #recv_data = socket_fuwu.recv(1024)
        epoch_lock.acquire()
        if recv_data:
            recv_data = recv_data.decode('utf-8')
            print("\n" + str(time.time()) + " " + recv_data)
            if recv_data.split(":")[1] == "tra":
                print('rank %d finishes %s in epoch %d' % (int(recv_data.split(":")[0]), 
                recv_data.split(":")[1], int(recv_data.split(":")[2])))
                epoch_in_machines[int(recv_data.split(":")[0])] = int(recv_data.split(":")[2])
                print(epoch_in_machines, epoch_in_progress)
            elif recv_data.split(":")[1] == "scale":
                print('rank %d finishes %s in epoch %d' % (int(recv_data.split(":")[0]), 
                recv_data.split(":")[1], int(recv_data.split(":")[2])))
                scaled[int(recv_data.split(":")[0])] = True
            elif recv_data.split(":")[1] == "fin":
                print('rank %d finishes training' % (int(recv_data.split(":")[0])))
                finishes[int(recv_data.split(":")[0])] = True

        epoch_lock.release()


def write(epoch_lock, config_lock):
    global config
    global old_config
    global epoch_in_machines
    global epoch_in_progress
    global machines
    while True:
        config_lock.acquire()
        if config != old_config:
            for each in machines:
                print("\n" + str(time.time()))
                each.send(("config:" + str(config)).encode('utf-8'))
            old_config = config
        config_lock.release()


        epoch_lock.acquire()
        flag = 1 # 是否允许所有进程进入下一个epoch
        for i in range(args.world_size): # 判断是否每一个进程都完成了当前epoch
            if epoch_in_machines[i] != epoch_in_progress:
                flag = 0
                break
        if flag == 1:
            # 给所有进程发信号
            for each in machines:
                each.send(str(1).encode('utf-8'))
            epoch_in_progress += 1
            print("\n" + str(time.time()) + ": allow all to next epoch")
            print(epoch_in_machines, epoch_in_progress)

        flag = 1
        for i in range(args.world_size):
            if finishes[i] == False:
                flag = 0
                break
        if flag == 1:
            # 给所有进程发信号
            for each in machines:
                each.send(str("finish").encode('utf-8'))

        epoch_lock.release()

def input_data(config_lock):
    global config, old_config, scaled
    while True:
        data=input('>')
        config_lock.acquire()
        if not str(data).isdigit():
            print("Error: invalid input!")
            config_lock.release()
            continue
        flag = True # allow scale
        for each in scaled:
            if not each:
                flag = False # last scaling not finished yet.
        if not flag:
            print("Last scaling not finished yet. Please wait.")
            config_lock.release()
            continue
        config = int(data)#.encode('utf-8')
        print(config, old_config, config==old_config)
        if not config==old_config:
            for i in range(args.world_size):
                scaled[i] = False
        config_lock.release()

tcp_socket_host = socket(AF_INET,SOCK_STREAM)

# 服务器端口回收操作（释放端口）
tcp_socket_host.setsockopt(SOL_SOCKET, SO_REUSEADDR, True)

# 2绑定端口
tcp_socket_host.bind(('',8080))

# 3监听  变为被动套接字
tcp_socket_host.listen(128)    #128可以监听的最大数量，最大链接数

epoch_lock = threading.Lock()
config_lock = threading.Lock()

t = threading.Thread(target=input_data, args=(config_lock,))
t.start()

# 给所有进程发信号以同步这些进程
t2=threading.Thread(target=write,args=(epoch_lock, config_lock)) # the thread to send new config to machines
t2.start()

# 4等待客户端连接
while True:
    socket_fuwu,addr_client=tcp_socket_host.accept()  #accept(new_socket,addr)
    print(socket_fuwu)
    print(addr_client)
    machines.append(socket_fuwu)

    t1=threading.Thread(target=read,args=(socket_fuwu, epoch_lock)) # 保持各GPU间训练过程同步
    t1.start()

#6服务套接字关闭
#socket_fuwu.close()    #服务器一般不关闭   此时服务端口因为需要一直执行所以也不能关闭

