# cat /server/vps_monitor.py
# coding:utf-8
'''
监控系统CPU，内存，磁盘，带宽，系统用户登录
'''

# python lib
import json
import psutil
import time
import requests
import traceback
import threading
from datetime import datetime


class DingDingNotification(object):
    def __init__(self):
        with open('/server/server_id.json', 'rt') as f:
            d = json.load(f)
        self.server_id = d
        self.webhook = 'https://oapi.dingtalk.com/robot/send?access_token=967b3e365ebb04834417471bf3e8cc4b1481f27de9599b529b041ef4b9da3682'

    def send(self, content):
        exception_tb = traceback.format_exc()
        if exception_tb.strip() != 'NoneType: None':
            content = f'===={self.server_id}====\n' + content + '\n========TRACEBACK ========:\n' + exception_tb
        else:
            content = f'===={self.server_id}====\n' + content
        requests.post(url=self.webhook, json={'msgtype': 'text',
                                              'text': {'content': content}})


class Monitor(object):
    # 监控登陆用户
    # ip白名单
    IP_Whitelist = ['localhost']

    # 监控流量
    # 服务器网速，单位为MBPS，详见服务器配置
    Control_Server_Network = 1
    # 上一阶段服务器上行的总数据(默认)
    Last_All_Server_Network = 0
    # 流量阀值(90%)
    Server_Network_Limit = 0.90
    # 监控磁盘使用率
    # 磁盘使用阀值(90%),这里是百分比
    Disk_Usage_Limit = 90

    # 监控Mem使用率
    # Mem使用阀值(90%),这里是百分比
    Mem_Usage_Limit = 90

    # 监控Cpu使用率
    # Cpu使用阀值(90%),这里是百分比
    Cpu_Usage_Limit = 101

    def __init__(self):
        self.dingding = DingDingNotification()

    # 获取用户登陆信息
    def logging_user_info(self):

        logging_user_info_list = []
        logging_user = psutil.users()
        for user in logging_user:
            logging_user_info_dict = {}
            if user.host not in self.IP_Whitelist:
                # 登陆用户的名字
                logging_user_info_dict["name"] = user.name
                # 登陆用户的终端类型
                logging_user_info_dict["terminal"] = user.terminal
                # 登陆用户的ip
                logging_user_info_dict["host"] = user.host
                # 登陆用户的登陆时间
                fmt_time = datetime.fromtimestamp(user.started).strftime('%Y-%m-%d %H:%M:%S.%f')
                logging_user_info_dict["started"] = fmt_time

                logging_user_info_list.append(logging_user_info_dict)

        return logging_user_info_list

    def disk_to_GB(self, size):
        res = size / 1024 / 1024 / 1024

        return '{}GB'.format(res)

    # 获取硬盘使用信息
    def disk_info(self):
        disk_info_dict = {}

        disk = psutil.disk_usage('/')
        # 硬盘总量
        disk_info_dict["total"] = self.disk_to_GB(disk.total)
        # 硬盘使用量
        disk_info_dict["used"] = self.disk_to_GB(disk.used)
        # 硬盘剩余量
        disk_info_dict["free"] = self.disk_to_GB(disk.free)
        # 硬盘使用比
        disk_info_dict["percent"] = disk.percent

        return disk_info_dict

    # 获取进程使用的cpu，mem等信息
    def pids_info(self, pid_type):
        pids_info_list = []

        pids = psutil.pids()
        for pid_id in pids:
            pid_info_dict = {}
            pid = psutil.Process(pid_id)
            # 进程用户
            pid_info_dict["username"] = pid.username()
            # 进程名
            pid_info_dict["name"] = pid.name()
            # 进程的bin路径
            # pid_info_dict["exe"] = pid.exe()
            # 进程的工作目录绝对路径
            # pid_info_dict["cwd"] = pid.cwd()
            # 进程状态
            # pid_info_dict["status"] = pid.status()
            # 进程创建时间
            # pid_info_dict["create_time"] = pid.create_time()
            # 进程内存利用率
            pid_info_dict["memory_percent"] = str(pid.memory_percent()) + '%'
            # 进程cpu利用率
            pid_info_dict["cpu_percent"] = str(pid.cpu_percent()) + '%s'

            pids_info_list.append(pid_info_dict)

        if pid_type == 'CPU':

            res = sorted(pids_info_list, key=lambda x: (x['cpu_percent']),
                         reverse=True)
        else:
            res = sorted(pids_info_list, key=lambda x: (x['memory_percent']),
                         reverse=True)
        return res

    # 监控用户登陆
    def monitor_logging_user(self):

        logging_user = psutil.users()
        for user in logging_user:
            if user.host not in self.IP_Whitelist:
                logging_user_info_list = self.logging_user_info()
                msg = 'Unknown logging_user:{}'.format(logging_user_info_list)
                self.dingding.send(msg)
            time.sleep(10)

    # 监控服务器流量
    def monitor_network(self):
        while True:
            if self.Last_All_Server_Network == 0:
                # 第一次初始化网络所发送的总数据
                self.Last_All_Server_Network = psutil.net_io_counters(pernic=False).bytes_sent / 1024.0 / 1024.0 / 2 * 8
            else:
                # 获取网络所发送的总数据
                all_network = psutil.net_io_counters(pernic=False).bytes_sent / 1024.0 / 1024.0 / 2 * 8
                # 目前带宽
                now_network = all_network - self.Last_All_Server_Network
                if now_network >= self.Control_Server_Network * self.Server_Network_Limit:
                    msg = 'Network Total {}M, Used:{}'.format(self.Control_Server_Network,
                                                              now_network)
                    self.dingding.send(msg)
                    self.Last_All_Server_Network = all_network
                    time.sleep(600)
                else:
                    self.Last_All_Server_Network = all_network
            time.sleep(10)

    # 监控磁盘使用
    def monitor_disk(self):
        while True:
            disk_percent = psutil.disk_usage('/').percent
            # 磁盘使用率
            if disk_percent > self.Disk_Usage_Limit:
                disk_info = self.disk_info()
                msg = 'High Disk Usage:{}%, disk info:{}'.format(disk_percent, disk_info)
                self.dingding.send(msg)
                time.sleep(1200)
            time.sleep(10)

    # 监控mem
    def monitor_mem(self):
        while True:
            mem_percent = psutil.virtual_memory().percent
            # 内存使用率
            if mem_percent > self.Mem_Usage_Limit:
                msg = 'High Memory Usage: {}%'.format(mem_percent)
                self.dingding.send(msg)
                time.sleep(600)
            time.sleep(10)

    # 监控cpu
    def monitor_cpu(self):
        while True:
            cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
            # cpu使用率
            if cpu_percent > self.Cpu_Usage_Limit:
                msg = 'High CPU Usage:{}%'.format(cpu_percent)
                self.dingding.send(msg)
                time.sleep(600)
            time.sleep(10)

    def monitor(self):
        print('monitor starting ....')

        agg_td = []
        for task in [self.monitor_cpu,
                     self.monitor_mem,
                     self.monitor_disk]:
            td = threading.Thread(target=task)
            agg_td.append(td)

        for t in agg_td:
            t.start()

        for d in agg_td:
            d.join()

    def main(self):
        self.monitor()


if __name__ == "__main__":
    Monitor().main()