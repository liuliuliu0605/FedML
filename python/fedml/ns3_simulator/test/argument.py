import argparse

parser = argparse.ArgumentParser(prog='ns3 simulator')

parser.add_argument('--pattern', type=str, default='pfl', help='communication pattern bettween pses')
parser.add_argument('--model_size', type=int, default=10000, help='The size of model parameters (Bytes)')
parser.add_argument('--ps_num', type=int, default=9, help='The number of parameter servers')
parser.add_argument('--client_num', type=int, default=10, help='The number of clients in each ps')
parser.add_argument('--topo_name', type=str, default='complete', help='The topology formed by pses')
parser.add_argument('--group_comm_round', type=int, default=1, help='The number of local aggregations')
parser.add_argument('--mix_comm_round', type=int, default=1, help='The number of mixing times')
parser.add_argument('--underlay', type=str, default='geantdistance', help='The underlay topology')
parser.add_argument('--access_link_capacity', type=float, default=1.0e+7, help='access link capacity')
parser.add_argument('--core_link_capacity', type=float, default=1.0e+10, help='core link capacity')
parser.add_argument('--fastfoward', action="store_true")
parser.add_argument('--disable_optimization', action="store_true")