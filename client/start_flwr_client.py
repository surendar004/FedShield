import sys
from client.fed_client import start_flwr_client

if __name__ == '__main__':
    client_id = sys.argv[1] if len(sys.argv) > 1 else 'client1'
    start_flwr_client(client_id)
