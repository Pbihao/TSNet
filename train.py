import os

from models.TSNet import TSNet
from main import args
import sys
from utils.Logger import Logger


def open_log(log_path=None):
    if log_path is None:
        log_path = os.path.join(os.getcwd(), 'snapshots', 'log.txt')
    log_dir = os.path.split(log_path)[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_path)


def close_log():
    if hasattr(sys.stdout, 'is_logger'):
        del sys.stdout


def train():
    open_log()

    print("dafsf")
    print("sdafdadfsafdsafa")

    close_log()



if __name__ == "__main__":
    train()
    print("test")
