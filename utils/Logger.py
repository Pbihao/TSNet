import sys


class Logger(object):
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout
        self.is_logger = True

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
