import sys


class Log(object):
  def __init__(self, *args):
    self.f = open(*args)
    sys.stdout = self

  def write(self, data):
    self.f.write(data)
    sys.__stdout__.write(data)

  def flush(self):
    pass
