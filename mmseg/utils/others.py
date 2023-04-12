import os

def parse_path(path):
    """
    :param path: A file path
    :return: dir, filename, extension
    """
    paths = os.path.split(path)
    dir = os.path.sep.join(paths[:-1])
    filename, extension = os.path.splitext(paths[-1])
    return dir, filename, extension