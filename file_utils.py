import os
import os.path
from os import path


def check_directory_exist(directory=""):
    result = False
    # ref: https://www.guru99.com/python-check-if-file-exists.html
    result = path.exists(directory)
    return result


# ref: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
def create_directory_if_not_exist(directory=""):
    result = False
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            result = True
        except OSError as os_error:
            print("OS Erorr: {}".format(str(os_error)))

    return result
