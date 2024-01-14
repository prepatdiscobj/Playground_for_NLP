import sys
import datetime


def setup_output_loggin(err_file_name="err.txt", output_file_name="output.txt"):
    stdout_file = open(output_file_name, "w")
    stderr_file = open(err_file_name, "w")
    sys.stdout = stdout_file
    sys.stderr = stderr_file

    return stdout_file, stderr_file


def get_latest_file_name(filename_prefix, other_prefix=""):
    file_name_without_ext = filename_prefix.split(".")[0]
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    return f"{file_name_without_ext}_{other_prefix}_{date_str}_{time_str}"
