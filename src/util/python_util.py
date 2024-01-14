import sys
import datetime
import pickle


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
    return f"{file_name_without_ext}_{other_prefix}_{date_str}_{time_str}.txt"


def pickle_object(obj, file_path):
    """
    Pickle (serialize) an object to a file.

    Parameters:
    - obj: The object to pickle.
    - file_path: The file path where the pickled object will be stored.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object pickled and saved to {file_path}")


def unpickle_object(file_path):
    """
    Unpickle (deserialize) an object from a file.

    Parameters:
    - file_path: The file path from which to load the pickled object.

    Returns:
    - The unpickled object.
    """
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    print(f"Object unpickled from {file_path}")
    return obj
