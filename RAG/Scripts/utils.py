import datetime
import logging
import os
import pytz


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def generate_time_stamp():
    time_stamp = datetime.datetime.now(pytz.timezone("US/Central"))
    time_stamp = time_stamp.strftime("%m_%d_%y_%H_%M_%S")
    return time_stamp


def create_logs_dir(logs_dir="Logs"):
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = "/".join(parent_dir.split("/")[:-1])
    logs_dir = os.path.join(parent_dir, logs_dir)
    create_directory(directory=logs_dir)
    return logs_dir


logs_dir = create_logs_dir(logs_dir="Logs")
time_stamp = generate_time_stamp()
log_filename = "{}/RAG_{}.log".format(logs_dir, time_stamp)

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print("Log Filename: {}".format(log_filename))
print("-------------------------------------")
