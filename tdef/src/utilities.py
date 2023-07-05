import logging


class LoggingSetup():
    def __init__(self):
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/debug.log"),
            logging.StreamHandler()
            ]
        )

LoggingSetup()