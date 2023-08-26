import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] - %(filename)s: line %(lineno)d - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler('logs/log.log'), logging.StreamHandler()]
# )

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(filename)s: line %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/log.log')]
)