import os

def DEBUG_ENABLED():
    return int(os.environ["DEBUG"]) if "DEBUG" in os.environ else 0 