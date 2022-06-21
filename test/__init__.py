import os
import sys
PROJECT_PATH = os.getcwd()
#SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
SOURCE_PATH = PROJECT_PATH
sys.path.append(SOURCE_PATH)


if __name__ == "__main__":
    import test_network
    test_network.main()

