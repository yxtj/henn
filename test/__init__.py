import os
import sys
PROJECT_PATH = os.getcwd()
#SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
SOURCE_PATH = PROJECT_PATH
sys.path.append(SOURCE_PATH)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit()
    if sys.argv[1] == "pl_basis":
        import test_pl_basis
        test_pl_basis.main()
    elif sys.argv[1] == "network":
        import test_network
        test_network.main()
    elif sys.argv[1] == "worker":
        import test_worker
        test_worker.main()
    elif sys.argv[1] == "phen":
        import test_phen
        test_phen.main()
    elif sys.argv[1] == "nnmodel":
        import test_nnmodel
        test_nnmodel.main()
    else:
        print(sys.argv[1], "not supported")
