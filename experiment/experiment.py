import argparse
import datetime
import json
import logging
import os
import subprocess
from logging import handlers

logger = logging.getLogger('experiment')




class experiment:
    '''
    计算元训练测试损失？
    Class to create directory and other meta information to store experiment results.
    此类用于创建目录和其他元信息来存储实验结果
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
    #  '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '_
    #  _lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__
    #  ', '__str__', '__subclasshook__', '__weakref__', 'add_result', 'command_
    #  args', 'dir', 'full_path', 'get_json', 'is_jsonable', 'name', 'params', 'path', 'results', 'root_folder', '
    #  store_json']

    '''

    def __init__(self, name, args, output_dir="../", commit_changes=False):
        import sys
        self.command_args = "python " + " ".join(sys.argv)
        self.run_cwd = os.getcwd()
        self.run_code_dir = os.path.basename(self.run_cwd)
        if commit_changes:
            try:
                self.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")


                subprocess.check_output(['git', 'add', '-u'])
                subprocess.check_output(['git', 'add', '-A'])

                subprocess.check_output(['git', 'commit', '-m', 'running experiment ' + name])
                self.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
            except:
                subprocess.check_output(['git', 'init'])
                subprocess.check_output(['git', 'add', '-A'])
                subprocess.check_output(['git', 'commit', '-m', 'running experiment ' + name])
                self.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")


        if not args is None:
            self.name = name
            self.params = vars(args)
            print(self.params)
            self.results = {}
            self.dir = output_dir

            root_folder = datetime.datetime.now().strftime("%d%B%Y")

            if not os.path.exists(output_dir + root_folder):
                try:
                    os.makedirs(output_dir + root_folder)
                except:
                    assert (os.path.exists(output_dir + root_folder))

            self.root_folder = output_dir + root_folder
            self.full_path = self.root_folder + "/" + self.name

            ver = 0

            while os.path.exists(self.full_path + "_" + str(ver)):
                ver += 1
            try:
                os.makedirs(self.full_path + "_" + str(ver))
            except:
                os.makedirs(self.full_path + "_" + str(ver) + "_race_condition")
            self.path = self.full_path + "_" + str(ver) + "/"

            self.results["Temp Results"] = [[1, 2, 3, 4], [5, 6, 2, 6]]
            fh = logging.FileHandler(self.path + "log.txt")

            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
            logger.addHandler(fh)

            ch = logging.handlers.logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)

            logger.info("Run code dir = %s", self.run_code_dir)
            logger.info("Run cwd = %s", self.run_cwd)
            logger.info("Command args = %s", self.command_args)

            self.store_json()


    def is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False

    def add_result(self, key, value):
        assert (self.is_jsonable(key))
        assert (self.is_jsonable(value))
        self.results[key] = value

    def store_json(self):
        with open(self.path + "metadata.json", 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4, separators=(',', ': '), sort_keys=True)

            outfile.write("")

    def get_json(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iCarl2.0')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')



    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs2', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lrs', type=float, nargs='+', default=[0.00001],
                        help='learning rate (default: 2.0)')
    parser.add_argument('--decays', type=float, nargs='+', default=[0.99, 0.97, 0.95],
                        help='learning rate (default: 2.0)')


    args = parser.parse_args()
    e = experiment("TestExperiment", args, "../../")
    e.add_result("Test Key", "Test Result")
    e.store_json()
