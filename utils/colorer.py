#!/usr/bin/env python

import logging





def add_coloring_to_emit_windows(fn):

    def _out_handle(self):
        import ctypes
        return ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)

    out_handle = property(_out_handle)

    def _set_color(self, code):
        import ctypes

        self.STD_OUTPUT_HANDLE = -11
        hdl = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
        ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code)

    setattr(logging.StreamHandler, '_set_color', _set_color)

    def new(*args):
        FOREGROUND_BLUE = 0x0001
        FOREGROUND_GREEN = 0x0002
        FOREGROUND_RED = 0x0004
        FOREGROUND_INTENSITY = 0x0008
        FOREGROUND_WHITE = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

        STD_INPUT_HANDLE = -10
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12


        FOREGROUND_BLACK = 0x0000
        FOREGROUND_BLUE = 0x0001
        FOREGROUND_GREEN = 0x0002
        FOREGROUND_CYAN = 0x0003
        FOREGROUND_RED = 0x0004
        FOREGROUND_MAGENTA = 0x0005
        FOREGROUND_YELLOW = 0x0006
        FOREGROUND_GREY = 0x0007
        FOREGROUND_INTENSITY = 0x0008

        BACKGROUND_BLACK = 0x0000
        BACKGROUND_BLUE = 0x0010
        BACKGROUND_GREEN = 0x0020
        BACKGROUND_CYAN = 0x0030
        BACKGROUND_RED = 0x0040
        BACKGROUND_MAGENTA = 0x0050
        BACKGROUND_YELLOW = 0x0060
        BACKGROUND_GREY = 0x0070
        BACKGROUND_INTENSITY = 0x0080

        levelno = args[1].levelno
        if (levelno >= 50):
            color = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
        elif (levelno >= 40):
            color = FOREGROUND_RED | FOREGROUND_INTENSITY
        elif (levelno >= 30):
            color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
        elif (levelno >= 20):
            color = FOREGROUND_GREEN
        elif (levelno >= 10):
            color = FOREGROUND_MAGENTA
        else:
            color = FOREGROUND_WHITE
        args[0]._set_color(color)

        ret = fn(*args)
        args[0]._set_color(FOREGROUND_WHITE)

        return ret

    return new


def add_coloring_to_emit_ansi(fn):

    def new(*args):
        levelno = args[1].levelno
        if (levelno >= 50):
            color = '\x1b[31m'
        elif (levelno >= 40):
            color = '\x1b[31m'
        elif (levelno >= 30):
            color = '\x1b[33m'
        elif (levelno >= 20):
            color = '\x1b[32m'
        elif (levelno >= 10):
            color = '\x1b[35m'
        else:
            color = '\x1b[0m'
        args[1].msg = color + args[1].msg + '\x1b[0m'

        return fn(*args)

    return new


import platform

if platform.system() == 'Windows':

    logging.StreamHandler.emit = add_coloring_to_emit_windows(logging.StreamHandler.emit)
else:

    logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)




'''
1.把下面代码复制到一个.py文件中
2.将文件放到项目的utils包中(或者项目其他路径)
3.在配置文件中导入这个py文件。比如在Django的setting.py文件中配置日志信息时,可以先导入这个文件
之后运行代码,控制台会根据日志等级显示不同的颜色
pycharm版本最好是17版本以上的
'''