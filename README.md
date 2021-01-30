# exp-data-analysis

辅助处理各类实验数据的脚本

# Install

e.g. 想要保存在path_to_folder/

1. Python 3.8
    - 下载安装: https://www.python.org/downloads/
    - 避免安装最新版本，往前推1-2个版本
2. Pycharm
    - 针对Python优化的IDE，对新手比较"人性化"
    - 下载: https://www.jetbrains.com/pycharm/download/
        - Professional版教育邮箱免费领取
3. Git(非必须)
    - 若不使用Git，可直接从github上下载最新zip包解压
        - https://github.com/Narumikun/exp-data-analysis/archive/master.zip
    - 下载安装: https://git-scm.com/downloads
4. Clone本项目
    - Pycharm: Get from VCS - 输入本项目网址和保存到
    - 或打开命令行工具(Optional):
        - cd some_folder
        - git clone https://github.com/Narumikun/exp-data-analysis.git
5. 虚拟环境Conda or virtualenv(default)
    - venv: 使用Pycharm生成
        - 在Settings-Project-Python Interpreter里新建一个Virtualenv Environment
    - 若需使用conda: 百度Anacanda或Miniconda下载安装，在上述地方选中或新建Conda Environment
6. 安装依赖
    - 打开Pycharm下方的Terminal标签页，确保包含了(venv)或(some-conda-env-name)
        - 可使用pip --version检查python和pip安装位置(path_to_folder/analysis/venv/...)
    - pip install -r requirement.txt
    - pip install scienceplots
7. 编辑脚本处理自己数据
    - main.py: 直接编辑相关流程运行，建议打开scientific mode
    - main.ipynb: 需启动jupyter notebook服务，Pycharm默认运行时自动打开，也可copy相应网址(下方的jupyter标签页)浏览器打开
8. Python-从入门到入土
    - https://www.liaoxuefeng.com/wiki/1016959663602400
    - https://www.runoob.com/python/python-tutorial.html
