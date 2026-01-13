# 怎么启动
在根目录创建一个.env文件，内容如下：

OPENAI_API_KEY=你的API KEY

OPENAI_API_BASE=你的base_URL

MODEL_NAME=你的模型名字
# 启动
选一个合适的地方，cmd,新建一个环境，python -m venv 218agent

激活环境：218agent\Scripts\activate

安装依赖：pip install -r requirements.txt

直接streamlitt run gui.py,这个是有图形界面的

python -m src.main,这个是纯CMD的启动.


#Todo
我现在有个想法，我想加入在线markdown笔记本功能，开放在服务器某一个端口，我想让我这个服务和我8218和我这个新笔记能互通，你懂我意思吗，同时我8218端口这个生成的内容也可以能写到我的笔记本里