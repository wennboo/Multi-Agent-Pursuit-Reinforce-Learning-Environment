### Multi-Agent Pursuit Reinforce Learning Environment 面向强化学习的多智能体追逃环境
The Reinforce Learning Environment on the multi-agent pursuit problem in a two-dimensional space.
The environment depends on tensorflow, python 3.x, gym and Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm, where each agent has the non-holonomic dynamics shown below:

	x = x + v * cos(theta) * (delta)t
	y = y + v * sin(theta) * (delta)t
	theta  = theta * w * (delta)t

本项目基于 tensorflow及python3.X编写了基于强化学习的多智能体追逃训练环境，并做了简单的测试，假定每个智能体有非完整动态。


#### Env Set 配置安装
python-3.6
tensorflow-1.8
matplotlib
numpy
gym

pycharm需要配置:Setting -> Tools ->Python Scientific ->取消勾选Show plots in tool windows
	
#### Visualization 可视化相关事项
The visualization is limited to single episode.
可视化目前仅只支持一个回合场景下的可视化,通常可以用来检测训练模型的效果
可视化的播放速度是可以在参数那边通过调整一帧的播放时间调整



#### Run
``pursuit_env`` is the Env.
``train.ty`` is the main.

Configure the number of agents and related parameters Before run the main code ``train.ty``, 
配置好智能体数量和相关参数,(障碍物暂时不支持体积可视化,只会产生一个点),运行train.py即可训练.所有参数均可在train.py的头部配置




#### Env环境
环境整个轮廓是个矩形，智能体在矩形环境中完成追捕任务。如果智能体越过环境边界，智能体会自动修正到边界位置。环境的尺寸可以在参数中设置。

#### Test测试
There is a trained model with four pursuers and one evaders. For run this model, you should set load_model as true, and run ``train.ty``
环境中有一个训练好的模型，场景是四个智能体追一个智能体。将参数的load_model设置为True运行Train.py即可

