@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Adam -t 1 -w checkpoint_agent_250.pth -o checkpoint_optimizer_250.pth
PAUSE