@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Adam -l cur_agent.pth
PAUSE