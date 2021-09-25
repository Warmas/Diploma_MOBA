@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -l cur_agent.pth
PAUSE