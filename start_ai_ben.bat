@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -w final_agent.pth
PAUSE