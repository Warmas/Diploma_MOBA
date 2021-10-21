@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -w other/checkpoint_agent_1050_1350.pth
PAUSE