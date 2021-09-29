@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -w test_success/checkpoint_agent_50.pth
PAUSE