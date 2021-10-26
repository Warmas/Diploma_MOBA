@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -t 1 -w checkpoint_agent_8000.pth -o checkpoint_optimizer_8000.pth -tb full_test_1 -e 8000
PAUSE