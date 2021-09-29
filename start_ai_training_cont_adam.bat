@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Adam -t 1 -w cont_agent.pth -o cont_optimizer.pth
PAUSE