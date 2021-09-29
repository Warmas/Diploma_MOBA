@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -t 1 -l temp_agent.pth -w cont_agent.pth -o cont_optimizer.pth
PAUSE