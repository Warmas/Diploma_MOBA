@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Adam -t 1 -w final_agent.pth -o final_optimizer.pth -tb Oct21_16-02-48_LAPTOP-KEGS3D41 -e 2
PAUSE