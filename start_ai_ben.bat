@ECHO OFF
call conda activate diploma
python start_ai_client.py -i AI_Ben -w test_success/learned_shapes_same_color_no_disc.pth
PAUSE