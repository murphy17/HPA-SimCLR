import os
from .util import bash

def start_tensorboard(login_node, tmux_name='tensorboard', logging_dir=None):
    if logging_dir is None:
        logging_dir = os.getcwd() + '/lightning_logs'
    logging_dir = logging_dir
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/tensorboard.sh'

    bash(f'chmod +x {script_path}')
    bash(f'ssh {login_node} \'tmux kill-session -t {tmux_name}; tmux new-session -s {tmux_name} -d srun --resv-ports=1 --pty bash -i -c "{script_path} {logging_dir}"\'')