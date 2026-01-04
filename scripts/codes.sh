#!/bin/bash

SESSION="GuajiraBot"

tmux has-session -t "$SESSION" 2>/dev/null
if [ $? != 0 ]; then
    # Ventana 1
    tmux new-session -d -s "$SESSION" -n editor

    # Crear 4 panes (2x2)
    tmux split-window -t "$SESSION:editor" -h
    tmux split-window -t "$SESSION:editor.0" -v
    tmux split-window -t "$SESSION:editor.1" -v

    # Ejecutar SOLO conda deactivate en cada pane (zsh login shell)
    #tmux send-keys -t "$SESSION:editor.0" "zsh -lc 'conda deactivate'" C-m
    
    # Run update scheduler: uv run python src/scheduler/update_scheduler.py --run-now
    #tmux send-keys -t "$SESSION:editor.1" "zsh -lc 'conda deactivate'" C-m
    
    # Run htop
    tmux send-keys -t "$SESSION:editor.2" "zsh -lc 'htop'" C-m
    
    # Free console
    #tmux send-keys -t "$SESSION:editor.3" "zsh -lc 'htop'" C-m
fi

tmux attach -t "$SESSION"
