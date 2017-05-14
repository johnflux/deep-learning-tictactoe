#!/usr/bin/env python3
#from __future__ import print_function
#from ansi import *
import os
import random, os, argparse, time
import numpy as np
from random import randint
import sys
from play import *
import getch
from ansi import *

def getargs():
	parser = argparse.ArgumentParser(description='Page undeformer')
	#parser.add_argument('-v', '--viewfirst', action='store_true', help='View images before training')
	return parser.parse_args()

args = getargs()
tz=0; tx=-1; to=1; # board values: unset, X, and O
bdim = 3          # might we expand from 3x3 to connect four ...?
num_players = 2   # might we have more than 2?
computer = 0      # index of computer (doubles as value for human winning)
human = 1         # index of human (doubles as value for wins/losses)
win_computer = 0  # computer wins, value
win_human = 1     # human wins, value
win_tie = .5      # tie (nobody wins), value
prompt_delay_default = 1 # seconds of delay for messages
prompt_delay = 0         # default, no delay

# Some printing conveniences (with flushing, in case we get elaborate)
def pf(*x, **y):
	print(*x, **y); sys.stdout.flush()
def pfp(*x, **y):
	y.setdefault('sep', ''); print(*x, **y); sys.stdout.flush()
def pfl(*x, **y):
	y.setdefault('end', '')
	print(*x, **y); sys.stdout.flush()
def pfpl(*x, **y):
	y.setdefault('sep', ''); y.setdefault('end', '')
	print(*x, **y); sys.stdout.flush()
def boardval_to_char(board,y,x):
	cval = board[computer][y][x] # computer value
	hval = board[human][y][x]    # human value
	return 'X' if hval else ('O' if cval else ' ')
def print_board(board):
	for y in range(0,bdim):
		pfp(" ", boardval_to_char(board,y,0), " | ",
			boardval_to_char(board,y,1), " | ",
			boardval_to_char(board,y,2))
def msgdelay():
	time.sleep(prompt_delay)
def new_board():
	return np.zeros((num_players,bdim,bdim))
def pause_toggle():
	global prompt_delay
	if not prompt_delay:
		prompt_delay = prompt_delay_default
		pf("Pause turned on")
	else:
		prompt_delay = 0
		pf("Pause turned off")
def prompt_human_move():
	while True:
		pf  ("uio 789   (T)oggle Pause")
		pf  ("jkl 456")
		pfpl("m,. 123   Your move: ")
		inp=getch.getch()
		pf("INPUT:", inp[0])
		if inp[0] in keymap:
			pf("")
			return keymap[inp[0]]      # y,x pair
		elif inp[0].upper() in options:
			pfpl(" ")
			options[inp[0].upper()]()
		else:
			pfp(yel, "I'm sorry, I don't understand that input. Try again.", rst)
			msgdelay()
def init():
	global keymap
	global options
	seed = 1
	random.seed(seed)
	keymap=dict()
	keymap['7']=keymap['u']=(0,0)
	keymap['8']=keymap['i']=(0,1)
	keymap['9']=keymap['o']=(0,2)
	keymap['4']=keymap['j']=(1,0)
	keymap['5']=keymap['k']=(1,1)
	keymap['6']=keymap['l']=(1,2)
	keymap['1']=keymap['m']=(2,0)
	keymap['2']=keymap[',']=(2,1)
	keymap['3']=keymap['.']=(2,2)
	options=dict()
	options['T']=pause_toggle
	pf('opt:', options['T'])
def get_computer_move(board):
	return find_next_best_move(board, human)
def available_move(board, y, x):
	if board[0][y][x] or board[1][y][x]:
		return False
	return True
init()
def playgame():
	board = new_board()
	player = randint(0,1) # initial move is player or computer
	while True:
		pf("\n\n================================================")
		pf(board)
		pf("\n------------------------------------------------\n")
		print_board(board)
		pf("")
		if has_won(board, human):
			pf(bcya, "\nYou won!", rst)
			msgdelay();
			notify_new_game(win_human)
			break
		elif has_won(board, computer):
			pf(bmag, "\nComputer won!", rst)
			msgdelay();
			notify_new_game(win_computer)
			break
		elif is_board_full(board):
			pf("\nThe glass is half empty.  Everybody loses.")
			msgdelay();
			notify_new_game(win_tie)
			break

		if player == human:
			y,x = prompt_human_move()
		else:
			pf("Please wait for computer's move...")
			y,x = get_computer_move(board)
		if player == human and not available_move(board, y, x):
			pf(yel, "Position already played.  Try again.", rst)
			msgdelay()
		else:
			board[player][y][x] = 1
			remember_game_board(board)
			player = 0 if player else 1

while True:
	playgame()
