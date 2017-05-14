#!/usr/bin/env python3
from __future__ import print_function
#from ansi import *
import os
import random, os, argparse, time
import numpy as np
from random import randint
import sys
from play import *

def getargs():
	parser = argparse.ArgumentParser(description='Page undeformer')
	#parser.add_argument('-v', '--viewfirst', action='store_true', help='View images before training')
	return parser.parse_args()

args = getargs()
tz=0; tx=-1; to=1; # board values: unset, X, and O
bdim = 3          # might we expand from 3x3 to connect four ...?
board=None
num_players=2     # might we have more than 2?
computer=0        # index of computer
human=1           # index of human
prompt_delay=.8

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
def prompt_human_move():
	while True:
		pfl("Your move (uio jkl m,.):")
		inp=raw_input('')
		pf("")
		if keymap.has_key(inp[0]):
			return keymap[inp[0]]      # y,x pair
		pf("I'm sorry, I don't understand that input. Try again.")
		msgdelay()
def init():
	global board, keymap
	seed = 1
	random.seed(seed)
	board=np.zeros((num_players,bdim,bdim))
	keymap=dict()
	keymap['u']=(0,0)
	keymap['i']=(0,1)
	keymap['o']=(0,2)
	keymap['j']=(1,0)
	keymap['k']=(1,1)
	keymap['l']=(1,2)
	keymap['m']=(2,0)
	keymap[',']=(2,1)
	keymap['.']=(2,2)
def get_computer_move(board):
	return find_next_best_move(board, computer)
def available_move(board, y, x):
	if board[0][y][x] or board[1][y][x]:
		return False
	return True
init()
def playgame():
	player=randint(0,1) # initial move is player or computer
	while True:
		pf("\n\n================================================")
		pf(board)
		pf("\n------------------------------------------------\n")
		print_board(board)
		pf("")
		if player == human:
			y,x = prompt_human_move()
		else:
			pf("Please wait for computer's move...")
			y,x = get_computer_move(board)
		if player == human and not available_move(board, y, x):
			pf("Position already played.  Try again.")
			msgdelay()
		else:
			board[player][y][x] = 1
			player = 0 if player else 1

while True:
	playgame()
