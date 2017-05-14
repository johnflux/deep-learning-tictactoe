all: run_computer
#all: run_interactive

run_computer:
	./play.py

run_interactive:
	@echo '"make vi" loads the files into vim'
	./deep-tic.py

vi:
	vim README Makefile deep-tic.py play.py
