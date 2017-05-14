all: run

run:
	@echo '"make vi" loads the files into vim'
	./deep-tic.py

vi:
	vim README Makefile deep-tic.py play.py
