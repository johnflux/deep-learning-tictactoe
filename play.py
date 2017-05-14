#!/usr/bin/env python3
import numpy as np
import copy
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout


# Call this like:

model = None
callbacks = []

def makeModel():
    global model, callbacks
    if model != None:
        return
    inputs = keras.layers.Input(shape=(2,3,3))

    output = Flatten()(inputs)
    output = Dense(100, activation='relu')(output)
    output = Dense(50, activation='relu')(output)
    output = Dense(20, activation='relu')(output)
    output = Dense(1, activation='relu')(output)
    print(output)

    model = Model(inputs=inputs, outputs=output)

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
        embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
    checkpointCallback = keras.callbacks.ModelCheckpoint(
        'model_running.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.0001)
    callbacks = [tbCallBack, checkpointCallback, reduce_lr]

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    from keras.models import load_model
    #model = load_weights('model_running.h5')

boardgames = []
whowon = []

def train():
    global model, boardgames, whowon
    makeModel()
    print("Boardgames is:", np.array(boardgames).shape, "whowon:", np.array(whowon).shape)
    model.fit(np.array(boardgames), np.array(whowon), epochs=10, validation_split=0.2, shuffle=True,
              verbose=1, callbacks=callbacks)

# board[0,:,:] is for computer player.  0 if there's no piece and 1 if there is
# board[1,:,:] is for other player.     0 if there's no piece and 1 if there is
current_game_boards = []

def find_next_best_move(board, player):
    global model
    makeModel()
    best_prob_to_win = -1
    if player == 1:
        best_prob_to_win = 2
    best_x = 0
    best_y = 0
    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                # Nobody has played in this position.
                # Let's play and see how good the board looks for us
                board[0, x, y] = 1
                prob_to_win = model.predict(np.array([board]), batch_size=1, verbose=1)[0]
                board[0, x, y] = 0
                if ((player == 0 and prob_to_win > best_prob_to_win) or
                          (player == 1 and prob_to_win < best_prob_to_win)):
                    best_x = x
                    best_y = y
                    best_prob_to_win = prob_to_win
    print("Best move is", best_x, best_y, "with probability to win: ", prob_to_win)
    return best_x, best_y

def remember_game_board(board):
    global current_game_boards
    current_game_boards.append(board)

# whowon_  should be 1 if computer, 0 if person, 0.5 if tie
def notify_new_game(whowon_):
    global boardgames, whowon, current_game_boards
    boardgames += current_game_boards
    whowon += (np.ones(len(current_game_boards)) * whowon_).tolist()
    current_game_boards = []
    train()

def get_valid_moves(board):
    valid_moves = []
    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                valid_moves.append((x,y))
    return valid_moves

def get_random_move(board):
    valid_moves = get_valid_moves(board)
    return valid_moves[np.random.randint(len(valid_moves))]

def has_won(board, player):
    p = player
    if ((board[p,0,0] and board[p,1,1] and board[p,2,2]) or
        (board[p,2,0] and board[p,1,1] and board[p,0,2])):
        return True
    for x in range(3):
        if ((board[p,x,0] and board[p,x,1] and board[p,x,2]) or
            (board[p,0,x] and board[p,1,x] and board[p,2,x])):
            return True
    return False

def is_board_full(board):
    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                return False
    return True

def playGame():
    if is_board_full(board):
        notify_new_game()

def playAgainstSelfRandomly():
    while True:
        player_who_won = playAgainstSelfRandomly_()
        notify_new_game(player_who_won)
        printBoard(board)
        print("Score:", player_who_won)
        print()


def printBoard(board):
    for x in range(3):
        for y in range(3):
            if board[0,x,y]:
                print('X', end='')
            elif board[1,x,y]:
                print('O', end='')
            else:
                print('.', end='')
        print()

# Return 0.5 if tie, 1 if computer player won, 0 if we lost
def playAgainstSelfRandomly_():
    board = np.zeros((2, 3, 3))
    # board[0,:,:] is for computer player.  0 if there's no piece and 1 if there is
    # board[1,:,:] is for other player.     0 if there's no piece and 1 if there is
    player = 0
    while True:
        if has_won(board, 0):
            return 1
        if has_won(board, 1):
            return 0
        if is_board_full(board):
            return 0.5
        #x, y = find_next_best_move(board, player)
        x,y = get_random_move(board)
        board[player, x, y] = 1
        remember_game_board(board)
        if player == 0:
            player = 1
        else:
            player = 0
        printBoard(board)
        print()

if __name__ == "__main__":
    print("Hello!")
    playAgainstSelfRandomly()



