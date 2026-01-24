


class TicTacToe:
    def __init__(self):
        self.board = [[' '] * 3 for _ in range(3)]
        self.current_player = 'X'

    def init_board(self):
        return self.board

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            print("Invalid move! Try again.")

    def check_win(self):
        # Check rows/columns/diagonals for three-in-a-row
        for i in range(3):
            if (
                (all([j[i] == self.current_player for j in self.board]) or 
                 all([self.board[j][i] == self.current_player for j in range(3)]))
            ):
                return True
        if (self.board[0][0] == self.board[1][1] == self.board[2][2] == self.current_player or
            self.board[0][2] == self.board[1][1] == self.board[2][0] == self.current_player):
            return True
        return False

# Example usage:
game = TicTacToe()
print(game.init_board())
for i in range(9):
    print(f"Player {game.current_player}'s turn: ", end="")
    row, col = map(int, input().split())
    game.make_move(row - 1, col - 1)
    if game.check_win():
        print(f"{game.current_player} wins!")
        break
else:
    print("It's a draw.")