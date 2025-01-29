import argparse
import json
import tensorflow as tf

def rotate_point(angle, point):

    theta = tf.constant(angle)
    points = tf.constant(point)

    rotation_matrix = [[tf.cos(theta), -tf.sin(theta)],
                       [tf.sin(theta), tf.cos(theta)]]
    rotated_points = tf.matmul(points, rotation_matrix)
    return rotated_points.numpy()

def solve_linear_system(first_linear, second_linear):

    A = tf.constant(first_linear)
    B = tf.constant(second_linear)
    solution = tf.linalg.solve(A, B)
    return solution.numpy()

def main():
    # print(rotate_point(3.14, [[1.0, 5.0]]))

    # Układ równań: 2x + 3y = 8,  x -  y = 2
    # print(solve_linear_system([[2.0, 3.0], [1.0, -1.0]], [[8.0], [2.0]]))
    # print(solve_linear_system([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]],
    # [[9.0], [8.0], [7.0]]))

    #poprawne
    #python TenorsMatrix.py '[[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]]' '[[9.0], [8.0], [7.0]]'
    #niepoprawne
    #python TenorsMatrix.py '[[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]]' '[[9.0], [8.0], [7.0], [3.0]]'

    # Ustawienia argumentów
    parser = argparse.ArgumentParser(description='Rozwiąż układ równań liniowych.')
    parser.add_argument('funkcje', type=str,
                        help='Współczynniki równań w formacie: [[1.0, 2.0], [3.0, 4.0]].')
    parser.add_argument('wyniki', type=str,
                        help='Wektor wyników równań w formacie: [[5.0], [6.0]].')

    args = parser.parse_args()

    first_linear = json.loads(args.funkcje)
    second_linear = json.loads(args.wyniki)

    if(len(first_linear) == len(second_linear)):
        solution = solve_linear_system(first_linear, second_linear)
        print("Rozwiązanie:", solution)
    else:
        print("Nie poprawne funkcje lub wyniki")

if __name__ == '__main__':
    main()