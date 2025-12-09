"""Shared geometric constants for the puzzle setup."""

table_top_z = -0.05 + 0.025

puzzle_center_x = 0.0
puzzle_center_y = -0.60
puzzle_center_z = table_top_z
puzzle_center = (puzzle_center_x, puzzle_center_y, puzzle_center_z)

puzzle_offset = 0.01

upper_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
upper_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
lower_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)
lower_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)

upper_right_square = (
    puzzle_center_x + 0.065,
    puzzle_center_y + 0.065,
    puzzle_center_z + 0.01,
)

lower_right_square = (
    puzzle_center_x + 0.065,
    puzzle_center_y - 0.065,
    puzzle_center_z + 0.01,
)


upper_left_square = (
    puzzle_center_x - 0.065,
    puzzle_center_y + 0.065,
    puzzle_center_z + 0.01,
)

lower_left_square = (
    puzzle_center_x - 0.065,
    puzzle_center_y - 0.065,
    puzzle_center_z + 0.01,
)


off_y = 0.2
off_x = 0.2

trapezoid_translation = (
    puzzle_center_x + 0.2 + off_x,
    puzzle_center_y - off_y,
    table_top_z,
)
infinity_translation = (
    puzzle_center_x + 0.4 + off_x,
    puzzle_center_y - off_y,
    table_top_z,
)
my_piece_translation = (
    puzzle_center_x + 0.2 + off_x,
    puzzle_center_y - off_y + 0.3,
    table_top_z,
)
rectangle_translation = (
    puzzle_center_x + 0.4 + off_x,
    puzzle_center_y - off_y + 0.4,
    table_top_z,
)
cross_translation = (
    puzzle_center_x + 0.2 + off_x,
    puzzle_center_y - off_y + 0.2,
    table_top_z,
)

camera_height = 0.9
camera_translation = (
    puzzle_center_x,
    puzzle_center_y - 0.1,
    puzzle_center_z + camera_height,
)

full_camera_translation = (
    puzzle_center_x,
    puzzle_center_y,
    puzzle_center_z + camera_height,
)

tray_camera_translation = (
    puzzle_center_x + 0.5,
    puzzle_center_y - 0.3,
    puzzle_center_z + 0.8,
)


tray_translations = {
    "rectangle": rectangle_translation,
    "my_piece": my_piece_translation,
    "trapezoid": trapezoid_translation,
    "infinity": infinity_translation,
    "cross": cross_translation,
}
